import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as FT
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
from torchsummary import summary
import xml.etree.ElementTree as Et
from typing import Any, Callable, Dict, Optional, Tuple, List
import collections
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import imgaug as ia  # imgaug
from imgaug import augmenters as iaa


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def xml_parser(xml_path):
    xml_path = xml_path
    xml = open(xml_path, "r")
    tree = Et.parse(xml)
    root = tree.getroot()
    size = root.find("size")
    file_name = root.find("filename").text
    object_name = []
    bbox = []
    objects = root.findall("object")
    for _object in objects:
        name = _object.find("name").text
        object_name.append(name)
        bndbox = _object.find("bndbox")
        one_bbox = []
        xmin = bndbox.find("xmin").text
        one_bbox.append(int(float(xmin)))
        ymin = bndbox.find("ymin").text
        one_bbox.append(int(float(ymin)))
        xmax = bndbox.find("xmax").text
        one_bbox.append(int(float(xmax)))
        ymax = bndbox.find("ymax").text
        one_bbox.append(int(float(ymax)))
        bbox.append(one_bbox)
    return file_name, object_name, bbox


def makeBox(voc_im, bbox, objects):
    image = voc_im.copy()
    for i in range(len(objects)):
        cv2.rectangle(image, (int(bbox[i][0]), int(bbox[i][1])), (int(
            bbox[i][2]), int(bbox[i][3])), color=(0, 255, 0), thickness=1)
        cv2.putText(image, objects[i], (int(bbox[i][0]), int(
            bbox[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # 크기, 색, 굵기
    return image


xml_list = os.listdir("VOCdevkit/VOC2012/Annotations")
xml_list.sort()

label_set = set()

for i in range(len(xml_list)):
    xml_path = "VOCdevkit/VOC2012/Annotations/"+str(xml_list[i])
    file_name, object_name, bbox = xml_parser(xml_path)
    for name in object_name:
        label_set.add(name)

label_set = sorted(list(label_set))

label_dic = {}
for i, key in enumerate(label_set):
    label_dic[key] = (i+1)
print(label_dic)


class Pascal_Voc(Dataset):

    def __init__(self, xml_list, len_data):

        self.xml_list = xml_list
        self.len_data = len_data
        self.to_tensor = transforms.ToTensor()
        self.flip = iaa.Fliplr(0.5)
        self.resize = iaa.Resize(
            {"shorter-side": 600, "longer-side": "keep-aspect-ratio"})

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        xml_path = "VOCdevkit/VOC2012/Annotations/"+str(xml_list[idx])

        file_name, object_name, bbox = xml_parser(xml_path)
        image_path = "VOCdevkit/VOC2012/JPEGImages/"+str(file_name)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        image, bbox = self.flip(image=image, bounding_boxes=np.array([bbox]))
        image, bbox = self.resize(image=image, bounding_boxes=bbox)
        bbox = bbox.squeeze(0).tolist()
        image = self.to_tensor(image)

        targets = []
        d = {}
        d['boxes'] = torch.tensor(bbox, device=device)
        d['labels'] = torch.tensor(
            [label_dic[x] for x in object_name], dtype=torch.int64, device=device)
        targets.append(d)

        return image, targets


backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
backbone_out = 512
backbone.out_channels = backbone_out

anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
    sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

resolution = 7
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=resolution, sampling_ratio=2)

box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
    in_channels=backbone_out*(resolution**2), representation_size=4096)
box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    4096, 21)  # 21개 class

model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                                                min_size=600, max_size=1000,
                                                rpn_anchor_generator=anchor_generator,
                                                rpn_pre_nms_top_n_train=6000, rpn_pre_nms_top_n_test=6000,
                                                rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                                                rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7,  rpn_bg_iou_thresh=0.3,
                                                rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                                                box_roi_pool=roi_pooler, box_head=box_head, box_predictor=box_predictor,
                                                box_score_thresh=0.05, box_nms_thresh=0.7, box_detections_per_img=300,
                                                box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                                                box_batch_size_per_image=128, box_positive_fraction=0.25
                                                )
# roi head 있으면 num_class = None으로 함

for param in model.rpn.parameters():
    torch.nn.init.normal_(param, mean=0.0, std=0.01)

for name, param in model.roi_heads.named_parameters():
    if "bbox_pred" in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.001)
    elif "weight" in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    if "bias" in name:
        torch.nn.init.zeros_(param)


writer = SummaryWriter("runs/Faster_RCNN")
%load_ext tensorboard
%tensorboard - -logdir = "runs"
