from xml.etree.ElementTree import Element, ElementTree
import xml.etree.ElementTree as Et
import os
import sys
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
print(os.path.abspath(__file__))
image_path = "../../../../data/train/JPEGImages/WN04_142.jpg"

image = Image.open(image_path).convert("RGB")

plt.figure(figsize=(25, 20))
plt.imshow(image)
plt.show()
plt.close()


xml_path = "../../../../data/train/Annotations/WN04_142.xml"

print("XML parsing Start\n")
xml = open(xml_path, "r")
tree = Et.parse(xml)
root = tree.getroot()

size = root.find("size")

width = size.find("width").text
height = size.find("height").text
channels = size.find("depth").text

print("Image properties\nwidth : {}\nheight : {}\nchannels : {}\n".format(
    width, height, channels))

objects = root.findall("object")
print("Objects Description")
for _object in objects:
    name = _object.find("name").text
    bndbox = _object.find("bndbox")
    xmin = bndbox.find("xmin").text
    ymin = bndbox.find("ymin").text
    xmax = bndbox.find("xmax").text
    ymax = bndbox.find("ymax").text

    print("class : {}\nxmin : {}\nymin : {}\nxmax : {}\nymax : {}\n".format(
        name, xmin, ymin, xmax, ymax))

print("XML parsing END")


dataset_path = "../../../../data/train/"

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(
    os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))

print("ROOT : {}\n".format(ann_root))
print("DIR : {}\n".format(ann_dir))
print("FILES : {}\n".format(ann_files))


dataset_path = "../../../../data/train/"

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(
    os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))

for xml_file in ann_files:
    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    print("Image properties\nwidth : {}\nheight : {}\nchannels : {}\n".format(
        width, height, channels))

    objects = root.findall("object")
    print("Objects Description")
    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = bndbox.find("xmin").text
        ymin = bndbox.find("ymin").text
        xmax = bndbox.find("xmax").text
        ymax = bndbox.find("ymax").text

        print("class : {}\nxmin : {}\nymin : {}\nxmax : {}\nymax : {}\n".format(
            name, xmin, ymin, xmax, ymax))

    print("XML parsing END")


dataset_path = "../../../../data/train/"

IMAGE_FOLDER = "JPEGImages"
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(
    os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, amg_dir, img_files = next(
    os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

ann_files.sort()
img_files.sort()
temp = 0

for xml_file in ann_files:
    # if temp == 3: break
    # # XML파일와 이미지파일은 이름이 같으므로, 확장자만 맞춰서 찾습니다.
    try:
        if img_files[img_files.index(".".join([xml_file.split(".")[0], "jpg"]))] == None:
            print("stop")
    except ValueError as e:
        print(e)
    img_name = img_files[img_files.index(
        ".".join([xml_file.split(".")[0], "jpg"]))]
    img_file = os.path.join(img_root, img_name)
    image = Image.open(img_file).convert("RGB")
    draw = ImageDraw.Draw(image)

    xml = open(os.path.join(ann_root, xml_file), "r")
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    objects = root.findall("object")

    for _object in objects:
        name = _object.find("name").text
        bndbox = _object.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Box를 그릴 때, 왼쪽 상단 점과, 오른쪽 하단 점의 좌표를 입력으로 주면 됩니다.
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=5)
        draw.text((xmin, ymin), name)
    # image = image.resize((224, 224), Image.LANCZOS)
    image.save('../../../../data/train/AfterImages/' + str(img_name), 'jpeg')
    # plt.figure(figsize=(25, 20))
    # plt.imshow(image)
    # print(type(image))
    # plt.show()
    # plt.close()
    # temp += 1
