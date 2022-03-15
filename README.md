# Faster R-CNN Implementation

- GPU v-100
- Custom Dataset : total 6287
- Pretrained Model : vgg16 (fine tuning)
- batch-size : 1
- epoch : 14 (iter : train + val / test)
- mAP:0.5
- optimizer : SGD

### [Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)

1. custom dataset
2. backbone fine-tuning
3. RPN 을 제대로 이해하고 구현해보자!!
4. loss는 어케.?
- iou 랑 L1 loss L2 loss cython으로 구하기!!
anchor 별 비율 앵커를 찍는 방법
anchor 어느 위치에 찍히냐???!!!!!
블로그는 믿지말고 paper를 믿자
cython은 사용하지 않았음
128개 sampling
---
---
이제 좀 끝내자...
거의 다완성!!!!!
데이터 처리를 좀 더 완벽히!!
끝!!!!!
코드정리해서 여기다가 올리기
