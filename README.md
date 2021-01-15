# YOLO-ADDA

README

Part - 1 : YOLOv3

YOLOv3 is an Object Detection algorithm which is (for now) state-of-the-art framework for detecting objects in images/videos either in static or real-time conditions.

Object Detection comes under Computer Vision domain where the model has to find objects inside the image (rather than just classifying the image). There could be multiple objects in an image of varying sizes.

![alt text](https://miro.medium.com/max/875/1*XbOnbcZmc50hyhhTwhD5QA.png)

Check out following article to get a grasp of what YOLO is and how it evolved to YOLOv3 with more technical breakthroughs made in the algorithm
https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088



About YOLOv3:

YOLOv3 is a 106 layer Fully Convolutional Neural Network with 75 Convolutional Layers (popularly called Darknet-53 architecture). It contains Shortcut, Routing & YOLO Detection layers apart from the Convolutional Layers

It takes use of several other concepts such as Anchor Boxes, Non-Max Suppression, Intersection over Union which needs to be understood along with how the algorithm works. Three scales (13x13, 26x26, 52x52) ensure objects of different sizes get localized.

<mark>The YOLO version uploaded in this repository is modified based on the dataset it had to transfer learning to (from COCO dataset). Apart from changes in model detection parameters, loss was modified to penalize False Positives and batches in training were stratified such that all classes are represented equally in a single batch<\mark>


