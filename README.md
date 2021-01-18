# YOLO

Part - 1 : YOLOv3 (Short for You Only Look Once)

YOLOv3 is an Object Detection algorithm which is state-of-the-art (for now) framework for detecting objects in images/videos either in static or real-time conditions.

Object Detection comes under Computer Vision domain where the model has to find objects inside the image (rather than just classifying the image). There could be multiple objects in an image of varying sizes.

![alt text](https://miro.medium.com/max/875/1*XbOnbcZmc50hyhhTwhD5QA.png)

Check out following article to get a grasp of what YOLO is and how it evolved to YOLOv3 with more technical breakthroughs made in the algorithm
https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088

##About YOLOv3:

YOLOv3 is a 106 layer Fully Convolutional Neural Network with 75 Convolutional Layers (popularly called Darknet-53 architecture). It contains Shortcut, Routing & YOLO Detection layers apart from the Convolutional Layers

It takes use of several other concepts such as Anchor Boxes, Non-Max Suppression, Intersection over Union which needs to be understood along with how the algorithm works. Three scales (13x13, 26x26, 52x52) ensure objects of different sizes get localized.

The YOLO version uploaded in this repository is modified based on the dataset it had to transfer learning to (transfer from COCO dataset). Apart from changes in model detection parameters, loss was modified to penalize False Positives and batches in training were stratified such that all classes are represented equally in a single batch.

It will also store weights and model losses, compute Recall/Precision/F1-score metrics on Validation Set at every epoch of training. Gives you flexibility to start-off training where you left it and also compute the metrics for any weight and dataset(Validation/Testing) provided.

<br>
</br>

# ADDA & YOLO

Part - 2 : ADDA (Adaptive Discriminative Adversarial Adaptation)

Check the following article to understand what ADDA is meant for and how it does what it does.
https://towardsdatascience.com/thoughts-on-adversarial-discriminative-domain-adaptation-f48e3938d518

ADDA is basically meant to tackle lack of data. When there is deficiency in data in target domain, one finds sufficient data in another domain (called source domain) and trains the model on this source domain. The learnings of this model is then shifted to the target domain using the ADDA mechanism (form of Transfer Learning)

The only catch is that both source and target domain needs to be close to each other for adaptation. A discriminator is employed for adaptation whose purpose is to identify from which domain the images are being fed to it (source or target). The target domain's architecture at this point tries to feed representations to the discriminator such that it gets fooled and classifies the image representations as though they were from source domain.

![alt text](https://www.researchgate.net/profile/Wang_Mei24/publication/323142148/figure/fig8/AS:631610605072460@1527599111747/The-Adversarial-discriminative-domain-adaptation-ADDA-architecture-96.png)

The ADDA algorithm was meant to shift domain in the application of Image Classification (and not Object Detection). Thus, this repository deals with modifying ADDA mechanism to overlay YOLOv3 architecture on the CNN architecture and the YOLOv3 detector over the CNN classifier (refer the same in above pic). This is a novel approach as it blends two advanced concepts into one.

To Remember this:- The CNN architecture gets replaced by YOLOv3 Feature Encoder. The Classifier gets replaced by YOLOv3 Object Detector

Now, dividing the process into phases -
1. Pre-training: The source domain dataset is utilized to train the Source YOLOv3 model
2. Adaptation: Target YOLOv3 is initialized with pre-trained weights of Source YOLOv3. The training takes place in batches and epochs with discriminator trying to discriminate image domains and target YOLOv3 encoding image representations to fool the discriminator
3. Testing: After training, one can test the target YOLOv3 learnings on target's testing set to realize how well the adaptation from source domain has happened
