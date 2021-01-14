# Importing relevant libraries
from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import glob

# Configuration file path
config_path='config/yolov3.cfg'
# Trained weights of the model
weights_path='./checkpoints/confthres_0.9_lr_0.01/weights/20.weights'
# Path for the file containing names of the classses
class_path='config/coco.names'
img_size=416 # input size of the image
# Thresholds
conf_thres=0.9
nms_thres=0.0005

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

# Detections in the test image
def detect_image(img):
    # Preprocessing image
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        # Non max suppression
        detections = utils.non_max_suppression(detections, 7, 
                        conf_thres, nms_thres)
    
    return detections[0]

# Name of the class to be dtected in images
cls_name = input('Enter the class name for images: ' )
# Dataset
data_name = input('Enter the dataset: ')
files = glob.glob('./data/' + data_name +'/Images/' + cls_name + '/*.jpg')

# Getting and drwing bbox predictions on each image
for img_path in files:
    n,ext = os.path.splitext(os.path.basename(img_path))
    pic = n+ext
    prev_time = time.time()
    img = Image.open(img_path)
    detections = detect_image(img)

    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print ('Inference Time: %s' % (inference_time))
    # Get bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    img = np.array(img)
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        # browse detections and draw bounding boxes
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
            color = bbox_colors[int(np.where(
                unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle((x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            plt.text(x1, y1, s=classes[int(cls_pred)], 
                    color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})
    plt.axis('off')
    plt.savefig('./data/results/' + cls_name + '/'+pic,        
                    bbox_inches='tight', pad_inches=0.0)
    