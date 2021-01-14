from __future__ import division

from models_domain import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import non_max_suppression
from utils.utils import bbox_iou
import matplotlib.pyplot as plt
import os
import sys
import time
import datetime
import argparse
import numpy as np
from PIL import Image
from skimage.transform import resize
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings
from custom_dataloader_class_7 import train_loader

warnings.filterwarnings('ignore')

torch.manual_seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="./data/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=14, help="size of each image batch")
parser.add_argument("--val_batch_size", type=int, default=20, help="size of each image batch during validation")
parser.add_argument("--encoder_config_path", type=str, default="config/yolov3_encoder.cfg", help="path to encoder's config file")
parser.add_argument("--yolo_config_path", type=str, default="config/yolov3_yolo.cfg", help="path to yolo layer's config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/weights/", help="directory where model checkpoints are saved")
parser.add_argument("--best_prec_dir", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/best_prec", help="directory where model checkpoints corresponding to best validation precision are saved")
parser.add_argument("--best_recall_dir", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/best_recall", help="directory where model checkpoints corresponding to best validation precision are saved")
parser.add_argument("--train_loss_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/train_loss_history.txt", help="directory to store train loss history")
parser.add_argument("--bbox_loss_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/bbox_loss_history.txt", help="directory to store bbox loss history")
parser.add_argument("--conf_loss_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/conf_loss_history.txt", help="directory to store conf loss history")
parser.add_argument("--class_loss_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/class_loss_history.txt", help="directory to store class loss history")
parser.add_argument("--val_prec_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/val_prec_history.txt", help="directory to store val precision history")
parser.add_argument("--val_recall_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/val_recall_history.txt", help="directory to store val recall history")
parser.add_argument("--val_F1_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/source_encoder/val_F1_history.txt", help="directory to store val F1 history")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")

opt = parser.parse_args()
cuda = torch.cuda.is_available() and opt.use_cuda
os.makedirs("checkpoints", exist_ok=True)
classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["source_train"]
val_path = data_config['source_val']
num_classes = int(data_config['classes'])
img_size = opt.img_size
conf_thres = opt.conf_thres
nms_thres = opt.nms_thres

with open(val_path, 'r') as f:
    val_len = len(f.readlines()) 

# Get hyper parameters
hyperparams = parse_model_config(opt.encoder_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
encoder = Darknet(opt.encoder_config_path, True)
yolo = Darknet(opt.yolo_config_path, False)
encoder.load_weights(opt.weights_path)

if cuda:
    encoder = encoder.cuda()
    yolo = yolo.cuda()

encoder.train()
yolo.train()

handgun = []
shuriken = []
usb = []
phone = []
knife = []
hard_disk = []
battery = []

batch_size = int(opt.batch_size)
img_size = 416
max_objects = 1

with open(train_path) as f:
    paths = f.readlines()

    for path in paths:

        if path.lower().find('shuriken') != -1:
            shuriken.append(path)

        if path.lower().find('usb') != -1:
            usb.append(path)
        
        if path.lower().find('phone') != -1:
            phone.append(path)
        
        if path.lower().find('knife') != -1:
            knife.append(path)
        
        if path.lower().find('hard_disk') != -1:
            hard_disk.append(path)

        if path.lower().find('battery') != -1:
            battery.append(path)

        if path.lower().find('handgun') != -1:
            handgun.append(path)

# Remove duplicates 
handgun = list(set(handgun))
shuriken = list(set(shuriken))
usb = list(set(usb))
phone = list(set(phone))
knife = list(set(knife))
hard_disk = list(set(hard_disk))
battery = list(set(battery))

handgun_labels = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in handgun]
shuriken_labels = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in shuriken]
usb_labels = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in usb]
phone_labels = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in phone]
knife_labels = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in knife]
hard_disk_labels = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in hard_disk]
battery_labels = [path.replace('images', 'labels').replace('.jpg', '.txt') for path in battery]
val_loader = torch.utils.data.DataLoader(ListValDataset(val_path), batch_size = opt.val_batch_size, shuffle = True, num_workers = opt.n_cpu, drop_last = True)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()))

train_loss_history = []
bbox_loss_history = []
conf_loss_history = []
class_loss_history = []

best_val_precision = 0
best_val_recall = 0

for epoch in range(100, opt.epochs):
   
    encoder.train()
    train_loss = 0 
    bbox_loss = 0
    conf_loss = 0
    class_loss = 0
         
    for batch_i, (imgs, targets) in enumerate(train_loader(handgun, shuriken, usb, phone, knife, hard_disk, battery, handgun_labels, shuriken_labels, usb_labels, phone_labels, knife_labels, hard_disk_labels, battery_labels, batch_size, num_classes)):

        torch.cuda.empty_cache()
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        optimizer.zero_grad()
        loss = yolo(encoder(imgs, targets), targets)
        train_loss += loss.item() * opt.batch_size
        bbox_loss += (yolo.losses['x'] + yolo.losses['y'] + yolo.losses['w'] + yolo.losses['h'])*opt.batch_size
        conf_loss += yolo.losses['conf']
        class_loss += yolo.losses['cls'] * opt.batch_size
        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                yolo.losses["x"],          
                yolo.losses["y"],
                yolo.losses["w"],
                yolo.losses["h"],
                yolo.losses["conf"],
                yolo.losses["cls"],
                loss.item(),
                yolo.losses["recall"],
                yolo.losses["precision"],
            )
        )
      
    with open(opt.train_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(train_loss/(batch_i*opt.batch_size)))

    with open(opt.bbox_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(bbox_loss / (batch_i*opt.batch_size)))
    
    with open(opt.class_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(class_loss / (batch_i*opt.batch_size) ))
    
    with open(opt.conf_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(conf_loss / batch_i))

    train_loss_history.append(train_loss/(batch_i*opt.batch_size))  
    bbox_loss_history.append(bbox_loss / (batch_i*opt.batch_size) )
    class_loss_history.append(class_loss / (batch_i*opt.batch_size) )
    conf_loss_history.append(conf_loss / batch_i) 

    if epoch % opt.checkpoint_interval == 0:
        encoder.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
    
    print('##############################')
    print('Running Validation........')
    total_detections = 0
    n_correct_detect = 0
 
    for batch_i, (_, imgs, targets) in enumerate(val_loader):
 
        torch.cuda.empty_cache()
        val_imgs = Variable(imgs.type(Tensor))
        val_targets = Variable(targets.type(Tensor), requires_grad=False)
        t_ids = val_targets[:,:,0] # (B, 1)        
        encoder.eval()
        yolo.eval()

        with torch.no_grad():

            detections = yolo(encoder(val_imgs)) #(B,10647,12)
            for i in range(detections.shape[0]):
                t_labels = [val_targets[i,0,1:]]
                true_class_id = int(t_ids[i].item())
                detection = detections[i].unsqueeze(0)
                detection = non_max_suppression(detection, num_classes, conf_thres, nms_thres)[0] # (k, 7)
                if detection is not None:            
                    total_detections += detection.shape[0]
                    unique_labels = detection[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)

                    # browse detections and draw bounding boxes
                    for j in range(detection.shape[0]):
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection[[j],:]:                                                                                      
                            x1_t = t_labels[0][0] - t_labels[0][2]/2
                            y1_t = t_labels[0][1] - t_labels[0][3]/2
                            x2_t = t_labels[0][0] + t_labels[0][2]/2
                            y2_t = t_labels[0][1] + t_labels[0][3]/2                            
                            iou = bbox_iou(torch.tensor([x1_t, y1_t, x2_t, y2_t]).unsqueeze(0).double(), 
                                                    torch.tensor([x1/img_size,y1/img_size,x2/img_size,y2/img_size]).unsqueeze(0).double()).item()
                            
                            if cls_pred.item() == true_class_id and iou>0.5 and conf > 0.5:
                                
                                n_correct_detect += 1
                                break 

                        if cls_pred.item() == true_class_id and iou>0.5 and conf > 0.5:
                            break
                                
    try:
        val_precision = n_correct_detect/total_detections
    except:
        val_precision = 'NA'

    val_recall = n_correct_detect/val_len
    
    try:
        val_F1 = val_precision*val_recall*2/(val_precision + val_recall)
    except:
        val_F1 = 'NA'

    print('Total_detection: ', total_detections)
    print('Correct_detection: ', n_correct_detect)
    print('Total Img files: ', val_len)
    print('Validation (after epoch {}): Precision: {} | Recall: {} | F1 score: {}'.format(epoch, val_precision, val_recall, val_F1))
    print('###################################')
    
    
    with open(opt.val_prec_history,'a') as f:
        f.write('\n')
        f.write(str(val_precision))

    with open(opt.val_recall_history,'a') as f:
        f.write('\n')
        f.write(str(val_recall))

    with open(opt.val_F1_history,'a') as f:
        f.write('\n')
        f.write(str(val_F1))
    

    if val_precision != 'NA':
        if val_precision > best_val_precision:
            best_val_precision = val_precision
            encoder.save_weights(opt.best_prec_dir + '/' + str(epoch) + '.weights')
            with open('./checkpoints/confthres_0.9_lr_0.01/source_encoder/recall_at_best_precision.txt', 'w') as f:
                f.write('Occurs at epoch {}: {}'.format(epoch, str(val_recall)))
                f.close()


    if val_recall > best_val_recall:
        best_val_recall = val_recall
        encoder.save_weights(opt.best_recall_dir + '/' + str(epoch) + '.weights')
        with open('./checkpoints/confthres_0.9_lr_0.01/source_encoder/precision_at_best_recall.txt', 'w') as f:
            f.write('Occurs at epoch {}: {}'.format(epoch,str(val_precision)))
            f.close()
  

