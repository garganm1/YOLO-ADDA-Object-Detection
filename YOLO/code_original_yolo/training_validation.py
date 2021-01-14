# Importing Relevant Libraries
from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.utils import non_max_suppression
from utils.utils import bbox_iou
import os
import sys
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings

warnings.filterwarnings('ignore')

# Arguments to be passed by the user from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="./data/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=18, help="size of each image batch")
parser.add_argument("--val_batch_size", type=int, default=15, help="size of each image batch during validation")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--best_prec_dir", type=str, default="best_prec", help="directory where model checkpoints corresponding to best validation precision are saved")
parser.add_argument("--best_recall_dir", type=str, default="best_recall", help="directory where model checkpoints corresponding to best validation precision are saved")
parser.add_argument("--train_loss_history", type=str, default="train_loss_history", help="directory to store train loss history")
parser.add_argument("--bbox_loss_history", type=str, default="bbox_loss_history", help="directory to store bbox loss history")
parser.add_argument("--conf_loss_history", type=str, default="conf_loss_history", help="directory to store conf loss history")
parser.add_argument("--class_loss_history", type=str, default="class_loss_history", help="directory to store class loss history")
parser.add_argument("--val_prec_history", type=str, default="val_prec_history", help="directory to store val precision history")
parser.add_argument("--val_recall_history", type=str, default="val_recall_history", help="directory to store val recall history")
parser.add_argument("--val_F1_history", type=str, default="val_F1_history", help="directory to store val F1 history")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()

# cuda if available
cuda = torch.cuda.is_available() and opt.use_cuda

# To store checkpoints
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path) # List of classes of banned items

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"] # path for file containing paths for train images
val_path = data_config['valid'] # path for file containing paths for validation images
img_size = opt.img_size # input size of the image to be fed to the model
conf_thres = opt.conf_thres # confidence threshold
nms_thres = opt.nms_thres # Non Max suppression threshold

# number of validation images
with open(val_path, 'r') as f:
    val_len = len(f.readlines()) 

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Instantiate model
model = Darknet(opt.model_config_path)
# load pretrained weights for the model
model.load_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.train()

# Get dataloaders
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last = True
) 
val_loader = torch.utils.data.DataLoader(ListDataset(val_path), batch_size = opt.val_batch_size, shuffle = True, num_workers = opt.n_cpu, drop_last = True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# List to store different losses during tarining
train_loss_history = []
bbox_loss_history = []
conf_loss_history = []
class_loss_history = []

# TO store best validation metrics
best_val_precision = 0
best_val_recall = 0

################ Training #########################
for epoch in range(opt.epochs):
   
    # Train mode
    model.train()

    # Different losses for a epoch
    train_loss = 0 
    bbox_loss = 0
    conf_loss = 0
    class_loss = 0
    
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        
        torch.cuda.empty_cache()
        imgs = Variable(imgs.type(Tensor)) # Input images
        targets = Variable(targets.type(Tensor), requires_grad=False) # True labels  
        optimizer.zero_grad()
        loss = model(imgs, targets)
        train_loss += loss.item() * opt.batch_size 
        bbox_loss += (model.losses['x'] + model.losses['y'] + model.losses['w'] + model.losses['h'])*opt.batch_size
        conf_loss += model.losses['conf']
        class_loss += model.losses['cls'] * opt.batch_size
        loss.backward()
        optimizer.step()

        # Priniting various training metrics
        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],          
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        model.seen += imgs.size(0) # to keep the record of number of images seen

    # Saving training metrics in files
    with open(opt.train_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(train_loss/(len(dataloader)*opt.batch_size)))

    with open(opt.bbox_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(bbox_loss / (len(dataloader)*opt.batch_size)))
    
    with open(opt.class_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(class_loss / (len(dataloader)*opt.batch_size) ))
    
    with open(opt.conf_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(conf_loss / len(dataloader)))
    
    # Appending losses in the corresponding list
    train_loss_history.append(train_loss/(len(dataloader)*opt.batch_size))  
    bbox_loss_history.append(bbox_loss / (len(dataloader)*opt.batch_size) )
    class_loss_history.append(class_loss / (len(dataloader)*opt.batch_size) )
    conf_loss_history.append(conf_loss / len(dataloader)) 

    # Saving the model every checkpoint_interval
    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
    
    ################# Validation #########################
    print('##############################')
    print('Running Validation........')

    total_detections = 0 # Total number of detections in an image
    n_correct_detect = 0 # Total number of correct detections in an iamge
 
    for batch_i, (_, imgs, targets) in enumerate(val_loader):
          
        torch.cuda.empty_cache()
        val_imgs = Variable(imgs.type(Tensor)) # Validation images
        val_targets = Variable(targets.type(Tensor), requires_grad=False) 
        t_ids = val_targets[:,:,0] # class id (B, 1) 
        
        # Eval mode
        model.eval()

        with torch.no_grad():
            detections = model(val_imgs) # Output of the model, total detections (B,10647,16)
            for i in range(detections.shape[0]):
                t_labels = [val_targets[i,0,1:]] # True labels
                true_class_id = int(t_ids[i].item()) # True class id of the image
                detection = detections[i].unsqueeze(0)
                detection = non_max_suppression(detection, 7, conf_thres, nms_thres)[0] # Applying NMS on detections for the image
                if detection is not None:
                    total_detections += detection.shape[0] # Total detections after NMS suppression
                    unique_labels = detection[:, -1].cpu().unique() # Unique obejcts detected
                    n_cls_preds = len(unique_labels) # Number of uniques object detected

                    # browse detections and draw bounding boxes
                    for j in range(detection.shape[0]):
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection[[j],:]:

                            # True coordinates of bounding box
                            x1_t = t_labels[0][0] - t_labels[0][2]/2
                            y1_t = t_labels[0][1] - t_labels[0][3]/2
                            x2_t = t_labels[0][0] + t_labels[0][2]/2
                            y2_t = t_labels[0][1] + t_labels[0][3]/2
                            
                            # IOU between true bbox and predicted bbpx
                            iou = bbox_iou(torch.tensor([x1_t, y1_t, x2_t, y2_t]).unsqueeze(0).double(), 
                                                    torch.tensor([x1/img_size,y1/img_size,x2/img_size,y2/img_size]).unsqueeze(0).double()).item()

                            if cls_pred.item() == true_class_id and iou>0.5 and conf > 0.5:
                                n_correct_detect += 1
                                break   
                        break
                        
        
    # Calcualting Validation metrics
    val_precision = n_correct_detect/total_detections
    val_recall = n_correct_detect/val_len
    val_F1 = val_precision*val_recall*2/(val_precision + val_recall)
    print('Validation (after epoch {}): Precision: {} | Recall: {} | F1 score: {}'.format(epoch, val_precision, val_recall, val_F1))
    print('###################################')
    
    # Saving validation metrics to a file
    with open(opt.val_prec_history,'a') as f:
        f.write('\n')
        f.write(str(val_precision))

    with open(opt.val_recall_history,'a') as f:
        f.write('\n')
        f.write(str(val_recall))

    with open(opt.val_F1_history,'a') as f:
        f.write('\n')
        f.write(str(val_F1))
    
    # Updating best validation metrics
    if val_precision > best_val_precision:
        best_val_precision = val_precision
        model.save_weights(opt.best_prec_dir + '/' + str(epoch) + '.weights')
        with open('./Checkpoints/confthres_0.9_lr_0.01/recall_at_best_precision.txt', 'w') as f:
            f.write('Occurs at epoch {}: {}'.format(epoch, str(val_recall)))
            f.close()
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        model.save_weights(opt.best_recall_dir + '/' + str(epoch) + '.weights')
        with open('./Checkpoints/confthres_0.9_lr_0.01/precision_at_best_recall.txt', 'w') as f:
            f.write('Occurs at epoch {}: {}'.format(epoch,str(val_precision)))
            f.close()
  

