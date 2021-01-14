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

# print('skkwfc')
warnings.filterwarnings('ignore')

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
# print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]
val_path = data_config['valid']

img_size = opt.img_size
conf_thres = opt.conf_thres
nms_thres = opt.nms_thres



with open(val_path, 'r') as f:
    val_len = len(f.readlines()) 


# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
# model.load_weights(opt.weights_path)
model.load_weights('./Checkpoints/confthres_0.9_lr_0.01/weights/25.weights')
#model.apply(weights_init_normal)


if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last = True
) 



val_loader = torch.utils.data.DataLoader(ListDataset(val_path), batch_size = opt.val_batch_size, shuffle = True, num_workers = opt.n_cpu, drop_last = True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

train_loss_history = []
bbox_loss_history = []
conf_loss_history = []
class_loss_history = []

# best_val_precision = 0
best_val_precision = 0.7833333333333333
# best_val_recall = 0
best_val_recall = 0.9428571428571428

# print('kjsc')

for epoch in range(26, opt.epochs):
    # print(epoch)
   
    model.train()
    train_loss = 0 
    bbox_loss = 0
    conf_loss = 0
    class_loss = 0
      
    
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        
        torch.cuda.empty_cache()
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        
        

        optimizer.zero_grad()

        loss = model(imgs, targets)
        train_loss += loss.item() * opt.batch_size
        bbox_loss += (model.losses['x'] + model.losses['y'] + model.losses['w'] + model.losses['h'])*opt.batch_size
        conf_loss += model.losses['conf']
        class_loss += model.losses['cls'] * opt.batch_size
        loss.backward()
        optimizer.step()

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
    
    
    

    train_loss_history.append(train_loss/(len(dataloader)*opt.batch_size))  
    bbox_loss_history.append(bbox_loss / (len(dataloader)*opt.batch_size) )
    class_loss_history.append(class_loss / (len(dataloader)*opt.batch_size) )
    conf_loss_history.append(conf_loss / len(dataloader)) 

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
    
    print('##############################')
    print('Running Validation........')
    total_detections = 0
    n_correct_detect = 0
 
    for batch_i, (_, imgs, targets) in enumerate(val_loader):
        
        # print('###########################')
        # print(batch_i)
        # input()

  
        torch.cuda.empty_cache()
        val_imgs = Variable(imgs.type(Tensor))
        val_targets = Variable(targets.type(Tensor), requires_grad=False)
        t_ids = val_targets[:,:,0] # (B, 1)
        # print([val_targets[0,0,1:]])
        # input()
        
        model.eval()

        with torch.no_grad():

            detections = model(val_imgs) #(B,10647,16)
            # print((detections[1,:,:] < 0).shape)
            # input()
            for i in range(detections.shape[0]):
                # print('############')
                t_labels = [val_targets[i,0,1:]]
                
                true_class_id = int(t_ids[i].item())
                # print('true class id:', true_class_id)
                # input()
                detection = detections[i].unsqueeze(0)
                # print(detection.shape)
                detection = non_max_suppression(detection, 7, conf_thres, nms_thres)[0]
                # print('output form nms',detection)
                # print(detection.shape)
                # print(detection.shape)
                # input()
                if detection is not None:
                    total_detections += detection.shape[0]
                    # print(detection.shape[0])
                    # print('################')
                    # print(detections.shape[0])
                    # print('Total detections:', total_detections)
                    # print(total_detections)
                    # print(detection.shape)
                    # print(detection[[0],:].shape)
                    # input()
                    # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection[0]:
                    #     print(x1)
                    #     input()



                    unique_labels = detection[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    
                    # browse detections and draw bounding boxes
                    for j in range(detection.shape[0]):
                        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection[[j],:]:
                                
                            # print(x1,y1,x2,y2)
                            
                            x1_t = t_labels[0][0] - t_labels[0][2]/2
                            y1_t = t_labels[0][1] - t_labels[0][3]/2

                            x2_t = t_labels[0][0] + t_labels[0][2]/2
                            y2_t = t_labels[0][1] + t_labels[0][3]/2

                            # print(torch.tensor([x1_t, y1_t, x2_t, y2_t]).unsqueeze(0).double())

                            # print(torch.tensor([x1/img_size,y1/img_size,x2/img_size,y2/img_size]).unsqueeze(0).double())

                    
                            # iou = bbox_iou(torch.tensor([(t_labels[0][0] - t_labels[0][2]/2), (t_labels[0][1] - t_labels[0][3]/2), (t_labels[0][0] + t_labels[0][2]/2), (t_labels[0][1] + t_labels[0][3]/2)]).unsqueeze(0).double(), 
                            #                          torch.tensor([x1/416,y1/416,x2/416,y2/416]).unsqueeze(0).double()).item()
                            
                            iou = bbox_iou(torch.tensor([x1_t, y1_t, x2_t, y2_t]).unsqueeze(0).double(), 
                                                    torch.tensor([x1/img_size,y1/img_size,x2/img_size,y2/img_size]).unsqueeze(0).double()).item()
                            # print (cls_pred)
                            # print('cls _ pred', cls_pred)
                            # print(iou)
                            # print(conf)
                            # input()
                            if cls_pred.item() == true_class_id and iou>0.5 and conf > 0.5:
                                n_correct_detect += 1
                                # print('COrrect detect:', n_correct_detect)
                                break   
                        break
                        
        
   
    val_precision = n_correct_detect/total_detections
    val_recall = n_correct_detect/val_len
    val_F1 = val_precision*val_recall*2/(val_precision + val_recall)
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
  

