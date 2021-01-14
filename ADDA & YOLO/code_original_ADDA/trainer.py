from __future__ import division

from models_domain import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
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
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--encoder_config_path", type=str, default="config/yolov3_encoder.cfg", help="path to encoder's config file")
parser.add_argument("--yolo_config_path", type=str, default="config/yolov3_yolo.cfg", help="path to yolo layer's config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="config/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="config/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/encoder_weights/", help="directory where encoder checkpoints are saved")
parser.add_argument("--discriminator_dir", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/discriminator_weights/", help="directory where discriminator checkpoints are saved")
parser.add_argument("--best_prec_dir", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/best_prec/", help="directory where model checkpoints corresponding to best validation precision are saved")
parser.add_argument("--best_recall_dir", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/best_recall/", help="directory where model checkpoints corresponding to best validation precision are saved")
parser.add_argument("--discriminator_loss_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/discriminator_loss_history.txt", help="directory to store train loss history")
parser.add_argument("--target_encoder_loss_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/target_encoder_loss_history.txt", help="directory to store bbox loss history")
parser.add_argument("--val_prec_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/val_prec_history.txt", help="directory to store val precision history")
parser.add_argument("--val_recall_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/val_recall_history.txt", help="directory to store val recall history")
parser.add_argument("--val_F1_history", type=str, default="./checkpoints/confthres_0.9_lr_0.01/target_encoder/val_F1_history.txt", help="directory to store val F1 history")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--slope", type=float, default=0.1, help="Slope of Leaky RELU")

opt = parser.parse_args()


cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
source_train_path = data_config["source_train"]
target_train_path = data_config["target_train"]
target_val_path = data_config['target_val']
slope = opt.slope
img_size = opt.img_size
conf_thres = opt.conf_thres
nms_thres = opt.nms_thres
n_classes = int(data_config['classes'])

# Get hyper parameters
hyperparams = parse_model_config(opt.encoder_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# # Initiate model
source_encoder = Darknet(opt.encoder_config_path, True)
target_encoder = Darknet(opt.encoder_config_path, True)
yolo = Darknet(opt.yolo_config_path, False)



discriminator = Discriminator(slope, n_classes)

source_encoder.load_weights(opt.weights_path)
target_encoder.load_weights(opt.weights_path)

if cuda:
    source_encoder = source_encoder.cuda()
    target_encoder = target_encoder.cuda()
    discriminator = discriminator.cuda()

# Get dataloader
source_dataloader = torch.utils.data.DataLoader(
    ListDataset(source_train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
) 

target_dataloader = torch.utils.data.DataLoader(
    ListDataset(target_train_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
)

val_target_dataloader = torch.utils.data.DataLoader(
    ListDataset(target_val_path), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = optim.Adam(
        target_encoder.parameters(),
        lr=2e-4, betas=(.5, .999), weight_decay=2.5e-5)

d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=2e-4, betas=(.5, .999), weight_decay=2.5e-5)

criterion = nn.CrossEntropyLoss()

best_val_precision = 0
best_val_recall = 0

for epoch in range(0, opt.epochs):
    torch.cuda.empty_cache()

    source_encoder.eval()
    target_encoder.train()
    discriminator.train()

    n_iters = min(len(source_dataloader), len(target_dataloader))

    discriminator_loss = 0
    target_encoder_loss = 0

    # Creating iterator from data loader
    source_iter, target_iter = iter(source_dataloader), iter(target_dataloader)

    for iter_i in range(n_iters):
        torch.cuda.empty_cache()
        # source_data and target_data (16,3,416,416)
        # source_target and target_target (16,1,5)
        _, target_data, target_target = target_iter.next()
        _, source_data, source_target = source_iter.next()
        
        # source_data = source_data.permute(0,3,1,2)
        # target_data = target_data.permute(0,3,1,2)
        
        if cuda:
            source_data = source_data.cuda()
            target_data = target_data.cuda()

        
        bs = source_data.size(0)
        
        # Output of source cnn encoder with source domain images as input
        D_input_source = source_encoder(source_data) # (bs, 10647, (num_classes + 5))

        # Output of target cnn encoder with target domain images as input
        D_input_target = target_encoder(target_data) # (bs, 10647, (num_classes + 5))

        # Creating true labels for the source and target domain images to be used for discriminator
        
        #tensor of shape [batch_size] containing all zeros to indicate that the images are from
        # source domain 
        D_target_source = torch.tensor(
            [0] * bs, dtype=torch.long).cuda()

        #tensor of shape [batch_size] containing all ones to indicate that the images are from
        # target domain 
        D_target_target = torch.tensor(
            [1] * bs, dtype=torch.long).cuda()

        # train Discriminator
        D_output_source = discriminator(D_input_source.reshape(bs, -1)) #(bs, 2)

        # Output of discriminator on the target domain images
        D_output_target = discriminator(D_input_target.reshape(bs, -1)) #(bs, 2)

        # Combining the output of discriminator on the source and target domain images
        D_output = torch.cat([D_output_source, D_output_target], dim=0) #(2*bs,2)

        # Combining the true label
        D_target = torch.cat([D_target_source, D_target_target], dim=0) #(2*bs)
        d_loss = criterion(D_output, D_target)
        discriminator_loss += d_loss.item()*2*bs 
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train Target_cnn_encoder, .grad() for the classifier of target cnn was false
        D_input_target = target_encoder(target_data)
        D_output_target = discriminator(D_input_target.reshape(bs, -1))

        # True label given is 0 as it wasnts to trick the discriminator
        loss = criterion(D_output_target, D_target_source)
        target_encoder_loss += loss.item()*bs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: {} ({}/{}) | Discriminator Loss: {} | target_encoder_loss: {}'.format(epoch, iter_i, n_iters, d_loss.item(),loss.item()))
        

    with open(opt.discriminator_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(discriminator_loss/(iter_i*2*opt.batch_size)))

    with open(opt.target_encoder_loss_history, 'a') as f:
        f.write('\n')
        f.write(str(target_encoder_loss/(iter_i*opt.batch_size)))
    
    if epoch % opt.checkpoint_interval == 0:
        target_encoder.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
        torch.save(discriminator, opt.discriminator_dir + '{}.pth'.format(epoch))
    
    # Validation
    print('####################################################################')
    print('Running Validation')
    print('####################################################################')
    with torch.no_grad():

        target_encoder.eval()
        yolo.eval()

        total_detections = 0
        n_correct_detect = 0
        val_len = 0
    
        for batch_i, (_, imgs, targets) in enumerate(val_target_dataloader):
            
            val_len+=imgs.shape[0]
            torch.cuda.empty_cache()
            val_imgs = Variable(imgs.type(Tensor))
            val_targets = Variable(targets.type(Tensor), requires_grad=False)
            t_ids = val_targets[:,:,0] # (B, 1)
            with torch.no_grad():
                detections = yolo(target_encoder(val_imgs)) #(B,10647,12)
                for i in range(detections.shape[0]):
                    t_labels = [val_targets[i,0,1:]]                   
                    true_class_id = int(t_ids[i].item())
                    detection = detections[i].unsqueeze(0)
                    detection = non_max_suppression(detection, n_classes, conf_thres, nms_thres)[0]
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
                                y2_t = t_labels[0][1] + t_labels[0][3]/
                                
                                iou = bbox_iou(torch.tensor([x1_t, y1_t, x2_t, y2_t]).unsqueeze(0).double(), 
                                                        torch.tensor([x1/img_size,y1/img_size,x2/img_size,y2/img_size]).unsqueeze(0).double()).item()
                                if cls_pred.item() == true_class_id and iou>0.5 and conf > 0.5:
                                    n_correct_detect += 1
                                    break   
                            if cls_pred.item() == true_class_id and iou>0.5 and conf > 0.5:
                                break

    if total_detections != 0:
        val_precision = n_correct_detect/total_detections
    else:
        val_precision = 'Nan'

    val_recall = n_correct_detect/val_len
    
    if (val_precision + val_recall) != 0:
        val_F1 = val_precision*val_recall*2/(val_precision + val_recall)
    else:
        val_F1 = 'Nan'
    
    print('Validation (after epoch {}): Precision: {} | Recall: {} | F1 score: {}'.format(epoch, val_precision, val_recall, val_F1))
    print('######################################################################')
    
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
        target_encoder.save_weights(opt.best_prec_dir + '/' + str(epoch) + '.weights')
        with open('./checkpoints/confthres_0.9_lr_0.01/target_encoder/recall_at_best_precision.txt', 'w') as f:
            f.write('Occurs at epoch {}: {}'.format(epoch, str(val_recall)))
            f.close()


    if val_recall > best_val_recall:
        best_val_recall = val_recall
        target_encoder.save_weights(opt.best_recall_dir + '/' + str(epoch) + '.weights')
        with open('./checkpoints/confthres_0.9_lr_0.01/target_encoder/precision_at_best_recall.txt', 'w') as f:
            f.write('Occurs at epoch {}: {}'.format(epoch,str(val_precision)))
            f.close()