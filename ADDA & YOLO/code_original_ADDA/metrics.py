from models_domain import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import glob
from utils.datasets import *
from utils.utils import bbox_iou

encoder_config_path='config/yolov3_encoder.cfg'
yolo_config_path = 'config/yolov3_yolo.cfg'
# weights_path='./checkpoints/confthres_0.9_lr_0.01/source_encoder/weights/121.weights'
weights_path='./checkpoints/confthres_0.9_lr_0.01/target_encoder/encoder_weights/1.weights'
class_path='./config/coco.names'
img_size=416
conf_thres=0.9
# conf_thres=0.97
# nms_thres=0.0005
nms_thres=0.4
# Load model and weights
encoder = Darknet(encoder_config_path, True)
yolo = Darknet(yolo_config_path, False)
encoder.load_weights(weights_path)
encoder.cuda()
yolo.cuda()
encoder.eval()
yolo.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

# cl = []
# for c in classes:
#      cl.append(c.lower())

# classes = cl



total_overall_detections = 0
total_overall_correct_detections = 0  
total_img_files = 0      

def detect_image(img):

#     # scale and pad image
#     ratio = min(img_size/img.size[0], img_size/img.size[1])
# #     print(img_size)
# #     print(img.size)
# #     print(img.size[0])
# #     print(img.size[1])
    
#     imw = round(img.size[0] * ratio) 
#     imh = round(img.size[1] * ratio) 
# #     print(imw, imh)

# #     Pads all the sides accordingly to get 416*416 and converts to Tensor

#     img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
#          transforms.Pad((max(int((imh-imw)/2),0), 
#               max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
#               max(int((imw-imh)/2),0)), (128,128,128)),
#          transforms.ToTensor(),
#          ])
#     # convert image to Tensor
#     image_tensor = img_transforms(img).float()
# #     print(image_tensor.shape)
# #     print(len(image_tensor.shape))
     
#     if image_tensor.shape[0] !=3:
#          image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)

    
    img = np.array(img)

     # Handles images with less than three channels
    while len(img.shape) != 3:
     #     print(img.shape)
     #     plt.imsave('before.jpg', img, cmap = 'gray')
         img = np.expand_dims(img,0)
         img = np.transpose(np.concatenate((img, img, img), 0), (1,2,0))
     #     plt.imsave('after.jpg', img)
     #     print('Check')
     #     input()

    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
     # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
     # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
     # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape
     # Resize and normalize
    input_img = resize(input_img, (*(img_size,img_size), 3), mode='reflect')
     # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
     # As pytorch tensor
    image_tensor = torch.from_numpy(input_img).float()
    
    
    image_tensor = image_tensor.unsqueeze_(0) #(1,3,416,416)
    
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = yolo(encoder(input_img)) #(1,10647,16)

     #    print(detections.shape)
     #    input()
             
        detections = utils.non_max_suppression(detections, 7, 
                        conf_thres, nms_thres)[0]
     
    return detections

# load image and get detections
def metric_calc_perclass(path, true_class_id, c):
     
     img_files = glob.glob(path+'/*.jpg')
     # print(c)
     # print(len(img_files))
          
     global total_overall_detections  
     global total_overall_correct_detections
     global total_img_files
     l = 0
     total_detections  = 0
     n_correct_detect = 0
     for img in img_files:
          # txt = img.strip()[:16]+'labels'+img.strip()[22:-3]+'txt' # For training
          # print(img[27:])
          # input()
          # print(img)
          img = img.strip()
          img_file = img

          if cls_name == 'test':
               a = img[:12]
               b = 'Labels/'
               txt = a + b + img[len(a+b):-3] + 'txt'
               # txt = img.strip()[:12]+'Labels/'+ c + '/' + img.strip()[27:-3]+'txt' # For test

          if cls_name == 'train':
               a = img[:13]
               b = 'Labels/'
               txt = a + b + img[len(a+b):-3] + 'txt'
               # txt = img.strip()[:13]+'Labels/'+ c + '/' + img.strip()[28:-3]+'txt' # For train
          
          if cls_name == 'val':
               a = img[:11]
               b = 'Labels/'
               txt = a + b + img[len(a+b):-3] + 'txt'

          t_labels = []
          
          with open(txt) as f:
               lines = f.readlines()
          for line in lines:
               txt = line.strip().split()
               t_labels.append(list(map(float, txt[1:])))
               # print(t_labels)
               # input()


          # ############################
          # print(img)
          img = Image.open(img.strip())
          detections = detect_image(img)
               # input()

          # total_detections += detections.shape[0]
          img = np.array(img)
          
          # pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
          # pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
          # unpad_h = img_size - pad_y
          # unpad_w = img_size - pad_x
         
          if detections is not None:
               total_detections += detections.shape[0]
               unique_labels = detections[:, -1].cpu().unique()
               n_cls_preds = len(unique_labels)
               # print(detections.shape)
               # input()
               # browse detections and draw bounding boxes
               for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    
                    x1_t = t_labels[0][0] - t_labels[0][2]/2
                    y1_t = t_labels[0][1] - t_labels[0][3]/2

                    x2_t = t_labels[0][0] + t_labels[0][2]/2
                    y2_t = t_labels[0][1] + t_labels[0][3]/2

               
                    # iou = bbox_iou(torch.tensor([(t_labels[0][0] - t_labels[0][2]/2), (t_labels[0][1] - t_labels[0][3]/2), (t_labels[0][0] + t_labels[0][2]/2), (t_labels[0][1] + t_labels[0][3]/2)]).unsqueeze(0).double(), 
                    #                          torch.tensor([x1/416,y1/416,x2/416,y2/416]).unsqueeze(0).double()).item()
                    
                    iou = bbox_iou(torch.tensor([x1_t, y1_t, x2_t, y2_t]).unsqueeze(0).double(), 
                                             torch.tensor([x1/img_size,y1/img_size,x2/img_size,y2/img_size]).unsqueeze(0).double()).item()
                    # print (cls_pred)
                    # print(true_class_id)
                    # input()
                    # print(iou)
                    if cls_pred == true_class_id and iou>0.5 and conf > 0.5:
                         # if c=='Hard_disk':
                         #      print(img_file)
                         #      print(x1_t, y1_t, x2_t, y2_t)
                         #      print(x1/img_size,y1/img_size,x2/img_size,y2/img_size)
                         #      print(iou)

                         n_correct_detect += 1

                         break    
               
          total_img_files += len(t_labels)  
     
     total_overall_detections += total_detections
     total_overall_correct_detections += n_correct_detect
     
     # print(n_correct_detect)
     print('####################')
     print(classes[true_class_id])
     print('####################')
     # print(total_detections)
     if  total_detections:
          # print(total_detections)
          # print(n_correct_detect)
          # print(len(img_files))
          precision = n_correct_detect/total_detections
          recall = n_correct_detect / len(img_files)
          print('Recall:',recall)
          print('Precision:',precision)
          try:
               F1 = 2*precision*recall/(precision+recall)
          except:
               F1 = 0
          print('F1 score:', F1)
          

def metric_calc(train_imgs_dir):

     p = Path(train_imgs_dir)
     sub = [x for x in p.iterdir() if x.is_dir()]

     for i in sub:
          
          if cls_name == 'test':
               c = str(i)[17:] # For test
          if cls_name == 'train':
               c = str(i)[18:]
          if cls_name == 'val':
               c = str(i)[16:]
               


          
          true_class_id = classes.index(c)

          metric_calc_perclass('./'+str(i), true_class_id, c)


cls_name = input('Enter dataset for metrics: ')

if cls_name == 'test':
     metric_calc('./data/test/Images')

if cls_name == 'train':
     metric_calc('./data/train/Images')

if cls_name == 'val':
     metric_calc('./data/val/Images')




# print(total_overall_correct_detections)
# print(total_img_files)
avg_recall = total_overall_correct_detections/total_img_files
avg_prec = total_overall_correct_detections/total_overall_detections
print('Average Recall:', avg_recall)
print('Average Precision:', avg_prec)
print('Avg F1 score:', 2*avg_prec*avg_recall/(avg_prec + avg_recall))


