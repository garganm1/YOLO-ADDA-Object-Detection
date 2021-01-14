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
import glob
from skimage.transform import resize




encoder_config_path='config/yolov3_encoder.cfg'
yolo_config_path = 'config/yolov3_yolo.cfg'
weights_path='./checkpoints/confthres_0.9_lr_0.01/target_encoder/encoder_weights/1.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.9
nms_thres=0.4
# Load model and weights
# model = Darknet(config_path, img_size=img_size)
encoder = Darknet(encoder_config_path, True)
yolo = Darknet(yolo_config_path, False)
encoder.load_weights(weights_path)
encoder.cuda()
yolo.cuda()
encoder.eval()
yolo.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor


def detect_image(img):
    # # scale and pad image
    # ratio = min(img_size/img.size[0], img_size/img.size[1])
    # imw = round(img.size[0] * ratio)
    # imh = round(img.size[1] * ratio)
    # img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
    #      transforms.Pad((max(int((imh-imw)/2),0), 
    #           max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
    #           max(int((imw-imh)/2),0)), (128,128,128)),
    #      transforms.ToTensor(),
    #      ])
    
    # # convert image to Tensor
    # image_tensor = img_transforms(img).float()
    # # plt.imsave('test.jpg', image_tensor.numpy().transpose((1,2,0)))
    # # input()

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

    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    
    with torch.no_grad():
        detections = yolo(encoder(input_img))

        detections = utils.non_max_suppression(detections, 7, 
                        conf_thres, nms_thres)[0]
 
    return detections
    
cls_name = input('Enter the class name for images: ' )
data_name = input('Enter the dataset: ')
files = glob.glob('./data/' + data_name +'/Images/' + cls_name + '/*.jpg')

for img_path in files:
    n,ext = os.path.splitext(os.path.basename(img_path))
    pic = n+ext
    prev_time = time.time()
    img = Image.open(img_path)
    # im = np.array(img)
    # plt.imsave('test.jpg', im)
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
    
    plt.show()