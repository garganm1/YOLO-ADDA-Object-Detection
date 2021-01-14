import numpy as np
from PIL import Image
import torch
from skimage.transform import resize
import os


def train_loader(handgun, battery, usb, shuriken, knife, phone, hard_disk, handgun_labels, battery_labels, usb_labels, shuriken_labels, knife_labels, phone_labels, hard_disk_labels):
    
    batch_size = 18
    img_size = 416
    max_objects = 1
    i = 72
    while i < (len(handgun)/2 - 1):
        
        img_paths = []
        label_paths = []

        if ((i+1)*2)%len(handgun) < (i*2)%len(handgun):
            img_paths.extend(handgun[(i*2)%len(handgun):len(handgun)])
            img_paths.extend(handgun[:((i+1)*2)%len(handgun)])
        else:
            img_paths.extend(handgun[(i*2)%len(handgun):((i+1)*2)%len(handgun)])
        
        if ((i+1)*2)%len(battery) < (i*2)%len(battery):
            img_paths.extend(battery[(i*2)%len(battery):len(battery)])
            img_paths.extend(battery[:((i+1)*2)%len(battery)])
        else:
            img_paths.extend(battery[(i*2)%len(battery):((i+1)*2)%len(battery)])

        if ((i+1)*2)%len(hard_disk) < (i*2)%len(hard_disk):
            img_paths.extend(hard_disk[(i*2)%len(hard_disk):len(hard_disk)])
            img_paths.extend(hard_disk[:((i+1)*2)%len(hard_disk)])
        else:
            img_paths.extend(hard_disk[(i*2)%len(hard_disk):((i+1)*2)%len(hard_disk)])
        
        if ((i+1)*3)%len(usb) < (i*3)%len(usb):
            img_paths.extend(usb[(i*3)%len(usb):len(usb)])
            img_paths.extend(usb[:((i+1)*3)%len(usb)])
        else:
            img_paths.extend(usb[(i*3)%len(usb):((i+1)*3)%len(usb)])

        if ((i+1)*3)%len(shuriken) < (i*3)%len(shuriken):
            img_paths.extend(shuriken[(i*3)%len(shuriken):len(shuriken)])
            img_paths.extend(shuriken[:((i+1)*3)%len(shuriken)])
        else:
            img_paths.extend(shuriken[(i*3)%len(shuriken):((i+1)*3)%len(shuriken)])

        if ((i+1)*3)%len(phone) < (i*3)%len(phone):
            img_paths.extend(phone[(i*3)%len(phone):len(phone)])
            img_paths.extend(phone[:((i+1)*3)%len(phone)])
        else:
            img_paths.extend(phone[(i*3)%len(phone):((i+1)*3)%len(phone)])        

        if ((i+1)*3)%len(knife) < (i*3)%len(knife):
            img_paths.extend(knife[(i*3)%len(knife):len(knife)])
            img_paths.extend(knife[:((i+1)*3)%len(knife)])
        else:
            img_paths.extend(knife[(i*3)%len(knife):((i+1)*3)%len(knife)])
        
        if ((i+1)*2)%len(handgun_labels) < (i*2)%len(handgun_labels):
            label_paths.extend(handgun_labels[(i*2)%len(handgun_labels):len(handgun_labels)])
            label_paths.extend(handgun_labels[:((i+1)*2)%len(handgun_labels)])
        else:
            label_paths.extend(handgun_labels[(i*2)%len(handgun_labels):((i+1)*2)%len(handgun_labels)])
        
        if ((i+1)*2)%len(battery_labels) < (i*2)%len(battery_labels):
            label_paths.extend(battery_labels[(i*2)%len(battery_labels):len(battery_labels)])
            label_paths.extend(battery_labels[:((i+1)*2)%len(battery_labels)])
        else:
            label_paths.extend(battery_labels[(i*2)%len(battery_labels):((i+1)*2)%len(battery_labels)])

        if ((i+1)*2)%len(hard_disk_labels) < (i*2)%len(hard_disk_labels):
            label_paths.extend(hard_disk_labels[(i*2)%len(hard_disk_labels):len(hard_disk_labels)])
            label_paths.extend(hard_disk_labels[:((i+1)*2)%len(hard_disk_labels)])
        else:
            label_paths.extend(hard_disk_labels[(i*2)%len(hard_disk_labels):((i+1)*2)%len(hard_disk_labels)])
        
        if ((i+1)*3)%len(usb_labels) < (i*3)%len(usb_labels):
            label_paths.extend(usb_labels[(i*3)%len(usb_labels):len(usb_labels)])
            label_paths.extend(usb_labels[:((i+1)*3)%len(usb_labels)])
        else:
            label_paths.extend(usb_labels[(i*3)%len(usb_labels):((i+1)*3)%len(usb_labels)])

        if ((i+1)*3)%len(shuriken_labels) < (i*3)%len(shuriken_labels):
            label_paths.extend(shuriken_labels[(i*3)%len(shuriken_labels):len(shuriken_labels)])
            label_paths.extend(shuriken_labels[:((i+1)*3)%len(shuriken_labels)])
        else:
            label_paths.extend(shuriken_labels[(i*3)%len(shuriken_labels):((i+1)*3)%len(shuriken_labels)])

        if ((i+1)*3)%len(phone_labels) < (i*3)%len(phone_labels):
            label_paths.extend(phone_labels[(i*3)%len(phone_labels):len(phone_labels)])
            label_paths.extend(phone_labels[:((i+1)*3)%len(phone_labels)])
        else:
            label_paths.extend(phone_labels[(i*3)%len(phone_labels):((i+1)*3)%len(phone_labels)])        

        if ((i+1)*3)%len(knife_labels) < (i*3)%len(knife_labels):
            label_paths.extend(knife_labels[(i*3)%len(knife_labels):len(knife_labels)])
            label_paths.extend(knife_labels[:((i+1)*3)%len(knife_labels)])
        else:
            label_paths.extend(knife_labels[(i*3)%len(knife_labels):((i+1)*3)%len(knife_labels)])

        batch_img = None
        img_shape = (img_size, img_size)

        # Image

        for img_path in img_paths:

            img_path = img_path.rstrip()
            img = np.array(Image.open(img_path))

            # Handles images with less than three channels
            while len(img.shape) != 3:
                index += 1
                img_path = img_path.rstrip()
                img = np.array(Image.open(img_path))

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
            input_img = resize(input_img, (*img_shape, 3), mode='reflect')
            # Channels-first
            input_img = np.transpose(input_img, (2, 0, 1))
            # As pytorch tensor
            input_img = torch.from_numpy(input_img).float().unsqueeze(0)
            if batch_img == None:
                batch_img = input_img
            else:
                batch_img = torch.cat((batch_img, input_img), dim = 0)
            
    
    #         #---------
    #         #  Label
    #         #---------
        batch_label = None
        for label_path in label_paths:
            

            label_path = label_path.rstrip()
            labels = None
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path).reshape(-1, 5)
                # Extract coordinates for unpadded + unscaled image
                x1 = w * (labels[:, 1] - labels[:, 3]/2)
                y1 = h * (labels[:, 2] - labels[:, 4]/2)
                x2 = w * (labels[:, 1] + labels[:, 3]/2)
                y2 = h * (labels[:, 2] + labels[:, 4]/2)
                # Adjust for added padding
                x1 += pad[1][0]
                y1 += pad[0][0]
                x2 += pad[1][0]
                y2 += pad[0][0]
                # Calculate ratios from coordinates
                labels[:, 1] = ((x1 + x2) / 2) / padded_w
                labels[:, 2] = ((y1 + y2) / 2) / padded_h
                labels[:, 3] *= w / padded_w
                labels[:, 4] *= h / padded_h
            # Fill matrix
            filled_labels = np.zeros((max_objects, 5))
            if labels is not None:
                filled_labels[range(len(labels))[:max_objects]] = labels[:max_objects]
            filled_labels = torch.from_numpy(filled_labels).unsqueeze(0)

            if batch_label == None:
                batch_label = filled_labels
            else:
                batch_label = torch.cat((batch_label, filled_labels), dim = 0)

        i += 1

        yield batch_img, batch_label


        
            
        