import numpy as np
from PIL import Image
import torch
from skimage.transform import resize
import os
import matplotlib.pyplot as plt


def train_loader(handgun, shuriken, usb, phone, knife, hard_disk, battery, handgun_labels, shuriken_labels, usb_labels, phone_labels, knife_labels, hard_disk_labels, battery_labels, batch_size, num_classes):

    batch_size = batch_size
    img_size = 416
    max_objects = 1
    i = 0

    while i < (len(handgun)/(batch_size/num_classes)-1):
        
        img_paths = []
        label_paths = []

        img_paths.extend(handgun[(i)%len(handgun):((i+1))%len(handgun)])
        img_paths.extend(shuriken[(i)%len(shuriken):((i+1))%len(shuriken)])
        img_paths.extend(usb[(i)%len(usb):((i+1))%len(usb)])
        img_paths.extend(phone[(i)%len(phone):((i+1))%len(phone)])
        img_paths.extend(knife[(i)%len(knife):((i+1))%len(knife)])
        img_paths.extend(hard_disk[(i)%len(hard_disk):((i+1))%len(hard_disk)])
        img_paths.extend(battery[(i)%len(battery):((i+1))%len(battery)])


        label_paths.extend(handgun_labels[(i)%len(handgun_labels):((i+1))%len(handgun_labels)])
        label_paths.extend(shuriken_labels[(i)%len(shuriken_labels):((i+1))%len(shuriken_labels)])
        label_paths.extend(usb_labels[(i)%len(usb_labels):((i+1))%len(usb_labels)])
        label_paths.extend(phone_labels[(i)%len(phone_labels):((i+1))%len(phone_labels)])
        label_paths.extend(knife_labels[(i)%len(knife_labels):((i+1))%len(knife_labels)])
        label_paths.extend(hard_disk_labels[(i)%len(hard_disk_labels):((i+1))%len(hard_disk_labels)])
        label_paths.extend(battery_labels[(i)%len(battery_labels):((i+1))%len(battery_labels)])

        batch_img = None
        img_shape = (img_size, img_size)

        # Image

        for img_path in img_paths:

            img_path = img_path.rstrip()
            img = np.array(Image.open(img_path))
            

            # Handles images with less than three channels
            while len(img.shape) != 3:
                img = np.expand_dims(img, axis = 2)
                img = np.concatenate((img, img, img), 2)


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
