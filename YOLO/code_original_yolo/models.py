from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
from utils.parse_config import *
from utils.utils import build_targets
from collections import defaultdict

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)  # to pop out the net layer from module_defs and stores in hyperparams 
    output_filters = [int(hyperparams["channels"])] # contains the number of channels of the input image mentioned in net block
                                                    #  filters of every layer are appended to the list to keep the track
    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs):
        '''Going over each block in config file which is stored as one dict in module_defs and adding it to modules(i.e. nn.Sequential)
        At the end one block added to module_list'''

        modules = nn.Sequential()
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"]) # batch_normalizationn is a flag to tell whether or not to perform batch norm
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0 #padding is done only if pad = 1 # padding is done in a way that activation map has same dimeension as input
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1], # this is the number of channels of the input to current conv layer
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn, # if batch norm done then bias cances out therefore no bias required
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        # When layers attribute has only one value, it outputs the feature maps of the layer indexed by the value. Then detection is done to this new scale
        # When layers has two values, it returns the concatenated feature maps of the layers indexed by it's values.
        
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        
        # The output of the shortcut layer is obtained by adding feature maps from the previous and the 3rd layer backwards from the shortcut layer.

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs] # list of tuples for the particular scale. A tuple has info about the width and ht. of the anchor bx
            num_classes = int(module_def["classes"])
            img_height = int(hyperparams["height"]) # takes the height from the net block, which is 416 in our case
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)  # Storing each block of layer in module_list
        output_filters.append(filters)   # Storing number of filters in each layer

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim #416
        self.ignore_thres = 0.5
        self.lambda_coord = 1
        self.lambda_noobj = 5
        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        # targets is of shape (B,1,5). Here one represents total number of actual banned items present in a single image
        # x is of shape (B, (5+num_classes)*3, 13,13)) / (B, (5+num_classes)*3, 26,26)) / (B, (5+num_classes)*3, 52,52)) 
        nA = self.num_anchors  
        nB = x.size(0)  # Number of images in a batch
        nG = x.size(2)  # Grid Size (13, 26, 52)
        stride = self.image_dim / nG  # Computing stride (416/13 = 32, 416/26 = 16, 416/52 = 8)

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous() #(B,3,13,13,(5+num_classes)) / (B,3,26,26,(5+num_classes)) / (B,3,52,52,(5+num_classes))
        
        # Get outputs
        # all have size (B, 3,13,13)
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf 

        # (B,3,13,13,num_classes)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred. # No softmax...####

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor) #(1,1,13,13) / (1,1,26,26) / (1,1,52,52) # x cordinates for all the grids in 13*13, 26*26, 52*52
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor) #(1,1,13,13) / (1,1,26,26) / (1,1,52,52) # y cordinates for all the grids in 13*13, 26*26, 52*52
        
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]) # anchors attributes mentioned in the yolo block is wrt to 416 * 416 scale. Rescaling it to 13*13 / 26*26 / 52*52 scale 
        # scaled_anchors: tensor of shape (3,2) where 3 implies the number of anchors and 2 are the width and height
        # Each row is one anchor box scaled to 13*13, 26*26, 52*52

        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1)) # contains width of all scaled anchors #(1,1,3,1)
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1)) # contains height of all scaled anchors #(1,1,3,1)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape) # (B,3,13,13,4) / (B,3,26,26,4) / (B,3,52,52,4)      
        pred_boxes[..., 0] = x.data + grid_x # Centre_x of bbox wrt to scale and wrt top left corner
        pred_boxes[..., 1] = y.data + grid_y # Centre_y of bbox wrt to scale and wrt top left corner
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w # Actual width of bbox wrt to scale
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h # Actual height of bbox wrt to scale

        # Training
        # (B,1,5)
        # attr are the class id, Cx, Cy, width, height 
        if targets is not None:
            
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim
            )
            
            # Out of the all predicted bbox, region proposals are the ones with objectness prob > 0.5. We take the prediction by the model only for those whose objectness prob > 0.5
            nProposals = int((pred_conf > 0.5).sum().item()) 
            
            # ratio of out of all objects present, how many were predicted correctly 
            recall = float(nCorrect / nGT) if nGT else 1
            precision = 0
            if nProposals > 0:
                # Ratio of out of the items predicted or proposed by the model, how many were actually correct.
                precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt

            # Values are 1 for the best anchors in the whole batch and objects in each image of batch

            # Values are 1 only in the grid cell responsible for the banned object in the image and that too corresposnding to the
            # anchor box with max overlapping area
            conf_mask_true = mask.type(torch.cuda.BoolTensor) if x.is_cuda else mask.type(torch.BoolTensor)
            
           
            # Values are 1 corresponding to all 3 anchor boxes for all the grid cells except the grid cell responsilble for 
            # banned item present in the image. Inside the grid cell responsible, values are 1 corresponding to anchor boxes 
            # which is not the best anchor box and whose iou is less than threshold
            conf_mask_false = (conf_mask - mask).type(torch.cuda.BoolTensor) if x.is_cuda else (conf_mask - mask).type(torch.BoolTensor)
            
            mask = mask.type(torch.cuda.BoolTensor) if x.is_cuda else mask.type(torch.BoolTensor)

            # x = Predicted Center_x (B,3,13,13)

            # Mask outputs to ignore non-existing objects
            # x[mask] denotes best values of x-coordinates predicted from the top corner of corresponding grid cell

            # x[mask] keeps the predicted tx only for the best anchors (as values of the mask tensor were 1 corresponding to the best anchors)
            # In utils.py, we stored the true tx (after normalization) at the place where mask was 1 corresponding to the best anchor position and all other values were zero
            #  Similarly for others
            loss_x = self.lambda_coord*self.mse_loss(x[mask], tx[mask])
            loss_y = self.lambda_coord*self.mse_loss(y[mask], ty[mask])
            loss_w = self.lambda_coord*self.mse_loss(w[mask], tw[mask])
            loss_h = self.lambda_coord*self.mse_loss(h[mask], th[mask])

            # tconf has values = 1 only corresponding to the position for the best anchor box, while other are zeros
            # While pred_conf has some some values at all positions

            # This picks out values for the positions not corresponding to the best anchor box
            # i.e.tconf[conf_mask_false] has 0 everywhere while pred_conf[conf_mask_false] has some values. This will force the predicted values at 
            # those positions to go to 0 as  don’t want the network to cheat by proposing objects everywhere

            # The second part picks out values for the positions  corresponding to the best anchor box
            # i.e.tconf[conf_mask_true] has 1 at these positions while pred_conf[conf_mask_true] has some values. This will force the predicted values at 
            # those positions to go to 1 
            
            loss_conf = self.lambda_noobj * self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])


            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride, # Transforming to original scale i.e. 416
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            
            # (B,3*13*13/3*26*26/3*52*52,5+num_classes)
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
               

            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:

                    x, *losses = module[0](x, targets) # WHy not module(x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        
                        self.losses[name] += loss # Being a default dict initial values for all the keys are zero
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x) # List of output of the each layer if not training, otherwise appending losses of one batch after every YOLO layer
            layer_outputs.append(x)  # Appending output of each layer sequentially if not training; otherwise losses of one batch after every YOLO layer


        self.losses["recall"] /= 3 # Avg recall score i.e. sum of recall for each scale / 3
        self.losses["precision"] /= 3
        
        # Returning total loss if training, otherwise return prediction output of shape (B, 10647, 5+num_classes)
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            # if self.module_defs[i+1]['type'] == 'yolo':

            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def save_weights(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
