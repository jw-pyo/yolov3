from __future__ import division
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PIL import Image

from utils.parse_config import *
from utils.utils import *
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

ONNX_EXPORT = False

def parse_module(i, module_def, hyperparams, output_filters, yolo_layer_count):
    modules = nn.Sequential()
    filters = None
    
    if module_def["type"] == "convolutional":
        bn = int(module_def["batch_normalize"])
        filters = int(module_def["filters"])
        kernel_size = int(module_def["size"])
        pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
        modules.add_module(
            "conv_%d" % i,
            nn.Conv2d(
                in_channels=output_filters[-1],
                out_channels=filters,
                kernel_size=kernel_size,
                stride=int(module_def["stride"]),
                padding=pad,
                bias=not bn,
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
        upsample = Upsample(scale_factor=int(module_def["stride"]))
        modules.add_module("upsample_%d" % i, upsample)

    elif module_def["type"] == "route":
        layers = [int(x) for x in module_def["layers"].split(",")]
        #filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
        filters = sum([output_filters[layer_i-1] for layer_i in layers]) # jwpyo: self-repaired error. it should be layer_i - 1 
        #filters = sum([output_filters[layer_i] for layer_i in layers]) # commented by https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/82
        modules.add_module("route_%d" % i, EmptyLayer())

    elif module_def["type"] == "shortcut":
        filters = output_filters[int(module_def["from"])]
        modules.add_module("shortcut_%d" % i, EmptyLayer())

    elif module_def["type"] == "yolo":
        anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
        # Extract anchors
        anchors = [int(x) for x in module_def["anchors"].split(",")]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        num_classes = int(module_def["classes"])
        img_height = int(hyperparams["height"])
        # Define detection layer
        yolo_layer = YOLOLayer(anchors, num_classes, img_height, yolo_layer_count[0])
        modules.add_module("yolo_%d" % i, yolo_layer)
        yolo_layer_count[0] += 1
    
    # print("parse_module call. {} {}".format(modules, filters))
    return modules, filters

def create_multitask_modules(shared_module_defs, diff_module_defs_list):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = shared_module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    num_tasks = len(diff_module_defs_list)
    yolo_layer_count = [0]
    module_list_shared = nn.ModuleList()
    module_list_diff = [nn.ModuleList() for _ in range(num_tasks)]

    # shared weight
    for i, module_def in enumerate(shared_module_defs):
        # Register module list and number of output filters
        modules, filters = parse_module(i, module_def, hyperparams, output_filters, yolo_layer_count)
        module_list_shared.append(modules)
        if filters is not None:
            output_filters.append(filters)
        #print(i, len(output_filters))
    shared_output_filters = copy.copy(output_filters) # shallow copy

    
    # individual task
    for j, diff_module_defs in enumerate(diff_module_defs_list):
        del output_filters
        output_filters = copy.copy(shared_output_filters)
        for i, module_def in enumerate(diff_module_defs):
            modules, filters = parse_module(i, module_def, hyperparams, output_filters, yolo_layer_count)
            module_list_diff[j].append(modules)
            if filters is not None:
                output_filters.append(filters)
            #print(j, len(output_filters))
    
    return hyperparams, module_list_shared, module_list_diff
def create_md_modules_single_cfg(module_defs, cutoff):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    branch_num = 3
    yolo_layer_count = [0]
    module_list = nn.ModuleList()

    # shared weight
    for i, module_def in enumerate(module_defs[:cutoff]):
        # Register module list and number of output filters
        modules, filters = parse_module(i, module_def, hyperparams, output_filters, yolo_layer_count)
        module_list.append(modules)
        if filters is not None:
            output_filters.append(filters)
        #print(i, len(output_filters))
    shared_output_filters = copy.copy(output_filters) # shallow copy
    
    # individual task
    for j in range(branch_num):
        del output_filters
        output_filters = copy.copy(shared_output_filters)
        for i, module_def in enumerate(module_defs[cutoff:]):
            modules, filters = parse_module(i, module_def, hyperparams, output_filters, yolo_layer_count)
            module_list.append(modules)
            if filters is not None:
                output_filters.append(filters)
            #print(j, len(output_filters))
    
    return hyperparams, module_list





def create_modules(module_defs, last_filter=None):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    if not diff:
        hyperparams = module_defs.pop(0)
    else:
    """
    hyperparams = module_defs.pop(0)
    yolo_layer_count = [0]
    if last_filter is None:
        output_filters = [int(hyperparams["channels"])]
    else:
        output_filters = [last_filter]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules, filters = parse_module(i, module_def, hyperparams, output_filters, yolo_layer_count)
        # Register module list and number of output filters
        module_list.append(modules)
        if filters is not None:
            output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, img_size, yolo_layer):
        super(YOLOLayer, self).__init__()

        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.img_size = 0
        # self.coco_class_weights = coco_class_weights()

    def forward(self, p, img_size, targets=None, var=None):
        if ONNX_EXPORT:
            bs, nG = 1, self.nG  # batch size, grid size
        else:
            bs, nG = p.shape[0], p.shape[-1]

            if self.img_size != img_size:
                create_grids(self, img_size, nG)

                if p.is_cuda:
                    #self.grid_xy = self.grid_xy.cuda()
                    #self.anchor_wh = self.anchor_wh.cuda() 
                    self.grid_xy = self.grid_xy.to(p.device) #jwpyo: to use multi-GPU
                    self.anchor_wh = self.anchor_wh.to(p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # xy, width and height
        xy = torch.sigmoid(p[..., 0:2])
        wh = p[..., 2:4]  # wh (yolo method)
        # wh = torch.sigmoid(p[..., 2:4])  # wh (power method)

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            # Get outputs
            p_conf = p[..., 4]  # Conf
            p_cls = p[..., 5:]  # Class

            txy, twh, mask, tcls = build_targets(targets, self.anchor_vec, self.nA, self.nC, nG)

            tcls = tcls[mask]
            if p.is_cuda:
                #txy, twh, mask, tcls = txy.cuda(), twh.cuda(), mask.cuda(), tcls.cuda()
                txy, twh, mask, tcls = txy.to(p.device), twh.to(p.device), mask.to(p.device), tcls.to(p.device) #jwpyo: to use multi-GPU

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            k = 1  # nM / bs
            if nM > 0:
                lxy = k * MSELoss(xy[mask], txy[mask])
                lwh = k * MSELoss(wh[mask], twh[mask])

                lcls = (k / 4) * CrossEntropyLoss(p_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * BCEWithLogitsLoss(p_cls[mask], tcls.float())
            else:
                FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
                lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])

            lconf = (k * 64) * BCEWithLogitsLoss(p_conf, mask.float())

            # Sum loss components
            loss = lxy + lwh + lconf + lcls

            return loss, loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item(), nT

        else:
            if ONNX_EXPORT:
                grid_xy = self.grid_xy.repeat((1, self.nA, 1, 1, 1)).view((1, -1, 2))
                anchor_wh = self.anchor_wh.repeat((1, 1, nG, nG, 1)).view((1, -1, 2)) / nG

                # p = p.view(-1, 85)
                # xy = xy + self.grid_xy[0]  # x, y
                # wh = torch.exp(wh) * self.anchor_wh[0]  # width, height
                # p_conf = torch.sigmoid(p[:, 4:5])  # Conf
                # p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
                # return torch.cat((xy / nG, wh, p_conf, p_cls), 1).t()

                p = p.view(1, -1, 85)
                xy = xy + grid_xy  # x, y
                wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
                p_conf = torch.sigmoid(p[..., 4:5])  # Conf
                p_cls = p[..., 5:85]
                # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
                # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
                p_cls = torch.exp(p_cls).permute((2, 1, 0))
                p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
                p_cls = p_cls.permute(2, 1, 0)
                return torch.cat((xy / nG, wh, p_conf, p_cls), 2).squeeze().t()

            p[..., 0:2] = xy + self.grid_xy  # xy
            p[..., 2:4] = torch.exp(wh) * self.anchor_wh  # wh yolo method
            # p[..., 2:4] = ((wh * 2) ** 2) * self.anchor_wh  # wh power method
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
            p[..., :4] *= self.stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return p.view(bs, -1, 5 + self.nC)

class Classifier(nn.Module):
    def __init__(self, img_size=416):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 208)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 94)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        #FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        #x = x.type(torch.FloatTensor)
        img = x
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        #softmax and pick one that have the highest probability
        x = F.softmax(x)
        index = None
        print("hihhih", x.data)
        print("hihhih", torch.max(x).data)
        for i in range(3):
            if torch.eq(torch.max(x).data, x.data[0][i]):
                index = i
        print("This image is classified as {}\n".format(index))
        #x.data[0][index]
        return index, img

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'xy', 'wh', 'conf', 'cls', 'nT']
        self.losses = []

    def freeze_layers(self, cutoff):
        for i, (name, p) in enumerate(self.named_parameters()):
            if int(name.split(".")[1]) < cutoff:
                p.requires_grad = False
            else:
                p.requires_grad = True
    
    def forward(self, x, targets=None, var=0):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        img_size = x.shape[-1]
        layer_outputs = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, img_size, targets, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module[0](x, img_size)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nT'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path, cutoff=-1):
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
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
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
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

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
    def get_n_params(self):
        MB = 1024 * 1024
        GB = 1024* MB
        total_params=0
        for i, p in enumerate(list(self.parameters())):
            layer_params = 1
            print("layer {}: {}".format(i,p.size()))
            for s in p.size():
                layer_params = layer_params * s
            total_params += layer_params
        print("Total number of parameters: {}".format(total_params))
        print("Capacity of paramaeters   : {}MB = {}GB".format(total_params*4/(1*MB + 1e-7), total_params*4/(1*GB + 1e-7)))

def get_yolo_layers(model):
    a = [module_def['type'] == 'yolo' for module_def in model.module_defs]
    return [i for i, x in enumerate(a) if x] #[82, 94, 106] for yolov3



def create_grids(self, img_size, nG):
    self.stride = img_size / nG

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)

def load_darknet_weights_to_md(self, weights, cutoff=-1):
    # Parses and loads the weights to multi-domain network stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    #TODO
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15
    elif weights_file == 'yolov3.weights': #shared 75, diff 32
        cutoff = 75
    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


class SubModule(nn.Module):
    def __init__(self, embedding):
        super(SubModule, self).__init__()
        self.embedding = embedding
        self.diff = parse_model_cfg("cfg/multidarknet/diff1.cfg")
    def forward(self, input):
        return self.diff(self.embedding(input))


class MultiDarknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, shared_config_path, diff_config_paths, img_size=416, branch_num=3, hill_climbing=False, shared_len=75):
        super(MultiDarknet, self).__init__()
        self.shared_config_path = shared_config_path
        self.diff_config_paths = diff_config_paths
        self.branch_num = branch_num
        if not hill_climbing:
            shared_module_defs = parse_model_cfg(shared_config_path) #the Module's type of python list 
            diff_module_defs_list = [parse_model_cfg(i) for i in diff_config_paths]
            self.hyperparams, \
            module_list_shared, \
            module_list_diff = create_multitask_modules(shared_module_defs, \
                                                            diff_module_defs_list)
            """
            self.diff1_module_list = SubModule(module_list_shared)
            self.diff2_module_list = SubModule(module_list_shared)
            self.diff3_module_list = SubModule(module_list_shared)
            """
            self.shared_len = len(module_list_shared)
            self.diff_len = len(module_list_diff[0])
            
            # module_def, module_list
            list_of_layers = list(module_list_shared.children())
            
            for i in range(len(module_list_diff)):
                list_of_layers.extend(list(module_list_diff[i].children()))
            self.module_list = nn.Sequential(*list_of_layers)
            
            self.module_def = shared_module_defs
            
            for i in range(len(diff_module_defs_list)):
                self.module_def.extend(diff_module_defs_list[i]) # per 32
        #hillclimbing
        else:
            self.shared_len = shared_len
            self.diff_len = 107 - self.shared_len
            self.branch_num = 3
            temp_module_defs = parse_model_cfg("cfg/multidarknet/bdd100k.cfg") # in hillclimbing mode, shared_config_path is equal to config_path
            
            self.hyperparams, self.module_list = create_md_modules_single_cfg(temp_module_defs, self.shared_len)
            self.module_def = temp_module_defs[:self.shared_len]
            for i in range(self.branch_num):
                self.module_def.extend(temp_module_defs[self.shared_len:])
        
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['loss', 'xy', 'wh', 'conf', 'cls', 'nT']
    def hillClimbing(self, cutoff_layer):
        """
        move current model to cutoff_layer-th architecture
        self.hyperparams
        self.module_list
        self.module_def
        """
        print("Call hillClimbing for cutoff {}".format(cutoff_layer))
        new_model = MultiDarknet(self.shared_config_path, self.diff_config_paths, self.img_size, hill_climbing=True, shared_len=cutoff_layer) 
        
        #cur_param = list(self.parameters())
        #new_param = list(new_model.parameters())
        #dict_prev = dict(cur_param)
        #dict_new = dict(new_param)
        #TODO: self.named_parameters() has no running_mean, ... but we need that if we use load_state_dict()
        dict_prev = self.state_dict()
        dict_new = new_model.state_dict()

        prev_layer_name = list(dict_prev)
        new_layer_name = list(dict_new)
        """
        for name1, param1 in cur_param -> for i, name1 in enumerate(prev_layer_name) param1 = dict_prev[name1]
        for name1, param2 in new_param -> for i, name2 in enumerate(new_layer_name) param2 = dict_new[name2]
        """
        tmp_shared_dict = OrderedDict()
        tmp_diff_dict = [OrderedDict() for _ in range(self.branch_num)]
        diff = self.shared_len - cutoff_layer
        
        # copy the shared weight 
        for i in range(self.branch_num):
            share_once = True # share shared_dict in each diff_dict once only.
            for _, name1 in enumerate(prev_layer_name):
                param1 = dict_prev[name1]
                layer_num = int(name1.split(".")[1])
                if layer_num < cutoff_layer:
                    if i == 0:
                        dict_new[name1].data.copy_(param1)
                elif layer_num >= cutoff_layer and layer_num < cutoff_layer + diff:
                    if i == 0:
                        tmp_shared_dict[name1] = param1
                elif layer_num == cutoff_layer + diff + i*self.diff_len:
                    if share_once:
                        tmp_diff_dict[i] = copy.copy(tmp_shared_dict)
                        share_once = False
                    tmp_diff_dict[i][name1] = param1 
                elif layer_num > cutoff_layer + diff + i*self.diff_len and layer_num < cutoff_layer + diff + (i+1)*self.diff_len:
                    tmp_diff_dict[i][name1] = param1
        
        #complete copying the each branch into tmp_diff_dict
        # copy the diff weight
        for i in range(self.branch_num):
            diff_index = 0
            for _, name2 in enumerate(new_layer_name):
                param2 = dict_new[name2]
                layer_num = int(name2.split(".")[1])
                if layer_num < cutoff_layer:
                    pass
                elif layer_num >= cutoff_layer + i*(self.diff_len + diff) and layer_num < cutoff_layer + (i+1)*(self.diff_len + diff): 
                    prev_param = dict_prev[list(tmp_diff_dict[i].keys())[diff_index]] 
                    #print("current model's key: {} {}".format(list(tmp_diff_dict[i].keys())[diff_index], prev_param.size()))
                    #print("new     model's key: {} {}".format(name2, param2.size()))
                    dict_new[name2].data.copy_(prev_param)
                    diff_index += 1
                else:
                    pass
        #update the changed parameters to new model 
        new_model.load_state_dict(dict_new)
        
        return new_model
        
    def freeze_layers(self, cutoff):
        for i, (name, p) in enumerate(self.named_parameters()):
            if int(name.split(".")[1]) < cutoff:
                p.requires_grad = False
            else:
                p.requires_grad = True
    
    def get_n_params(self):
        MB = 1024 * 1024
        GB = 1024* MB
        total_params=0
        for i, p in enumerate(list(self.parameters())):
            layer_params = 1
            print("layer {}: {}".format(i,p.size()))
            for s in p.size():
                layer_params = layer_params * s
            total_params += layer_params
        print("Total number of parameters: {}".format(total_params))
        print("Capacity of paramaeters   : {}MB = {}GB".format(total_params*4/(1*MB + 1e-7), total_params*4/(1*GB + 1e-7)))
    def to_device(self, device):
        for i, p in enumerate(list(self.parameters())):
            if p.device == device:
                p.to(device)
    def forward(self, x, targets=None, cond=0):
        """
        cond: the index of branch which to choose
        """
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        x = x.type(FloatTensor) 
        img_size = x.shape[-1]
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        """ 
        module_defs = self.shared_module_defs
        module_defs.extend(self.diff_module_defs_list[cond])
        module_list = self.module_list_shared
        module_list.extend(self.module_list_diff[cond])
        """
        # match the device type of x and weight
        if self.module_list[0][0].weight.device == x.device:
            pass
        else:
            x.to(self.module_list[0][0].weight.device)
        # for i, (module_def, module_) in enumerate(zip(module_defs, module_list)):
        # print("length of shared_module_defs: ", len(self.shared_module_defs))
        
        #shared layer(default value of self.shared_len = 75)
        for i, (module_def, module_) in enumerate(zip( \
                self.module_def[0:self.shared_len], self.module_list[0:self.shared_len])):
            
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module_(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)

            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module_[0](x, self.img_size, targets, cond)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module_[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)
        
        # diff layer
        # print("length of diff_module_defs: ", len(self.diff_module_defs_list[cond]))
        """
        if cond == 0:
            self.diff = self.diff1_module_list
        elif cond == 1:
            self.diff = self.diff2_module_list
        else:
            self.diff = self.diff3_module_list
        """
        
        #Default value of self.shared len, self.diff_len = (75, 32)
        for i, (module_def, module_) in enumerate(zip( \
            self.module_def[self.shared_len + self.diff_len*cond : self.shared_len + self.diff_len*(cond+1)], \
            self.module_list[self.shared_len + self.diff_len*cond : self.shared_len + self.diff_len*(cond+1)])): 
            
            #module_ = module_.cuda()
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module_(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module_[0](x, img_size, targets, cond)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module_[0](x, img_size, None, cond)
                output.append(x)  
            layer_outputs.append(x)
        
        #print("shared {}\n diff1 {}\n diff2 {}\n diff3 {}\n".format(
        #    self.module_list[5][0].weight[5][10][1], self.module_list[89][0].weight[5][10:13].permute(2,1,0), self.module_list[121][0].weight[5][10:13].permute(2,1,0), self.module_list[153][0].weight[5][10:13].permute(2,1,0)))
        if is_training:
            self.losses['nT'] /= 3
        
        #self.losses["recall"] /= 3
        #self.losses["precision"] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):
        print("load-weights is called")
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

        # fetch the shared weight first, then individual branch
        for i, (module_def, module) in enumerate(zip( \
               # self.shared_module_defs + sum(self.diff_module_defs_list, []), \
               # self.module_list_shared + sum(self.module_list_diff, []))):
                 self.module_def,
                 self.module_list)):
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
    
    def save_weights(self, path, cutoff=-1):

        """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers - shared weight first
        
        for i, (module_def, module) in enumerate(zip(self.shared_module_defs, self.module_list_shared)):
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
        
        # Iterate through layers - individual branch next
        num_tasks = len(self.diff_module_defs_list)
        for j in range(num_tasks):
            for i, (module_def, module) in enumerate(zip(self.diff_module_defs_list[j][:cutoff], self.module_list_diff[j][:cutoff])): 
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
if __name__ == "__main__":
    model = MultiDarknet()
    model.get_n_params()
