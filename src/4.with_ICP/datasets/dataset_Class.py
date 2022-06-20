import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import timeit

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, rgb, depth, label, bbox, noise_trans, refine):
        self.objlist = [1]
        self.mode = mode

        self.list_rgb = rgb
        self.list_depth = depth
        self.list_label = label
        self.list_bbox = bbox
        self.list_obj = [1]
        self.noise_trans = noise_trans
        self.refine = refine

        # camera intrinsic matrix 
        self.cam_cx = 642.827137787509
        self.cam_cy = 365.5359193507853
        self.cam_fx = 614.628928745069
        self.cam_fy = 616.8066524222719

        grid = np.indices((720, 1280))
        self.xmap = grid[0]
        self.ymap = grid[1]
        self.num = num
        self.add_noise = add_noise
        

    def __getitem__(self):   
        
        depth = np.array(self.list_depth) # depth image
        label = np.array(self.list_label) # bounding box 

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0)) # remove nan in depth image
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255))) # get mask where 255
        mask = mask_label * mask_depth


        rmin, rmax, cmin, cmax = get_bbox(self.list_bbox) # transform bounding box format
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) == 0:            
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])
        
        cam_scale = 1000.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        return cloud.astype(np.float32)

    



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 720, 520, 560, 600, 1280, 680]
img_width = 720
img_length = 1280

def get_bbox(bbox):
    '''
        modify bounding box range
    '''
    bbx = [bbox[0], bbox[2], bbox[1], bbox[3]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 720:
        bbx[1] = 719
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 1280:
        bbx[3] = 1279
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 720:
        delt = rmax - 720
        rmax = 720
        rmin -= delt
    if cmax > 1280:
        delt = cmax - 1280
        cmax = 1280
        cmin -= delt
    return rmin, rmax, cmin, cmax

