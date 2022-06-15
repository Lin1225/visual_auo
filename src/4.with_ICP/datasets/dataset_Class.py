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
        self.list_rank = []
        self.noise_trans = noise_trans
        self.refine = refine

        self.length = len(self.list_rgb)

        self.cam_cx = 642.827137787509
        self.cam_cy = 365.5359193507853
        self.cam_fx = 614.628928745069
        self.cam_fy = 616.8066524222719

        start2 = timeit.default_timer()
        # self.xmap = np.zeros((720,1280),np.int64)
        # self.ymap = np.zeros((720,1280),np.int64)
        # self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        # self.ymap = np.array([[i for i in range(1280)] for j in range(720)])
        grid = np.indices((720, 1280))
        self.xmap = grid[0]
        self.ymap = grid[1]
        # print(self.xmap)
        # print(self.ymap)
        stop2 = timeit.default_timer()
        # print('Timexxxx: ', stop2-start2) 
        self.num = num
        self.add_noise = add_noise
        # self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        #self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.norm = transforms.Normalize(mean = [110.92494, 112.14303, 112.76024], std = [53.10094, 50.37343, 51.4505]) #(mean = [112.985016, 93.449036, 106.11367], std = [57.782948, 51.31353, 54.971416]) #(mean = [104.86711, 109.139305, 115.26725], std = [54.103874, 50.71502, 46.96569 ])
        #self.norm = transforms.Normalize(mean = [0.411, 0.427, 0.452], std = [0.212, 0.198, 0.184 ])
        

        
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 720, 520, 560, 600, 1280, 680]
        self.num_pt_mesh_large =800
        self.num_pt_mesh_small = 100
        self.symmetry_obj_idx = []

    def __getitem__(self, index):   
        
        img = np.array(self.list_rgb)
        depth = np.array(self.list_depth)
        label = np.array(self.list_label)
        # print(label.shape)
        #print(index)
        obj = self.list_obj[index-1]
        # print("depth:",len(depth.nonzero()[0]))

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        # print("mask_depth:",len(mask_depth.nonzero()[0]))

        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        # print("mask_label:",len(mask_label.nonzero()[0]))

        mask = mask_label * mask_depth
        # print("mask:",len(mask.nonzero()[1]))
        # print("mask shape: ",np.shape(mask))

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img

        rmin, rmax, cmin, cmax = get_bbox(self.list_bbox)
        #rmin = 85
        #rmax = 245
        #cmin = 343
        #cmax = 543
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        p_img = np.transpose(img_masked, (1, 2, 0))
        # scipy.misc.imsave('./{0}_input.png'.format(index), p_img)
        # import imageio
        # imageio.imwrite('./{0}_input.png'.format(index), p_img)

        
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc)

        #print("yaya")
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
        
        # cam_scale = 1.0
        # pt2 = depth_masked / cam_scale
        # pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        # pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        # cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        # cloud = cloud/1000.0
        cam_scale = 1000.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        # print("cloud shape: ",np.shape(cloud))
        
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(cloud)
        # o3d.io.write_point_cloud("filename.pcd", pcd)

        #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        #for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        #for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        #for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        #print(np.shape(cloud))
        #print(np.shape(choose))
        #print(np.shape(img_masked))
        #print(self.objlist.index(obj))
        
        #cloud = np.expand_dims(cloud, axis=0)
        #choose = np.expand_dims(choose, axis=0)
        #img_masked = np.expand_dims(img_masked, axis=0)

        return cloud.astype(np.float32)

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 720, 520, 560, 600, 1280, 680]
img_width = 720
img_length = 1280

def get_bbox(bbox):
    #bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
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


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    #f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)
