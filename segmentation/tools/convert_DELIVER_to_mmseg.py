# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from tqdm import tqdm
import pickle

import mmcv
import cv2
import glob
import numpy as np
# from torchvision.utils import save_image
import torchvision.transforms.functional as TF 
from torchvision import io
import shutil
from PIL import Image
import torch

def save_image(img, path):
    ndarray=img.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if ndarray.shape[2]==1:
        ndarray=ndarray.squeeze()
    im = Image.fromarray(ndarray)
    im.save(path)

def _open_img(file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        return img

def loop_data_organizing(iterator, out_dir, total, data_division):
    for img_name in tqdm(iterator, total=total):

        rgb=img_name
        x1 = rgb.replace('img', 'hha').replace('_rgb', '_depth')
        x2 = rgb.replace('img', 'lidar').replace('_rgb', '_lidar')
        x3 = rgb.replace('img', 'event').replace('_rgb', '_event')
        lbl_path = rgb.replace('img', 'semantic').replace('_rgb', '_semantic')

        # sample = {}
        img = io.read_image(rgb)[:3, ...]
        H, W = img.shape[1:]
        # if 'depth' in self.modals:
        depth = _open_img(x1)
        # if 'lidar' in self.modals:
        lid=_open_img(x2)
        # if 'event' in self.modals:
        eimg = _open_img(x3)
        eimg = TF.resize(eimg, (H, W), TF.InterpolationMode.NEAREST)
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        if np.max(np.unique(np.array(label[0])))>25:
            print("found problem",np.unique(np.array(label[0])))
        label[label==255] = 0
        label -= 1
        save_image(img, osp.join(out_dir, 'samples','images', data_division,img_name.split(os.path.sep)[-4]+"_"+img_name.split(os.path.sep)[-2]+"_"+img_name.split(os.path.sep)[-1]))
        save_image(label, osp.join(out_dir, 'samples','annotations', data_division,lbl_path.split(os.path.sep)[-4]+"_"+lbl_path.split(os.path.sep)[-2]+"_"+lbl_path.split(os.path.sep)[-1]))
        save_image(eimg, osp.join(out_dir, 'samples','event', data_division, x3.split(os.path.sep)[-4]+"_"+x3.split(os.path.sep)[-2]+"_"+x3.split(os.path.sep)[-1]))
        save_image(lid, osp.join(out_dir, 'samples','lidar', data_division, x2.split(os.path.sep)[-4]+"_"+x2.split(os.path.sep)[-2]+"_"+x2.split(os.path.sep)[-1]))
        save_image(depth, osp.join(out_dir, 'samples','depth', data_division, x1.split(os.path.sep)[-4]+"_"+x1.split(os.path.sep)[-2]+"_"+x1.split(os.path.sep)[-1]))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert DELIVER dataset to mmsegmentation format')
    parser.add_argument('dataset_path',nargs='?', const='/datasets/DELIVER' ,default='/datasets/DELIVER', type=str, help='path of DELIVER')
    parser.add_argument('--tmp_dir',default=None, type=str,help='path of the temporary directory')
    # parser.add_argument('--rm_dir', help='directory to be used as test')
    parser.add_argument('-o', '--out_dir',default=None, type=str, help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    args.rm_dir=None
    if args.out_dir is None:
        out_dir = osp.join('datasets', 'DELIVER')
    else:
        out_dir = args.out_dir

    try:
        open(osp.join(out_dir,"folderlist.pkl"),"rb")
    except:
        folder_list=[]
    print('Making directories...')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'images'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'images', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'images', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'images', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'annotations'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'annotations', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'annotations', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'annotations', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'depth'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'depth', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'depth', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'depth', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'event'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'event', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'event', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'event', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'lidar'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'lidar', 'training'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'lidar', 'validation'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'samples', 'lidar', 'test'))
    for split in ['train','val', 'test']: #'train', 'test','val',
        print(f'Generating {split} dataset...')
        files = sorted(glob.glob(os.path.join(*[args.dataset_path, 'img', '*', split, '*', '*.png'])))
        # DELIVER_LEN_temp= len(files)
        iterator=iter(sorted(files))
        if split=="train":
            new_folder='training'
        elif split=="val":
            new_folder='validation'
        else:
            new_folder='test'
        loop_data_organizing(iterator,out_dir,len(files), new_folder)
        print(f"len of {new_folder} set", len(os.listdir(out_dir+"/"+"samples"+"/"+"images"+"/"+new_folder)), len(os.listdir(out_dir+"/"+"samples"+"/"+"annotations"+"/"+new_folder)), len(os.listdir(out_dir+"/"+"samples"+"/"+"depth"+"/"+new_folder)), len(os.listdir(out_dir+"/"+"samples"+"/"+"event"+"/"+new_folder)), len(os.listdir(out_dir+"/"+"samples"+"/"+"lidar"+"/"+new_folder)))

    print('Done!')


if __name__ == '__main__':
    main()