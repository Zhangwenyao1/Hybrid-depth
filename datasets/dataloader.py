# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications
# This version has been further modified from the AdaBins version; modifications are only to accommodate the
# differen args format.

import os
import random
import sys
sys.path.append('./')
import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.utils import get_transforms_new, get_transforms
import json
import os
import os.path as osp

class DepthDataLoader(object):
    def __init__(self, args, mode):
        self.args = args


        train_json_file = "/data/zhangwenyao/drive_data/TUSimple/train_set/train_label_data.json"
        train_images_root = "/data/zhangwenyao/drive_data/TUSimple/train_set/"
        
        self.input_transforms = ['random_resized_crop', 'random_hflip', 'normalize']
        self.input_resize = [256, 256] 
        self.input_size = [224, 224]
        self.pixel_mean =  [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        
    
        # val_data_file = test_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32_order/test_data/data_list/test.txt"   
        val_json_file = "/data/zhangwenyao/drive_data/TUSimple/test_set/test_tasks_0627.json"
        val_images_root = test_images_root = "/data/zhangwenyao/drive_data/TUSimple/train_set"

    
        train_transforms, eval_transforms = get_transforms(self.input_transforms,self.input_resize,self.input_size,self.pixel_mean,self.pixel_std)

    
        if mode == 'train':
            self.training_samples = RegressionDataset(train_images_root, train_json_file, train_transforms)
            self.data =DataLoader(self.training_samples, batch_size=self.args.basic.batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=False, num_workers=self.args.hardware.num_workers, 
            pin_memory=True, persistent_workers=not(self.args.debug))
        elif mode == 'online_eval':
            self.testing_samples = RegressionDataset(val_images_root, val_json_file, eval_transforms)
            self.data = DataLoader(self.testing_samples, batch_size=self.args.basic.batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=self.args.hardware.num_workers)
        elif mode == 'test':
            self.testing_samples = RegressionDataset(test_images_root, val_json_file, eval_transforms)
            self.data = DataLoader(self.testing_samples, 1, collate_fn=custom_collate_fn, shuffle=False, num_workers=1)
        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    heights = [item['height'] for item in batch]
    widths = [item['width'] for item in batch]
    return {'image': images, 'height': heights, 'width': widths}


class RegressionDataset(Dataset):
    def __init__(self, images_root, data_file, transforms=None):
        self.images_root = images_root
        self.heights = []
        self.widthes = []
        self.images_file = []
        self.transforms = transforms
        with open(data_file) as fin:
            for line in fin:
                data = json.loads(line)
                self.images_file.append(os.path.join(self.images_root, data['raw_file'])) # record the image file
                self.heights.append(data['h_samples']) # record the column coordinate 
                self.widthes.append(data['lanes'])  # record the row coordinate 

        self.name = osp.splitext(osp.basename(data_file))[0].lower()


        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"Dataset prepare: len of images: {len(self.images_file)}")

    def __getitem__(self, index):
#         print(index)
        img_file, height, width = self.images_file[index], self.heights[index], self.widthes[index]
        full_file = os.path.join(self.images_root, img_file)
        img = Image.open(full_file)

        if img.mode == "L":
            img = img.convert("RGB")

        if self.transforms:
            img = self.transforms(img)
        
        height = torch.tensor(height)
        width = torch.tensor(width)
        sample = {'image':img, 'height': height, 'width': width}

        return sample


    def __len__(self):
        return len(self.images_file)


class LaneDepthMixDataLoader(object):
    def __init__(self, args, mode):
        self.args = args


        train_json_file = "/data/zhangwenyao/drive_data/TUSimple/train_set/train_label_data.json"
        train_images_root = "/data/zhangwenyao/drive_data/TUSimple/train_set/"

        train_depth_file = 'datasets/split_filenames_files/zwy_kitti_eigen_train_files_with_gt.txt'
        train_depth_images_root = '/data/zhangwenyao/drive_data/KITTI/raw/train'
        train_pesudo_gt_root = '/data/ericliu/DepthAnythingV2L-KITTI/depthanything_kitti_eigen_train'
        
        self.input_transforms = ['color_jitter', 'normalize']
        # TODO: 能同时被14和32整除？
        self.input_resize = [224, 224]
        self.pixel_mean =  [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        
    
        # val_data_file = test_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32_order/test_data/data_list/test.txt"   
        val_json_file = "/data/zhangwenyao/drive_data/TUSimple/test_set/test_tasks_0627.json"
        val_images_root = test_images_root = "/data/zhangwenyao/drive_data/TUSimple/train_set"

    
        train_transforms, eval_transforms = get_transforms_new(self.input_transforms,self.input_resize,self.pixel_mean,self.pixel_std, transform_target=False)

    
        if mode == 'train':
            self.training_samples = LaneDepthMixDataset(train_images_root, train_json_file, train_depth_file, train_depth_images_root, train_pesudo_gt_root, train_transforms, tusimple_kitti_mix_ratio=self.args.basic.get('tusimple_kitti_mix_ratio', 0.5))
            self.data =DataLoader(self.training_samples, batch_size=self.args.basic.batch_size, shuffle=True, collate_fn=custom_mix_collate_fn, drop_last=False, num_workers=self.args.hardware.num_workers, pin_memory=True, persistent_workers=not(self.args.debug),)
        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

def custom_mix_collate_fn(batch):
    input_shapes = np.array([item['image'].shape[-2:] for item in batch])
    resize_ratios = np.array([item['resize_ratio'] for item in batch])
    max_shape = input_shapes.max(0)
    batched_images = torch.zeros((len(batch), 3, *max_shape), dtype=batch[0]['image'].dtype)
    for i in range(len(batch)):
        batched_images[i][..., :input_shapes[i][0], :input_shapes[i][1]] = batch[i]['image']
    choices = [item['choice'] for item in batch]
    heights = [item['height'] if 'height' in item else None for item in batch]
    widths = [item['width'] if 'width' in item else None for item in batch]
    depths = [item['depth'] if 'depth' in item else None for item in batch]
    img_files = [item['img_file'] for item in batch]
    return {'input_shapes':input_shapes, 'image': batched_images, 'height': heights, 'width': widths, 'choices': choices, 'depths': depths, 'resize_ratios':resize_ratios, 'img_files': img_files}

class LaneDepthMixDataset(Dataset):
    def __init__(self, images_root, data_file, depth_split_file, depth_image_root, depth_gt_root, transforms=None, tusimple_kitti_mix_ratio=0.5):
        self.images_root = images_root
        self.heights = []
        self.widthes = []
        self.images_file = []
        self.transforms = transforms
        with open(data_file) as fin:
            for line in fin:
                data = json.loads(line)
                self.images_file.append(os.path.join(self.images_root, data['raw_file'])) # record the image file
                self.heights.append(data['h_samples']) # record the column coordinate 
                self.widthes.append(data['lanes'])  # record the row coordinate 

        self.depth_image_files = []
        self.pesudo_depth_files = []
        with open(depth_split_file, 'r') as f:
            for line in f:
                p = line.split(' ')[0]
                self.depth_image_files.append(os.path.join(depth_image_root, p))
                self.pesudo_depth_files.append(os.path.join(depth_gt_root, p[:-4]+'.npy'))
        self.mix_ratio = tusimple_kitti_mix_ratio

        self.name = osp.splitext(osp.basename(data_file))[0].lower()


        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"Dataset prepare: len of images: {len(self.images_file)}")

    def __getitem__(self, _):
#         print(index)
        choice = np.random.uniform(0, 1) > self.mix_ratio
        if choice == 0:
            index = np.random.randint(0, len(self.images_file))
            img_file, height, width = self.images_file[index], self.heights[index], self.widthes[index]
            full_file = os.path.join(self.images_root, img_file)
            img = Image.open(full_file)

            if img.mode == "L":
                img = img.convert("RGB")

            height = torch.tensor(height)
            width = torch.tensor(width)

            ori_width, ori_height = img.size[0], img.size[1]
            if self.transforms:
                target = dict(h_samples=height, lanes=width)
                img, transformed_target = self.transforms(img, target)
                if transformed_target is not None:
                    height = transformed_target['h_samples']
                    width = transformed_target['lanes']
            ratio = img.shape[-2] / ori_height

            sample = {'image':img, 'height':height, 'width':width, 'choice':0, 'resize_ratio':ratio, 'img_file':img_file}
        else:
            def try_fetch(idx):
                img_file, depth_file = self.depth_image_files[index], self.pesudo_depth_files[index]
                if not os.path.exists(img_file):
                    return False
                else:
                    return True
            index = np.random.randint(0, len(self.depth_image_files))
            while not try_fetch(index):
                index = np.random.randint(0, len(self.depth_image_files))
            img_file, depth_file = self.depth_image_files[index], self.pesudo_depth_files[index]
            
            img = Image.open(img_file)
            if img.mode == 'L':
                img = img.convert('RGB')

            depth = torch.from_numpy(np.load(depth_file))

            ori_width, ori_height = img.size[0], img.size[1]
            if self.transforms:
                img, _ = self.transforms(img)
            ratio = img.shape[-2] / ori_height

            sample = {'image':img, 'depth':depth, 'choice':1, 'resize_ratio':ratio, 'img_file':img_file}

        return sample


    def __len__(self):
        return len(self.images_file) + len(self.depth_image_files)



if __name__ == '__main__':
    from easydict import EasyDict
    args = EasyDict()
    args['basic']=EasyDict(batch_size=8)
    args['hardware']=EasyDict(num_workers=1)
    args['debug']=False
    dataloader = LaneDepthMixDataLoader(args=args, mode='train')
    for batch in dataloader.data:
        pass




