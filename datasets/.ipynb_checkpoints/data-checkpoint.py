import math
import os
import os.path as osp
import random
from collections import defaultdict

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# from crowdclip.utils.logging import get_logger, print_log

from utils.utils import get_transforms

# logger = get_logger(__name__)
# print = lambda x: print_log(x, logger=logger)

# torch.cuda.set_device(1)

class RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        # train_images_root,
        # val_images_root,
        # test_images_root,
        # train_data_file,
        # val_data_file,
        # test_data_file,
        args,
        mode,
        transforms_cfg=None,
        train_dataloder_cfg=None,
        eval_dataloder_cfg=None,
        few_shot=None,
        label_distributed_shift=None,
        use_long_tail=False,
        
    ):
        super().__init__()
        self.args = args
        train_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32/train_data/data_list/train.txt"
        train_images_root = "/data/zhangwenyao/drive_data/TUSimple/process_32/train_data"
        # self.train_data_file = train_data_file
        # self.train_images_root = train_images_root

        # self.train_labels = []
        # self.train_images_file = []
        
        
        self.input_transforms = ['random_resized_crop', 'random_hflip', 'normalize']
        self.input_resize = [256,256]
        self.input_size = [224,224]
        self.pixel_mean =  [0.485, 0.456, 0.406]
        self.pixel_std = [0.229, 0.224, 0.225]
        
    
        val_data_file = test_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data/data_list/test.txt"   
        val_images_root = test_images_root = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data"
        # self.test_data_file = test_data_file
        # self.test_images_root = test_images_root    
        # self.test_labels = []
        # self.test_images_file = []
        # train_dataloder_cfg = {'num_workers': 8, 'batch_size': 7, 'shuffle': False}
        # eval_dataloder_cfg = {'num_workers': 8, 'batch_size': 7, 'shuffle': False}
        # few_shot = {'num_shots': 0}
        # label_distributed_shift = {'num_topk_scaled_class': 0, 'scale_factor': 1.0}
        # use_long_tail = False   
        
        
        
        train_transforms, eval_transforms = get_transforms(self.input_transforms,self.input_resize,self.input_size,self.pixel_mean,self.pixel_std)

        self.train_set = RegressionDataset(train_images_root, train_data_file, train_transforms)
        self.val_set = RegressionDataset(val_images_root, val_data_file, eval_transforms)
        self.test_set = RegressionDataset(test_images_root, test_data_file, eval_transforms)

        # print(self.test_set)

        # self.train_set.generate_fewshot_dataset(**few_shot)
        # self.train_set.generate_distribution_shifted_dataset(**label_distributed_shift)
        # if use_long_tail:
        #     self.val_set.generate_long_tail()
        #     self.test_set.generate_long_tail()

        # self.train_dataloder_cfg = train_dataloder_cfg
        # self.eval_dataloder_cfg = eval_dataloder_cfg
        
        
        
        
        if mode == 'train':
            self.training_samples = RegressionDataset(train_images_root, train_data_file, train_transforms)
            self.data =DataLoader(self.training_samples, batch_size=self.args.basic.batch_size, shuffle=False, drop_last=False, num_workers=self.args.hardware.num_workers, pin_memory=True, persistent_workers=not(self.args.debug),)
        elif mode == 'online_eval':
            self.testing_samples = RegressionDataset(test_images_root, test_data_file, eval_transforms)
            self.data = DataLoader(self.testing_samples, batch_size=self.args.basic.batch_size, shuffle=False, num_workers=self.args.hardware.num_workers)
        elif mode == 'test':
            self.testing_samples = RegressionDataset(test_images_root, test_data_file, eval_transforms)
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.train_dataloder_cfg)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg)


class RegressionDataset(Dataset):
    def __init__(self, images_root, data_file, transforms=None):
        self.images_root = images_root
        self.labels = []
        self.images_file = []
        self.transforms = transforms

        with open(data_file) as fin:
            for line in fin:
                # image_file, image_label = line.split()
                splits = line.split()
                image_file = splits[0]
                labels = splits[1:]
                if len(labels)>1:
                    self.labels.append(labels[0] +' '+ labels[1])
                else:
                    self.labels.append(labels) 
                # self.labels.append([int(label) for label in labels])
                # self.labels.append([label for label in labels])
                # self.labels.append(labels)
                self.images_file.append(image_file)

        self.name = osp.splitext(osp.basename(data_file))[0].lower()
        if "val" in self.name or "test" in self.name:
            print(f"Dataset prepare: val/test data_file: {data_file}")
        elif "train" in self.name:
            print(f"Dataset prepare: train data_file: {data_file}")
        else:
            raise ValueError(f"Invalid data_file: {data_file}")
        print(f"Dataset prepare: len of labels: {len(self.labels)}")
        print(f"Dataset prepare: len of dataset: {len(self.labels)}")

    def __getitem__(self, index):
#         print(index)
        img_file, target_list = self.images_file[index], self.labels[index]
        if "val" in self.name or "test" in self.name:
            target = target_list[len(target_list) // 2]
        else:
            target = random.choice(target_list)
        # print(self.images_file[index])
        # img_file, target = self.images_file[index], self.labels[index]
        full_file = os.path.join(self.images_root, img_file)
        # print(full_file)
        img = Image.open(full_file)
        # img = Image.open(full_file)

        if img.mode == "L":
            img = img.convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        sample = {'image':img, 'depth': target}

        return sample


















    def __len__(self):
        return len(self.images_file)
