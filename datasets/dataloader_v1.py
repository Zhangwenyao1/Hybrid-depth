# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications
# This version has been further modified from the AdaBins version; modifications are only to accommodate the
# differen args format.

import os
import random
import sys

import numpy as np
import torch
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os.path as osp
# from utils import get_transforms
from utils.utils import get_transforms



def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, args, mode):
        self.args = args
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            # if args.distributed:
            #     self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            # else:
            #     self.train_sampler = None
            if self.args.basic.dataset == 'tusimple':
                self.train_shuffle = False
            else:
                self.train_shuffle = True
            self.data = DataLoader(self.training_samples, batch_size=self.args.basic.batch_size,
                                #    shuffle=(self.train_sampler is None),
                                shuffle = False,
                                drop_last=False,
                                num_workers=self.args.hardware.num_workers,
                                pin_memory=True,
                                persistent_workers=not(self.args.debug)
                                #    sampler=self.train_sampler
                                   )

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            # if args.distributed:  # redundant. here only for readability and to be more explicit
            #     # Give whole test set to all processes (and perform/report evaluation only on one) regardless
            #     self.eval_sampler = None
            # else:
            #     self.eval_sampler = None
            if self.args.get("validate") or self.args.get("inference"):
                assert self.args.basic.batch_size == 1, "ERROR: validation mode batch size is not 1 and should be."
            self.data = DataLoader(self.testing_samples, batch_size=self.args.basic.batch_size,
                                    shuffle=False,
                                    num_workers=1,
                                    pin_memory=False,
                                    drop_last=False,
                                    persistent_workers=not(self.args.debug)
                                #    sampler=self.eval_sampler
                                   )

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(self.args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(self.args[self.args.basic.dataset].filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(self.args[self.args.basic.dataset].filenames_file_train, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

        self.base_path = os.path.join(self.args.paths.data_dir, self.args[self.args.basic.dataset].base_path)
        if self.args.basic.dataset == "kitti":
            self.data_path = os.path.join(self.base_path, self.args.kitti.data_path)
            self.gt_path = os.path.join(self.base_path, self.args.kitti.gt_path)
        elif self.args.basic.dataset == "nyu":
            self.train_path = os.path.join(self.base_path, self.args.nyu.train_path)
            self.eval_path = os.path.join(self.base_path, self.args.nyu.eval_path)
            if self.mode == "train":
                self.data_path = self.train_path
            else:
                self.data_path = self.eval_path
            self.gt_path = self.data_path   # Same as data_path for NYU, different for KITTI.
            
            
        # for tusimple    
            
        elif self.args.basic.dataset == "tusimple":
            # self.data_path = os.path.join(self.base_path, self.args.tusimple.data_path)
            # self.gt_path = os.path.join(self.base_path, self.args.tusimple.gt_path)
            # if self.mode == "train":
            #     self.data_path = self.train_path
            # else:
            #     self.data_path = self.eval_path

            train_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32/train_data/data_list/train.txt"
            train_images_root = "/data/zhangwenyao/drive_data/TUSimple/process_32/train_data"
            self.train_data_file = train_data_file
            self.train_images_root = train_images_root

            self.train_labels = []
            self.train_images_file = []
            
            
            self.input_transforms = ['random_resized_crop', 'random_hflip', 'normalize']
            self.input_resize = [256,256]
            self.input_size = [224,224]
            self.pixel_mean =  [0.485, 0.456, 0.406]
            self.pixel_std = [0.229, 0.224, 0.225]
            
            
            self.train_transforms, self.eval_transforms = get_transforms(self.input_transforms,self.input_resize,self.input_size,self.pixel_mean,self.pixel_std)
            with open(train_data_file) as fin:
                for line in fin:
                    # image_file, image_label = line.split()
                    splits = line.split()
                    train_image_file = splits[0]
                    labels = splits[1:]
                    # self.labels.append([int(label) for label in labels])
                    # self.train_labels.append([label for label in labels])
                    if len(labels)>1:
                        self.train_labels.append(labels[0]+' '+labels[1])
                    else:
                        self.train_labels.append([label for label in labels])
                    self.train_images_file.append(train_image_file)

            self.train_name = osp.splitext(osp.basename(train_data_file))[0].lower()
            
            
            test_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data/data_list/test.txt"   
            test_images_root = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data"
            self.test_data_file = test_data_file
            self.test_images_root = test_images_root    
            self.test_labels = []
            self.test_images_file = []
                 
            with open(test_data_file) as fin:
                for line in fin:
                    # image_file, image_label = line.split()
                    splits = line.split()
                    test_image_file = splits[0]
                    labels = splits[1:]
                    # self.labels.append([int(label) for label in labels])
                    # self.test_labels.append([label for label in labels])
                                    # if len(labels)>1:
                    if len(labels)>1:
                        self.test_labels.append(labels[0]+' '+labels[1])
                    else:
                        self.test_labels.append([label for label in labels])
                    self.test_images_file.append(test_image_file)

            self.test_name = osp.splitext(osp.basename(test_data_file))[0].lower()
            
            
            
            
            
            
            
            
            
            # if "val" in self.name or "test" in self.name:
            #     print(f"Dataset prepare: val/test data_file: {data_file}")
            # elif "train" in self.name:
            #     print(f"Dataset prepare: train data_file: {train_data_file}")
            # else:
            #     raise ValueError(f"Invalid data_file: {data_file}")
            # print(f"Dataset prepare: len of labels: {len(self.labels[0])}")
            # print(f"Dataset prepare: len of dataset: {len(self.labels)}")

            # train_images_root = "/data/zhangwenyao/drive_data/TUSimple/process_32/train_data"
            # val_images_root = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data"
            # test_images_root = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data"
            
            # val_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data/data_list/test.txt"
            # test_data_file = "/data/zhangwenyao/drive_data/TUSimple/process_32/test_data/data_list/test.txt"

            
            
            
        if self.mode == "train":
            self.input_height, self.input_width = self.args[self.args.basic.dataset].dimensions_train
        else:
            self.input_height, self.input_width = self.args[self.args.basic.dataset].dimensions_test


    def __getitem__(self, idx):
        print(idx)
        if idx >= len(self.filenames):
            raise StopIteration
        sample_path = self.filenames[idx]
        if  isinstance(sample_path.split()[1], str):
            focal = 0
        else:
            focal = float(sample_path.split()[2])
            
        if self.mode == 'train' and self.args.basic.dataset != 'tusimple':
            if self.args.basic.dataset == 'kitti' and self.args.kitti.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(self.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                if self.args.basic.dataset == 'kitti':
                    image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[0]))
                    depth_path = os.path.join(self.gt_path, remove_leading_slash(sample_path.split()[1]))
                elif self.args.basic.dataset == 'nyu':
                    image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[0]))
                    depth_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[1]))
                else:
                    sys.exit("Error: unrecognised dataset")


            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            if self.args[self.args.basic.dataset].do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            if self.args.basic.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if self.args[self.args.basic.dataset].do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args[self.args.basic.dataset].degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.basic.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0

            image, depth_gt = self.random_crop(image, depth_gt, self.input_height, self.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
            if self.transform:
                sample = self.transform(sample)  
            
        # for tusimple 
        elif self.mode == 'train' and self.args.basic.dataset == 'tusimple':
            img_file, target_list = self.train_images_file[idx], self.train_labels[idx]
            target = random.choice(target_list)
            full_file = os.path.join(self.train_images_root, img_file)
            img = Image.open(full_file)

            if img.mode == "L":
                img = img.convert("RGB")

            if self.train_transforms:
                img = self.train_transforms(img)


            # return img, target
            sample = {'image':img, 'depth': target}


          
            
            
            
            # sample = {'image':image, 'depth': depth_gt}
        
        else:
            if self.args.basic.dataset == 'tusimple':
                img_file, target_list = self.test_images_file[idx], self.test_labels[idx]
                target = random.choice(target_list)
                full_file = os.path.join(self.test_images_root, img_file)
                img = Image.open(full_file)
            # img = Image.open(full_file)

                if img.mode == "L":
                    img = img.convert("RGB")

                if self.train_transforms:
                    img = self.train_transforms(img)


                # return img, target
                sample = {'image':img, 'depth': target}

     
            
            
            else:
                # if self.mode == 'online_eval':
                #     data_path = self.args.data_path_eval
                # else:
                #     data_path = self.args.data_path
                data_path = self.data_path  # This was set for the current split in the init.

                image_path = os.path.join(data_path, remove_leading_slash(sample_path.split()[0]))
                image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

                if self.mode == 'online_eval':
                    gt_path = self.gt_path
                    depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
                    has_valid_depth = False
                    try:
                        depth_gt = Image.open(depth_path)
                        has_valid_depth = True
                    except IOError:
                        depth_gt = False
                        # print('Missing gt for {}'.format(image_path))
                        del self.filenames[idx]
                        return self.__getitem__(idx)

                    if has_valid_depth:
                        depth_gt = np.asarray(depth_gt, dtype=np.float32)
                        depth_gt = np.expand_dims(depth_gt, axis=2)
                        if self.args.basic.dataset == 'nyu':
                            depth_gt = depth_gt / 1000.0
                        else:
                            depth_gt = depth_gt / 256.0

                if self.args[self.args.basic.dataset].do_kb_crop is True:
                    height = image.shape[0]
                    width = image.shape[1]
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    if self.mode == 'online_eval' and has_valid_depth:
                        depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

                if self.mode == 'online_eval':
                    sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                            'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}
                else:
                    sample = {'image': image, 'focal': focal}

                if self.transform:
                    sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.basic.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
