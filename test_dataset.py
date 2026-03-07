from utils.utils import get_transforms
from datasets.dataloader import RegressionDataset

class Dataloader:
    def __init__(self) -> None:
        
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

        self.training_samples = RegressionDataset(train_images_root, train_json_file, train_transforms)

dataloader = Dataloader()
dataloader.training_samples[0]