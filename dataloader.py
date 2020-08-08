import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import yaml
from glob import glob
import numpy as np
import random
 
normalize = transforms.Normalize(mean=[0.47103423, 0.44983818, 0.40353369], std=[0.27600256, 0.26770896, 0.28263377]) # MiniImagenet

class MetaDataset(Dataset):
    def __init__(self, config, mode="train", n_way=5, n_shot=5, n_query=10, transform=None):
        super(MetaDataset, self).__init__()

        self.config = config
        self.mode = mode
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.transform = transform

        data_root = os.path.join(self.config.dataset_root, mode)
        class_names = os.listdir(data_root)
        self.class_paths = [os.path.join(data_root, class_name) 
                            for class_name in class_names]

    def __len__(self):
        if self.mode == "train":
            return self.config.n_episode_each_epoch 
        else:
            return self.config.test_episode
     
    def __getitem__(self, idx):
        image_size = [self.config.IMAGE_SIZE, self.config.IMAGE_SIZE]
        sample_images = torch.Tensor(self.n_way * self.n_shot, 3, *image_size)
        query_images = torch.Tensor(self.n_way * self.n_query, 3, *image_size)
        sample_id = 0
        query_id = 0

        chosen_classes = random.sample(self.class_paths, self.n_way)
        for chosen_class in chosen_classes:
            image_names = os.listdir(chosen_class)
            chosen_images = random.sample(image_names, self.n_shot + self.n_query)
            chosen_image_paths = [os.path.join(chosen_class, chosen_image) 
                                    for chosen_image in chosen_images]
            for i, chosen_image_path in enumerate(chosen_image_paths):
                image = Image.open(chosen_image_path).convert("RGB").resize(image_size, Image.ANTIALIAS)
                if self.transform is not None:
                    image = self.transform(image)
                if i < self.n_shot:
                    sample_images[sample_id] = image
                    sample_id += 1
                else:
                    query_images[query_id] = image
                    query_id += 1
        return sample_images, query_images

def get_dataloader(config, mode="train", n_way=5, n_shot=5, n_query=15, num_workers=3):
    dataset = MetaDataset(config, mode=mode, n_way=n_way, n_shot=n_shot, n_query=n_query,
                          transform=transforms.Compose([transforms.ToTensor(), normalize]))

    loader = DataLoader(dataset, num_workers=num_workers)

    return loader

