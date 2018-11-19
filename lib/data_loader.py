import torch
import os
from torch.utils import data
from PIL import Image
from torchvision import transforms


class TestDataSetDefine(data.Dataset):
    def __init__(self, img_list, label_list, config):
        self.img_list = img_list
        self.label_list = label_list
        self.cfg = config

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        """
        :param index:
        :return: (1, 3, h, w), (1, 3, h, w)
        """
        img_name = self.img_list[index]
        img_path = os.path.join(self.cfg.test_dir, self.img_list[index])
        label_path = os.path.join(self.cfg.test_dir, self.label_list[index])

        with Image.open(img_path) as img:
            img = img.convert('RGB')
        img = self.transform(img)

        with Image.open(label_path) as label:
            label = label.convert('RGB')
        label = self.transform(label)

        assert img.size() == label.size()
        return img, label, img_name

    def __len__(self):
        len_0 = len(self.img_list)
        len_1 = len(self.label_list)

        assert len_0 == len_1
        return len_0


class DataSet(object):
    def __init__(self, config):
        self.cfg = config

        self.test_img_list, self. test_label_list = self.get_list(self.cfg.test_dir)

        self.test_dataset = TestDataSetDefine(self.test_img_list, self.test_label_list, self.cfg)
        self.test_loader = data.DataLoader(dataset=self.test_dataset,
                                           batch_size=self.cfg.test_batch_size,
                                           shuffle=False,)

    @staticmethod
    def get_list(dir):
        img_list = []
        label_list = []

        name_list = os.listdir(dir)
        for name in name_list:
            if 'no' not in name:
                img_list.append(name)
                label_list.append('no' + name)
        return img_list, label_list

