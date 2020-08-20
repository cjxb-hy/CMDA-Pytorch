import os
import glob
import time
from torchvision.transforms.transforms import Resize
import visdom
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Data_Loader(Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        self.img_path = glob.glob(os.path.join(
            data_path, 'image/*.png'))

    def __getitem__(self, index):

        image_path = self.img_path[index]
        label_path = image_path.replace('image', 'label')

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)  # .convert('RGB')

        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        tf1 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.ToTensor()
        ])

        image, label = tf(image), tf1(label)

        return image, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    dataset = Data_Loader("data/train/")
    print("数据个数：", len(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=2, shuffle=True)

    for x, y in train_loader:
        print(x.shape, y.shape)

    # vis = visdom.Visdom()

    # for x, y in train_loader:
    #     vis.images(x, nrow=1, win='batch', opts=dict(title='batch'))
    #     vis.images(y, nrow=1, win='batch1', opts=dict(title='batch1'))

    #     time.sleep(1)
