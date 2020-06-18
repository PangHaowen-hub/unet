import torch.utils.data as data
import os
import PIL.Image as Image
from torchvision.transforms import transforms


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if (os.path.splitext(file)[1] == '.png'):
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class MyDataset(data.Dataset):
    def __init__(self, root_imgs, root_masks):
        imgs = []
        img = get_listdir(root_imgs)
        mask = get_listdir(root_masks)
        n = len(img)
        for i in range(n):
            imgs.append([img[i], mask[i]])
        self.imgs = imgs
        self.images_transform = transforms.ToTensor()
        self.masks_transform = transforms.ToTensor()

    def __getitem__(self, index):
        images_path, masks_path = self.imgs[index]
        image = self.images_transform(Image.open(images_path))
        mask = self.masks_transform(Image.open(masks_path))
        return image, mask

    def __len__(self):
        return len(self.imgs)
