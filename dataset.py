import torch.utils.data as data
import os
import PIL.Image as Image


def get_listdir(path):  # 获取目录下所有png格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if (os.path.splitext(file)[1] == '.png'):
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class LiverDataset(data.Dataset):
    def __init__(self, root_imgs, root_masks, images_transform=None, masks_transform=None):  # root表示图片路径
        imgs = []
        img = get_listdir(root_imgs)
        mask = get_listdir(root_masks)
        n = len(img)
        for i in range(n):
            imgs.append([img[i], mask[i]])

        self.imgs = imgs
        self.images_transform = images_transform
        self.masks_transform = masks_transform

    def __getitem__(self, index):
        images_path, masks_path = self.imgs[index]
        image = self.images_transform(Image.open(images_path).convert('L'))
        mask = self.masks_transform(Image.open(masks_path).convert('L'))
        return image, mask

    def __len__(self):
        return len(self.imgs)
