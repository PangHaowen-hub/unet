import torch
from torchvision.transforms import transforms
import argparse
import unet
from torch import optim
from dataset import LiverDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import PIL.Image as Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, ], [0.5, ])])
masks_transform = transforms.ToTensor()


def train_model(model, loss_fn, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), 'epoch_%d.pth' % epoch)
    return model


def train(model):
    batch_size = args.batch_size
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train/imgs", "data/train/masks", images_transform=images_transform,
                                 masks_transform=masks_transform)
    train_dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, loss_fn, optimizer, train_dataloader)


def test(model):
    model.load_state_dict(torch.load(args.weight, map_location='cuda'))
    liver_dataset = LiverDataset("data/test/imgs", "data/test/masks", images_transform=images_transform,
                                 masks_transform=masks_transform)
    test_dataloader = DataLoader(liver_dataset)  # batch_size默认为1
    model.eval()
    with torch.no_grad():
        k = 0
        '''
        for x, _ in test_dataloader:
            y = model(x.to(device))
            img_y = torch.squeeze(y).to('cpu').numpy() * 255
            img_y = img_y < 0.9
            print(img_y)
            img = Image.fromarray(img_y)  #.convert('L')
            print(img)
            img.save(str(k)+'.png')
            k = k+1
        '''
        for x, _ in test_dataloader:
            y = model(x.to(device))
            img_y = torch.squeeze(y).to('cpu').numpy()
            plt.imshow(img_y)
            plt.savefig(str(k) + '.png')
            k = k + 1

if __name__ == '__main__':
    model = unet.UNet(1, 1).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('-action', type=str, help='train or test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    args = parser.parse_args()

    if args.action == 'train':
        train(model)
    elif args.action == 'test':
        test(model)
