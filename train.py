import torch
import argparse
import unet
from torch import optim
from dataset import MyDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train(model):
    model.train()
    batch_size = args.batch_size
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataset = MyDataset("data/train/imgs", "data/train/masks")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        print('-' * 10)
        dataset_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y in train_dataloader:
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.5f" % (step, dataset_size // train_dataloader.batch_size, loss.item()))
        print("epoch %d loss:%0.5f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), 'epoch_%d.pth' % epoch)


def test(model):
    model.eval()
    model.load_state_dict(torch.load(args.load, map_location='cuda'))
    test_dataset = MyDataset("data/test/imgs", "data/test/masks")
    test_dataloader = DataLoader(test_dataset)
    with torch.no_grad():
        k = 0
        for x, _ in test_dataloader:
            y = model(x.to(device))
            img_y = torch.squeeze(y).to('cpu').numpy() > 0.5
            plt.imsave(str(k) + '.png', img_y, format='png', cmap='gray')
            k = k + 1


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet.UNet(1, 1).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, help='the path of the .pth file')
    parser.add_argument('--epoch', dest='num_epochs', type=int, help='num_epochs')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='learning_rate')
    args = parser.parse_args()

    if args.type == 'train':
        train(model)
    elif args.type == 'test':
        test(model)
