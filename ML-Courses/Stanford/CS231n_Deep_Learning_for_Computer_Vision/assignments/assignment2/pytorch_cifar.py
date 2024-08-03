import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.transforms as T


def main():
    dtype = torch.cuda.FloatTensor

    cifar10_train = dset.CIFAR10(".", train=True, download=True, transform=T.ToTensor())
    loader_train = DataLoader(cifar10_train, batch_size=64)

    cifar10_test = dset.CIFAR10(".", train=False, download=True, transform=T.ToTensor())
    loader_test = DataLoader(cifar10_test, batch_size=64)

    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=2, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        Flatten(),
        nn.Linear(2048, 1024),
        nn.Dropout(p=0.5),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 10),
    ).type(dtype)
    loss_fn = nn.CrossEntropyLoss().type(dtype)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5
    print_every = 100

    for epoch in range(num_epochs):
        print("Starting epoch %d / %d" % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())

            scores = model(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print("t = %d, loss = %.4f" % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Checking accuracy on test set")
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader_test:
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print("Got %d / %d correct (%.2f)" % (num_correct, num_samples, 100 * acc))


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


if __name__ == "__main__":
    main()
