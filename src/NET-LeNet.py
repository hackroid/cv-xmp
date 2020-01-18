import torchvision.datasets as ds
import torch
from collections import OrderedDict
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

NUM_WORKER = 4


def disp_progress(count: int, total: int, verbose=True, freq=10, **kwargs):
    LEN = 30
    i = int(count / total * LEN)
    bar = ''.join(['=' * i, '>', '.' * (LEN - i - 1)]) if LEN - i - 1 else '=' * LEN
    msg = ''.join([f'- {k}: {v:.4f} ' for k, v in kwargs.items()])
    out = ''.join(['\r', f'{count + 1}/{total}', ' [', bar, '] ', msg])
    if verbose:
        print(out, end='')
        return

    if count > 0 and total % count == freq:
        print(''.join([f'{count + 1}/{total}', msg]))


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=True)),
            ('act1', nn.Tanh()),
            ('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv3', nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0, bias=True)),
            ('act3', nn.Tanh()),
            ('pool4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('conv5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('act5', nn.Tanh())
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('fc6', nn.Linear(120, 84)),
            ('act6', nn.Tanh()),
            ('fc7', nn.Linear(84, 10)),
            ('act7', nn.Tanh()),
            ('sft7', nn.Softmax())
        ]))

    def forward(self, img):
        out = self.conv(img)
        out = out.view(img.size(0), -1)
        out = self.fc(out)
        return out


class Model(object):
    def __init__(self, model: nn.Module, ngpus=1):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)

        self.epoch_training_loss = None

        self.use_gpu = torch.cuda.is_available() and ngpus
        self.ngpus = ngpus

    def epoch_init(self):
        self.epoch_training_loss = 0
        pass

    def step(self, x: Tensor, y: Tensor):
        self.model.train()
        if self.use_gpu:
            x.cuda()
            y.cuda()
            x.requires_grad = True
            y.requires_grad = True
            # x = torch.autograd.Variable(x.cuda())
            # y = torch.autograd.Variable(y.cuda())

        # Make gradients zero for parameters 'W', 'b'
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_pred_prob = self.model(x)
            print(y_pred_prob)
        return 1, 2

    def load(self):
        pass

    def save(self):
        pass


def load_data(batch_size=32, valid_batch_size=32):
    train_transformations = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transformations = transforms.Compose([
        transforms.ToTensor(),
    ])
    kwargs_dl = {'root': '../data', 'download': True}
    train_set = ds.MNIST(train=True, transform=train_transformations, **kwargs_dl)
    test_set = ds.MNIST(train=False, transform=test_transformations, **kwargs_dl)
    kwargs_train = {'shuffle': True, 'batch_size': batch_size, 'num_workers': NUM_WORKER}
    kwargs_test = {'shuffle': True, 'batch_size': valid_batch_size, 'num_workers': NUM_WORKER}
    train_set = DataLoader(train_set, **kwargs_train)
    test_set = DataLoader(test_set, **kwargs_test)
    return train_set, test_set


def fit(
        model,
        dataloader,
        max_epochs,
        batch_size,
        valid_batch_size,
        verbose=True,
        disp_freq=10
):
    print("Using Torch backend")
    assert isinstance(model, Model)

    train_set, test_set = dataloader(batch_size, valid_batch_size)

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}/{max_epochs}')
        for i, (x, y) in enumerate(train_set):
            train_loss, train_acc = model.step(x, y)
            kwargs = {'loss': train_loss, 'acc': train_acc}
            disp_progress(i, len(train_set), verbose, disp_freq, **kwargs)


def main():
    model = Model(LeNet())
    fit(model, load_data, max_epochs=5, batch_size=32, valid_batch_size=32)


if __name__ == '__main__':
    main()
