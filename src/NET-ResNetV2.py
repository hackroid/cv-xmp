import torchvision.datasets as ds
import torch
import time
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from collections import OrderedDict
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

NUM_WORKER = 4
torch.manual_seed(66)
writer = SummaryWriter(os.path.join("..", "result", "runs"))


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.ct = 1
        self.in_chan = 64
        self.conv0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, self.in_chan, kernel_size=3, stride=1, padding=1, bias=True)),
            ('bn0', nn.BatchNorm2d(self.in_chan)),
            ('act0', nn.ReLU()),
        ]))

        self.layer0b0 = self.gen_unit(64, 192, 0, 0, False)
        self.layer0b1 = self.gen_unit(128, 192, 0, 1, False)
        self.layer0cut = self.gen_unit(64, 192, 0, 1, False, True)

        self.in_chan = 192
        self.layer1b0 = self.gen_unit(64, 192, 1, 0, True)
        self.layer1b1 = self.gen_unit(128, 192, 1, 1, True)

        self.layer2b0 = self.gen_unit(64, 192, 2, 0, True)
        self.layer2b1 = self.gen_unit(128, 192, 2, 1, True)

        self.layer3b0 = self.gen_unit(64, 192, 3, 0, True)
        self.layer3b1 = self.gen_unit(128, 192, 3, 1, True)

        self.layer4b0 = self.gen_unit(64, 192, 4, 0, True)
        self.layer4b1 = self.gen_unit(128, 192, 4, 1, True)

        self.layer5b0 = self.gen_unit(64, 192, 5, 0, True)
        self.layer5b1 = self.gen_unit(128, 192, 5, 1, True)

        self.layer6b0 = self.gen_unit(64, 192, 6, 0, True)
        self.layer6b1 = self.gen_unit(128, 192, 6, 1, True)

        self.layer7b0 = self.gen_unit(64, 192, 7, 0, True)
        self.layer7b1 = self.gen_unit(128, 192, 7, 1, True)

        self.layer8b0 = self.gen_unit(64, 192, 8, 0, True)
        self.layer8b1 = self.gen_unit(128, 192, 8, 1, True)

        self.layer9b0 = self.gen_unit(64, 192, 9, 0, True)
        self.layer9b1 = self.gen_unit(128, 192, 9, 1, True)

        self.layer10b0 = self.gen_unit(64, 192, 10, 0, True)
        self.layer10b1 = self.gen_unit(128, 192, 10, 1, True)

        self.layer11b0 = self.gen_unit(64, 192, 11, 0, True)
        self.layer11b1 = self.gen_unit(128, 192, 11, 1, True)

        self.end = nn.Sequential(OrderedDict([
            ('end-bn', nn.BatchNorm2d(self.in_chan)),
            ('end-a', nn.ReLU()),
        ]))
        self.fc = nn.Linear(3072, 10)

    def gen_unit(self, passway: int, pass_out: int, n_layer, br: int, bn: bool, shortcut=False):
        if shortcut:
            return nn.Sequential(OrderedDict([
                ('L{}-cut'.format(n_layer, br), nn.Conv2d(self.in_chan, pass_out, kernel_size=1, bias=True))
            ]))
        scale = pow(2, br)
        layer_list = [('L{}b{}-down'.format(n_layer, br), nn.MaxPool2d(kernel_size=scale, stride=scale))]
        if not bn:
            layer_list.append(
                ('L{}b{}-c1'.format(n_layer, br), nn.Conv2d(self.in_chan, passway, kernel_size=1, bias=True)))
        else:
            layer_list.append(('L{}b{}-bn1'.format(n_layer, br), nn.BatchNorm2d(self.in_chan)))
            layer_list.append(
                ('L{}b{}-c1'.format(n_layer, br), nn.Conv2d(self.in_chan, passway, kernel_size=1, bias=True)))
            layer_list.append(
                ('L{}b{}-a1'.format(n_layer, br), nn.ReLU(inplace=True)))

        layer_list.append(('L{}b{}-bn2'.format(n_layer, br), nn.BatchNorm2d(passway)))
        layer_list.append(('L{}b{}-c2'.format(n_layer, br),
                           nn.Conv2d(passway, passway, kernel_size=3, stride=1, padding=1, bias=True)))
        layer_list.append(('L{}b{}-a2'.format(n_layer, br), nn.ReLU(inplace=True)))

        layer_list.append(('L{}b{}-bn3'.format(n_layer, br), nn.BatchNorm2d(passway)))
        layer_list.append(('L{}b{}-c3'.format(n_layer, br), nn.Conv2d(passway, pass_out, kernel_size=1, bias=True)))
        layer_list.append(('L{}b{}-a3'.format(n_layer, br), nn.ReLU(inplace=True)))

        layer_list.append(('L{}b{}-up'.format(n_layer, br), nn.UpsamplingBilinear2d(scale_factor=scale)))

        ResUnit = nn.Sequential(OrderedDict(layer_list))
        return ResUnit

    def forward(self, x):
        x = self.conv0(x)
        x = (self.layer0b0(x) + self.layer0b1(x) + self.layer0cut(x))
        x = (self.layer1b0(x) + self.layer1b1(x))
        x = (self.layer2b0(x) + self.layer2b1(x))
        x = (self.layer3b0(x) + self.layer3b1(x))
        x = (self.layer4b0(x) + self.layer4b1(x))
        x = (self.layer5b0(x) + self.layer5b1(x))
        x = (self.layer6b0(x) + self.layer6b1(x))
        x = (self.layer7b0(x) + self.layer7b1(x))
        x = (self.layer8b0(x) + self.layer8b1(x))
        x = (self.layer9b0(x) + self.layer9b1(x))
        x = (self.layer10b0(x) + self.layer10b1(x))
        x = (self.layer11b0(x) + self.layer11b1(x))
        x = self.end(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Model(object):
    def __init__(self, model: nn.Module, ngpus=1):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)

        self.history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
        self.best_loss = 2e16 - 1
        self.best_corr = 0.0
        self.best_model = None

        self.epoch_num = 0
        self.meet_n_samples = None
        self.epoch_training_loss = None
        self.epoch_training_corr = None
        self.eval_n_samples = None
        self.epoch_eval_loss = None
        self.epoch_eval_corr = None

        self.use_gpu = torch.cuda.is_available() and ngpus
        self.ngpus = ngpus

        if self.use_gpu and ngpus == 1:
            torch.cuda.set_device(0)
            self.model.cuda()
            print('Using {}'.format(torch.cuda.get_device_name()))
        else:
            # raise PermissionError('There are not so many GPUs.')
            pass

    def epoch_init(self):
        self.epoch_num += 1
        if self.epoch_num > 1:
            self.history['loss'].append(
                self.epoch_training_loss / self.meet_n_samples)
            self.history['acc'].append(
                self.epoch_training_corr / self.meet_n_samples)
            self.history['val_loss'].append(
                self.epoch_eval_loss / self.eval_n_samples)
            self.history['val_acc'].append(
                self.epoch_eval_corr / self.eval_n_samples)

        self.epoch_training_loss = 0.0
        self.epoch_training_corr = 0
        self.meet_n_samples = 0
        self.epoch_eval_loss = 0.0
        self.epoch_eval_corr = 0
        self.eval_n_samples = 0

    def step(self, x: Tensor, y: Tensor):
        self.model.train()
        if self.use_gpu:
            x = torch.autograd.Variable(x.cuda())
            y = torch.autograd.Variable(y.cuda())

        # Make gradients zero for parameters 'W', 'b'
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_pred_prob = self.model(x)
            y_pred = torch.max(y_pred_prob.data, 1).indices
            loss = self.loss_fn(y_pred_prob, y)
            loss.backward()
            self.optimizer.step()
        self.meet_n_samples += x.size(0)
        self.epoch_training_loss += loss.item() * x.size(0)
        self.epoch_training_corr += torch.sum(y_pred == y.data).double()
        mean_loss = self.epoch_training_loss / self.meet_n_samples
        mean_acc = self.epoch_training_corr / self.meet_n_samples
        return mean_loss, mean_acc

    def eval(self, x, y):
        self.model.eval()
        if self.use_gpu:
            x = torch.autograd.Variable(x.cuda())
            y = torch.autograd.Variable(y.cuda())

        with torch.set_grad_enabled(False):
            y_pred_prob = self.model(x)
            y_pred = torch.max(y_pred_prob.data, 1).indices
            loss = self.loss_fn(y_pred_prob, y)

        self.eval_n_samples += x.size(0)
        self.epoch_eval_loss += loss.item() * x.size(0)
        self.epoch_eval_corr += torch.sum(y_pred == y.data).item()
        mean_loss = self.epoch_eval_loss / self.eval_n_samples
        mean_corr = self.epoch_eval_corr / self.eval_n_samples

        if mean_corr > self.best_corr:
            d = {k: p.clone().cpu().numpy()
                 for k, p in self.model.state_dict().items()}
            self.best_loss = mean_loss
            self.best_corr = mean_corr
            self.best_model = d
        return mean_loss, mean_corr

    def load(self):
        pass

    def save(self):
        pass


def imshow(img):
    plt.imshow(img)
    plt.show()


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


def load_data(batch_size=32, valid_batch_size=32):
    train_transformations = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    kwargs_dl = {'root': '../data', 'download': True}
    train_set = ds.CIFAR10(train=True, transform=train_transformations, **kwargs_dl)
    test_set = ds.CIFAR10(train=False, transform=test_transformations, **kwargs_dl)
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
    ct = 0
    train_set, test_set = dataloader(batch_size, valid_batch_size)

    for epoch in range(max_epochs):
        print(f'Epoch {epoch + 1}/{max_epochs}')

        start_time = time.time()
        model.epoch_init()

        for i, (x, y) in enumerate(train_set):
            ct += 1
            train_loss, train_acc = model.step(x, y)
            kwargs = {'loss': train_loss, 'acc': train_acc}
            disp_progress(i, len(train_set), verbose, disp_freq, **kwargs)
            writer.add_scalars(
                "train_loss&acc",
                {
                    "loss": train_loss,
                    "acc": train_acc
                },
                ct
            )

        print('\n', end='')

        test_loss = 0.0
        test_acc = 2e16 - 1
        for i, (x, y) in enumerate(test_set):
            _test_loss, _test_acc = model.eval(x, y)
            test_loss = (test_loss if test_loss >= _test_loss else _test_loss)
            test_acc = (test_acc if test_acc <= _test_acc else _test_acc)

        time_cost = round(time.time() - start_time)
        msg_1 = f' - val_loss: {test_loss:.4f} - val_acc: {test_acc:.4f}'
        msg_2 = f' - {time_cost}s/epoch.'
        print(''.join([msg_1, msg_2]))


def main():
    model = Model(ResNet())
    print(model.model)
    fit(model, load_data, max_epochs=5, batch_size=32, valid_batch_size=32)
    writer.close()


if __name__ == '__main__':
    main()
