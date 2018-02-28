from torch import nn, autograd
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch


class NucleiModel(nn.Module):
    def __init__(self, input_shape):
        super(NucleiModel, self).__init__()
        self.input_shape = input_shape
        self.input_channel = input_shape[2]
        self.conv_1_1 = nn.Conv2d(self.input_channel, 32, 3, padding=1)
        self.conv_1_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_1_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_2_1 = nn.Conv2d(self.input_channel, 32, 3, padding=1)
        self.conv_2_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_3_1 = nn.Conv2d(self.input_channel, 32, 3, padding=1)
        self.conv_4 = nn.Conv2d(128 + 64 + 32, 1, 1)

    def forward(self, x):
        x = autograd.Variable(x, requires_grad=False)
        x = x.transpose(3,1)
        conv_1_output = self.conv_1_1(x)
        conv_1_output = nn.BatchNorm2d(32)(conv_1_output)
        conv_1_output = F.relu(conv_1_output)
        conv_1_output = self.conv_1_2(conv_1_output)
        conv_1_output = nn.BatchNorm2d(64)(conv_1_output)
        conv_1_output = F.relu(conv_1_output)
        conv_1_output = self.conv_1_3(conv_1_output)
        conv_2_output = self.conv_2_1(x)
        conv_2_output = nn.BatchNorm2d(32)(conv_2_output)
        conv_2_output = F.relu(conv_2_output)
        conv_2_output = self.conv_2_2(conv_2_output)
        conv_3_output = self.conv_3_1(x)
        out = torch.cat([conv_1_output, conv_2_output, conv_3_output], dim=1)
        out = nn.BatchNorm2d(128 + 64 + 32)(out)
        out = F.relu(out)
        out = self.conv_4(out)
        out = nn.BatchNorm2d(1)(out)
        out = torch.sigmoid(out)
        return out


class NucleiDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return {'x': torch.Tensor(self.x[item]), 'y': torch.Tensor(self.y[item])}

class MultiCrossEntrophyLoss(nn.Module):
    def __init__(self):
        super(MultiCrossEntrophyLoss, self).__init__()

    def forward(self, predict, label):
        result = label*torch.log(predict) + (1-label)*torch.log(1-predict)
        return - torch.sum(result)/(predict.size(0)*predict.size(1))

