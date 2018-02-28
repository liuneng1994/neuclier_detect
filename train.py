import numpy as np
import torch
from tqdm import tqdm
from torch import optim, nn, autograd
from torch.utils.data.dataloader import DataLoader
from model import NucleiModel, NucleiDataset
from view import load_data
from sklearn.model_selection import train_test_split
from metrics import mean_iou
import matplotlib.pyplot as plt

train, test = load_data()

# preprocessing
X_train = train['x'] / 255
Y_train = train['y'] / 255

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

batch_size = 1
dataset = NucleiDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = NucleiModel((128, 128, 3))


def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1 = inputs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
    score = 1 - score.sum() / num
    return score


epochs = 10
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    for index, data in tqdm(enumerate(dataloader)):
        x = data['x']
        y = autograd.Variable(data['y'], requires_grad=False)
        predict = model.forward(x)
        loss = soft_dice_loss(predict, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_model = model.eval()
    valid_predict = test_model(torch.Tensor(X_test[:32]))
    m_iou = mean_iou(np.asarray(y_test[:32]).astype(np.bool), valid_predict.data.numpy())
    print('epoch %d: mean iou %f' % (epoch, m_iou))
