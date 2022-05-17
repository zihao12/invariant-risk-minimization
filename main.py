import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms

from colored_mnist import ColoredMNIST
import argparse
import pickle
import pdb
import numpy as np

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(3 * 28 * 28, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 1)

  def forward(self, x):
    x = x.view(-1, 3 * 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x).flatten()
    return logits


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 20, 5, 1)
    self.conv2 = nn.Conv2d(20, 50, 5, 1)
    self.fc1 = nn.Linear(4 * 4 * 50, 500)
    self.fc2 = nn.Linear(500, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4 * 4 * 50)
    x = F.relu(self.fc1(x))
    logits = self.fc2(x).flatten()
    return logits


def test_model(model, device, test_loader, set_name="test set"):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  #pdb.set_trace()
  test_loss /= len(test_loader.dataset)

  print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)

def eval_model(model, device, test_loader, set_name="test set"):
  model.eval()
  acc = np.array([])
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      acc = np.append(acc, pred.eq(target.view_as(pred)).cpu().detach().numpy())

  print(f'acc on {set_name} is {acc.mean()}')
  return acc


def erm_train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device).float()
    optimizer.zero_grad()
    output = model(data)
    loss = F.binary_cross_entropy_with_logits(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def train_and_test_erm(maxiter, out_result_name, out_model_name):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  data_train = ColoredMNIST(root='./data', env='all_train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))

  complete_data_loader = torch.utils.data.DataLoader(data_train,
    batch_size=2000, shuffle=False, **kwargs)

  indices = torch.randperm(len(data_train))[:int(0.7 * len(data_train))] ## add seed??
  print(indices[:5])
  all_train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data_train, indices),
    batch_size=2000, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=2000, shuffle=False, **kwargs)

  model = ConvNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  for epoch in range(1, maxiter + 1): 
    erm_train(model, device, all_train_loader, optimizer, epoch)
    test_model(model, device, all_train_loader, set_name='train set')
    test_model(model, device, test_loader)

  out = {}
  out['acc_train'] = eval_model(model, device, complete_data_loader, set_name="complete train set")
  out['acc_test'] = eval_model(model, device, test_loader)
  out['indices'] = indices.cpu().detach().numpy()
  file = open(out_result_name, 'wb')
  pickle.dump(out, file)
  torch.save(model, out_model_name)
  
  


def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()


def irm_train(model, device, train_loaders, optimizer, epoch):
  model.train()

  train_loaders = [iter(x) for x in train_loaders]

  dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)

  batch_idx = 0
  penalty_multiplier = epoch ** 1.6
  print(f'Using penalty multiplier {penalty_multiplier}')
  while True:
    optimizer.zero_grad()
    error = 0
    penalty = 0
    for loader in train_loaders:
      data, target = next(loader, (None, None))
      if data is None:
        return
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
      penalty += compute_irm_penalty(loss_erm, dummy_w)
      error += loss_erm.mean()
    (error + penalty_multiplier * penalty).backward()
    optimizer.step()
    if batch_idx % 2 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loaders[0]._dataset),
               100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
      print('First 20 logits', output.data.cpu().numpy()[:20])

    batch_idx += 1


def train_and_test_irm(maxiter, out_result_name, out_model_name):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  data_train = ColoredMNIST(root='./data', env='all_train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))
  complete_data_loader = torch.utils.data.DataLoader(data_train,
    batch_size=2000, shuffle=False, **kwargs)

  data_train1 = ColoredMNIST(root='./data', env='train1',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))

  indices1 = torch.randperm(len(data_train1))[:int(0.7 * len(data_train1))] ## add seed??
  train1_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data_train1, indices1),
    batch_size=2000, shuffle=True, **kwargs)

  data_train2 = ColoredMNIST(root='./data', env='train2',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))

  indices2 = torch.randperm(len(data_train2))[:int(0.7 * len(data_train2))] ## add seed??
  train2_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data_train2, indices2),
    batch_size=2000, shuffle=True, **kwargs)


  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=False, **kwargs)

  model = ConvNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(1, maxiter + 1):
    irm_train(model, device, [train1_loader, train2_loader], optimizer, epoch)
    train1_acc = test_model(model, device, train1_loader, set_name='train1 set')
    train2_acc = test_model(model, device, train2_loader, set_name='train2 set')
    test_acc = test_model(model, device, test_loader)
    if train1_acc > 70 and train2_acc > 70 and test_acc > 60:
      print('found acceptable values. stopping training.')
      return

  out = {}
  out['acc_train'] = eval_model(model, device, complete_data_loader, set_name="complete train set")
  out['acc_test'] = eval_model(model, device, test_loader)
  # out['indices1'] = indices1.cpu().detach().numpy()
  # out['indices2'] = indices2.cpu().detach().numpy()
  out['indices'] = np.append(indices1.cpu().detach().numpy(), 20000 + indices2.cpu().detach().numpy()) ## easy to break

  file = open(out_result_name, 'wb')
  pickle.dump(out, file)
  torch.save(model, out_model_name)
  


def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 16))
  columns = 6
  rows = 6
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label = dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label))  # set title
    plt.imshow(img)

  plt.show()  # finally, render the plot


def main(method, maxiter, out_result_name, out_model_name):
  if method == 'irm':
    train_and_test_irm(maxiter, out_result_name, out_model_name)
  if method == 'erm':
    train_and_test_erm(maxiter, out_result_name, out_model_name)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--i', type=int, default= 1, help='model index')
  parser.add_argument('--method', type=str, default='irm',  help='method')
  parser.add_argument('--maxiter', type=int, default=1,  help='method')
  parser.add_argument('--out_result', type=str,  help='out_result')
  parser.add_argument('--out_model', type=str,  help='out_model')
  args = parser.parse_args()

  method = args.method
  maxiter = args.maxiter
  i = args.i
  out_result_name = args.out_result
  out_model_name = args.out_model

  torch.manual_seed(i)
  main(method, maxiter, out_result_name, out_model_name)
