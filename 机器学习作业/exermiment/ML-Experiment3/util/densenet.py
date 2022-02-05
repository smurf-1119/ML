# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch


## 数据
np.random.seed(1) # set a seed so that the results are consistent

#导入数据
X, Y = load_planar_dataset()
# print(X)
# print(len(X))
# print(len(X[0]))
# print(Y)
# print(len(Y))

# # Visualize the data:pi
# plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral);
# plt.show()

x = X[0]
y = X[1]
labels = np.array(Y.ravel())
points = []

for xi,yi in zip(x,y):
   points.append([xi,yi])

points = np.array(points)

labels = torch.from_numpy(labels).type(torch.LongTensor)
points = torch.from_numpy(points).type(torch.FloatTensor)


print(labels.shape)
print(points.shape)

class PointsDataset(Dataset):
   def __init__(self,points,labels):
      super().__init__()
      self.points = points
      self.labels = labels


   def __getitem__(self, index):
      point = self.points[index]
      label = self.labels[index]
      return point,label

   def __len__(self):
      return len(self.points)

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

points_dataset = PointsDataset(points,labels)
dataloaders = DataLoader(points_dataset, batch_size=1,shuffle=True)
for points, label in dataloader:
    print("points")
    print(points)
    print("labels")
    print(label)


## 网络
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

        # 对上面定义的Classifier类进行实例化
model = Classifier()

print(model)

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adam( model.parameters(), lr = learning_rate)

num_epochs = 1

for epoch in range(num_epochs):
   print('Epoch {}/{}'.format(epoch, num_epochs - 1))
   epoch_loss = 0
   step = 0
   currentPoints = []
   currentOutputs = []
   for point, label in dataloaders:
      step += 1
      optimizer.zero_grad()

      outputs = model(point)

      gclass = torch.argmax(outputs)
      glabel = label[0]
      print(str(gclass) + "----" + str(glabel))
      
      loss = loss_fn(outputs,label)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      currentPoints.append(point)
      currentOutputs.append(gclass)
      # print(str(epoch) + ": train_loss:  " + str(loss.item()))

   print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
# torch.save(model.state_dict(), "1.pth")

print(currentPoints)
print(currentOutputs)

drawPoints = [[],[]]
for point in currentPoints:
   point = point.numpy()[0]
   x = point[0]
   y = point[1]
   drawPoints[0].append(x)
   drawPoints[1].append(y)

print(len(drawPoints[0]))
print(len(drawPoints))
drawPoints = np.array(drawPoints)


drawoutputs = []

for output in currentOutputs:
   output = output.numpy().item()
   drawoutputs.append(output)

drawoutputs = np.array(drawoutputs)
print(drawoutputs)

plt.scatter(drawPoints[0, :], drawPoints[1, :], c=drawoutputs, s=40, cmap=plt.cm.Spectral);
plt.show()
