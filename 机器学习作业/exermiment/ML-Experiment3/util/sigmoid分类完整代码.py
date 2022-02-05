# Package imports
# -*- coding: UTF-8 -*-
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn
from sklearn.metrics import accuracy_score
from torch import nn, optim
import torch.nn.functional as F
import torch
import torch.utils.data as data
from planar_utils import  sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)  # set a seed so that the results are consistent
# 获取当前时间戳
def current_time():
    ctime=time.strftime("_%m_%d_%H_%M_%S_", time.localtime())
    return ctime

# 获取生成文件路径
abs_path=os.path.abspath("./")
output_path=os.path.join(abs_path,"output")
print(output_path)

# 生成数据集
X, Y = load_planar_dataset()
print(X.shape)
print(Y.shape)
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y.flatten(), s=40, cmap=plt.cm.Spectral)

'''
导入数据类型：
X：
type:ndarray
shape:(400,2)
output:
[[x1,x2,x3...],[y1,y2,y3...]] 其中yi为纵坐标

Y:
type:ndarray
shape:(1,400)
output:
[[y1,y2,y3...]] 其中yi为标签，0蓝色，1红色
'''
'''
目标数据类型：
X:
type:ndarry
shape:(400,2)
output:
[[x1,y1],[x2,y2],[x3,y3]...]

Y:
type:ndarray
shape:(400,)为一维行向量
output:
[y1,y2,y3...]
'''
# # 归一化,方法：除最大值法，用于？？
# def data_in_one(inputdata,dim=1):
#     outputdata=inputdata.copy()
#     for i in range(dim):
#         min = np.nanmin(inputdata[i,:])
#         max = np.nanmax(inputdata[i,:])
#         outputdata[i,:] = (inputdata[i,:]-min)/(max-min)
#     return outputdata


# 对X转置，对Y降维
X = X.transpose()
Y = Y.ravel()
print("Dataset shape:")
print(X.shape)
print(Y.shape)
# print(X)
# print(Y)


# 首先要继承torch.utils.data.Dataset类，完成点及标签的读取。
class my_dataset(data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        # pytorh的训练集必须是tensor形式，可以直接在dataset类中转换，省去了定义transform
        # 转换Y数据类型为长整型
        self.point = torch.from_numpy(x).type(torch.FloatTensor)
        self.label = torch.from_numpy(y).type(torch.FloatTensor)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        x = self.point[index]
        label = self.label[index]
        # print(x)
        # print(label)
        return x, label

    def __len__(self):
        return len(self.label)

# 然后定义训练集的transform函数，将数组转换成torch tensor


# 加载训练集，每次读取20个点，并且打乱顺序，防止一次读取的全是红色点或蓝色点
trainset = my_dataset(X, Y)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=20, shuffle=True)
testset = my_dataset(X, Y)
# 加载验证集，一次读入所有的点
testloader = torch.utils.data.DataLoader(testset, batch_size=400, shuffle=True)
# 查看训练集
# for points, label in trainloader:
#     print("points")
#     print(points)
#     print("labels")
#     print(label)

# 搭建简单logistic回归模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2,1)
    
    def forward(self,x):
        x=F.sigmoid(self.fc1(x))
        return x
logisticmodel=LogisticRegression()
criterion = nn.BCELoss()
optimizer = optim.Adam(logisticmodel.parameters(), lr=0.01)
epochs=10
print('开始训练')
for e in range(epochs):
    running_loss = 0
    # 对trainloader中每个batch的所有点进行训练，计算损失函数，反向传播优化权重，将损失求和
    for points, labels in trainloader:
        # 获得模型输出
        out = logisticmodel(points)
        # 误差反向传播, 计算参数更新值
        # print(out)
        # print(labels)
        loss = criterion(out, labels)
        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()
        # 进行反向传播
        loss.backward()
        # 将参数更新值添加到网络的 parameters 上
        optimizer.step()
        running_loss += loss.item()
    if (e+1) % 10 == 0:
        print("epoch"+str(e+1))
# 验证画出验证的分类效果
else:
    # 预测模式
    logisticmodel.eval()
    for points, labels in testloader:
        out = logisticmodel(points)
        # 输出预测结果，>=0.5为class 1（红色），<0.5为class 0（蓝色）。round函数作用为四舍五入
        prediction = torch.round(out)
        pred_y = prediction.data.numpy().squeeze()
        # print(pred_y)
        color = []
        for label in pred_y:
            if label == 1:
                color.append('b')
            else:
                color.append('r')
        target_y = labels.data.numpy()
        plt.title("Output of classification")
        plt.scatter(points.data.numpy()[:, 0], points.data.numpy()[
                    :, 1], c=color, s=40, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/400.  # 预测中有多少和真实值一样
        plt.text(1.7, -4, 'Accuracy=%.2f' %
                accuracy, fontdict={'size': 20, 'color':  'red'})
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        plt.show()
    # 恢复训练模式
    logisticmodel.train()

# 搭建简单的三层全连接神经网络
#     n_x -- the size of the input layer，2
#     n_h -- the size of the hidden layer,4，推荐大小：5。可以尝试改变大小
#     n_y -- the size of the output layer,2，预测标签的种类有2类
class Classifier(nn.Module):
    def __init__(self,input_size=2,hidden_layersize=4):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(2, hidden_layersize)
        self.fc2 = nn.Linear(hidden_layersize, 1)

    # 定义正向传播函数
    def forward(self, x):
        # flatten tensor 为一维
        # 此处x的类型是torch tensor
        # x = x.view(x.shape[0], -1)
        x = F.tanh(self.fc1(x))
        # 使用sigmoid作为激活函数
        x =F.sigmoid(self.fc2(x))
        return x

# 对上面定义的Classifier类进行实例化
model = Classifier(input_size=2,hidden_layersize=4)
print(model)
#损失函数为二分类交叉熵损失函数
criterion = nn.BCELoss()
# 优化方法为Adam梯度下降方法，学习率为0.003，如果因为训练次数太少而达不到分类效果，或者发现loss下降的时候震荡，可以适当调小rl
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 或者用SGD试一下
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 对训练集的全部数据学习5遍，这个数字越大，训练时间越长，推荐次数：10000
# epochs = 10000
epochs = 100
# 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
train_losses = []
print('开始训练')
for e in range(epochs):
    running_loss = 0
    # 对trainloader中每个batch的所有点进行训练，计算损失函数，反向传播优化权重，将损失求和
    for points, labels in trainloader:
        # 获得模型输出
        out = model(points)
        # 误差反向传播, 计算参数更新值
        # print(out)
        # print(labels)
        loss = criterion(out, labels)
        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()
        # 进行反向传播
        loss.backward()
        # 将参数更新值添加到网络的 parameters 上
        optimizer.step()
        running_loss += loss.item()
    if (e+1) % 10 == 0:
        print("epoch"+str(e+1))
#   训练完一个epoch后统计loss作图用
    train_losses.append(running_loss/len(trainloader))
# 验证画出验证的分类效果
else:
    # 预测模式
    model.eval()
    for points, labels in testloader:
        out = model(points)
        # 输出预测结果，>=0.5为class 1（红色），<0.5为class 0（蓝色）。round函数作用为四舍五入
        prediction = torch.round(out)
        pred_y = prediction.data.numpy().squeeze()
        # print(pred_y)
        color = []
        for label in pred_y:
            if label == 1:
                color.append('b')
            else:
                color.append('r')
        target_y = labels.data.numpy()
        plt.title("Output of classification")
        plt.scatter(points.data.numpy()[:, 0], points.data.numpy()[
                    :, 1], c=color, s=40, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/400.  # 预测中有多少和真实值一样
        plt.text(1.7, -4, 'Accuracy=%.2f' %
                accuracy, fontdict={'size': 20, 'color':  'red'})
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        filename="traind_classify"+current_time()+".png"
        opath=os.path.join(output_path,filename)
        plt.savefig(opath)
        plt.show()
    # 恢复训练模式
    model.train()
# 绘制loss下降图（针对不同的学习速率）
plt.title("Train_loss")
plt.plot(train_losses, label='Training loss',)
plt.legend()
filename="Train_loss"+current_time()+".png"
opath=os.path.join(output_path,filename)
plt.savefig(opath)
plt.show()

def predict(pmodel,x):
    X = torch.from_numpy(x).type(torch.FloatTensor)
    ans= torch.round(pmodel(X))
    return ans.detach().numpy()


def plot_decision_boundary(pred_func, X, y,hidden=4):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title("Decision boundary of hidden layer size "+str(hidden))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    filename="Show_boundary"+str(hidden)+current_time()+".png"
    opath=os.path.join(output_path,filename)
    plt.savefig(opath)
    plt.show()
    plt.close()

# 可视化边界
plot_decision_boundary(lambda x: predict(model, x), X, Y,hidden=4)


# 改变隐藏层大小再次进行训练，查看欠拟合和过拟合的情况
def train_model(input_size=2,hidden_layersize=4):
    model = Classifier(input_size,hidden_layersize)
    print(model)
    # 定义损失函数为二分类交叉熵
    criterion = nn.BCELoss()
    # 优化方法为Adam梯度下降方法，学习率为0.003，如果因为训练次数太少而达不到分类效果，或者发现loss下降的时候震荡，可以适当调小rl
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 或者用SGD试一下
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 对训练集的全部数据学习5遍，这个数字越大，训练时间越长，推荐次数：10000
    # epochs = 10000
    epochs = 100
    # 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
    train_losses = []
    print('开始训练')
    for e in range(epochs):
        running_loss = 0
        # 对trainloader中每个batch的所有点进行训练，计算损失函数，反向传播优化权重，将损失求和
        for points, labels in trainloader:
            # 获得模型输出
            out = model(points)
            # 误差反向传播, 计算参数更新值
            loss = criterion(out, labels)
            # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
            optimizer.zero_grad()
            # 进行反向传播
            loss.backward()
            # 将参数更新值添加到网络的 parameters 上
            optimizer.step()
            running_loss += loss.item()
        if (e+1) % 10 == 0:
            print("epoch"+str(e+1))
    #   训练完一个epoch后统计loss作图用
        train_losses.append(running_loss/len(trainloader))
    # 验证画出验证的分类效果
    else:
        # 预测模式
        model.eval()
        for points, labels in testloader:
            out = model(points)
            # 输出预测结果，>=0.5为class 1（红色），<0.5为class 0（蓝色）。round函数作用为四舍五入
            prediction = torch.round(out)
            pred_y = prediction.data.numpy().squeeze()
            # print(pred_y)
            color = []
            for label in pred_y:
                if label == 1:
                    color.append('b')
                else:
                    color.append('r')
            target_y = labels.data.numpy()
            plt.title("Output of classification")
            plt.scatter(points.data.numpy()[:, 0], points.data.numpy()[
                        :, 1], c=color, s=40, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y)/400.  # 预测中有多少和真实值一样
            plt.text(1.7, -4, 'Accuracy=%.2f' %
                    accuracy, fontdict={'size': 20, 'color':  'red'})
            # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
            filename="traind_classify"+current_time()+".png"
            opath=os.path.join(output_path,filename)
            print(opath)
            plt.savefig(opath)
            plt.show()
        # 恢复训练模式
        model.train()
    # 绘制loss下降图（针对不同的学习速率）
    plt.title("Train_loss")
    plt.plot(train_losses, label='Training loss',)
    plt.legend()
    filename="Train_loss"+current_time()+".png"
    opath=os.path.join(output_path,filename)
    plt.savefig(opath)
    plt.show()
    return model


size1_model=train_model(hidden_layersize=1)
# 可视化边界
plot_decision_boundary(lambda x: predict(size1_model, x), X, Y,hidden=1)

# size2_model=train_model(hidden_layersize=2)
# # 可视化边界
# plot_decision_boundary(lambda x: predict(size2_model, x), X, Y,hidden=2)

# size3_model=train_model(hidden_layersize=3)
# # 可视化边界
# plot_decision_boundary(lambda x: predict(size3_model, x), X, Y,hidden=3)

# size4_model=train_model(hidden_layersize=4)
# # 可视化边界
# plot_decision_boundary(lambda x: predict(size4_model, x), X, Y,hidden=4)

# size5_model=train_model(hidden_layersize=5)
# # 可视化边界
# plot_decision_boundary(lambda x: predict(size5_model, x), X, Y,hidden=5)

# size20_model=train_model(hidden_layersize=20)
# # 可视化边界
# plot_decision_boundary(lambda x: predict(size20_model, x), X, Y,hidden=20)

# size50_model=train_model(hidden_layersize=50)
# # 可视化边界
# plot_decision_boundary(lambda x: predict(size50_model, x), X, Y,hidden=50)



