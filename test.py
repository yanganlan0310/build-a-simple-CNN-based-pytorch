import torch
import cv2
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class myModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

class myModule(nn.Module):
    # 定义卷积神经网络结构
    def __init__(self):
        "__init__函数对神经网络初始化"
        super(myModule,self).__init__()
        self.flatten = nn.Flatten()
        # 搭建全连接层,在容器中构建
        self.fc_stack = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(256,128),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(128,64),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(64,10),          
        )
    def forward(self,x):
        """
        在forward方法中进行数据操作
        parameter:
        x: 输入的数据
        """
        # 先进行数据展平操作
        x = self.flatten(x)
        logits = self.fc_stack(x)
        softmax = nn.Softmax(dim=1)
        x = softmax(logits)
        return x
# 加载测试集
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
# 构建测试集数据载入器 
test_loader = torch.utils.data.DataLoader(test_data,batch_size=16,shuffle=False)

model = myModule2() # 实例化一个神经网络模型,需要和权重文件的网络模型一样
# model = nn.DataParallel(model)
model.load_state_dict(torch.load("FashionMnist_weight.pth"))    # 加载预训练模型
model.eval()    # 预测结果前必须要做的步骤，其作用为将模型转为evaluation模式
# test_loader = torch.tensor(test_loader(),device='cpu')
images,labels = iter(test_loader).next()
img = images[0]
img = img.reshape((28,28)).numpy()
# plt.imshow(img)

# 图片个数转化为tensor
img = torch.from_numpy(img)
img = img.view(1,784)   # 矩阵压扁
# 测试
with torch.no_grad():
    output = model.forward(img)
ps = torch.exp(output)
# 返回矩阵每一行最大值和下标,元组类型
top_p, top_class = ps.topk(1, dim=1)
labellist = ['T恤','裤子','套衫','裙子','外套','凉鞋','汗衫','运动鞋','包包','靴子']
prediction = labellist[top_class]
probability = float(top_p)
print(f'神经网络猜测图片里是 {prediction}，概率为{probability*100}%')