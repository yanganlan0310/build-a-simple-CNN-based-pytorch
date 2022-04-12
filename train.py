import torch
import cv2
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 使用Composes将transforms组合在一起，tranforms子类进行数据预处理和数据增强 
data_transforms = {
    'train':transforms.Compose([transforms.ToTensor(),  # 图像转换成pytorch中的张量
                                transforms.RandomHorizontalFlip(p=0.5),  # 图像依概率随机翻转
                                transforms.RandomGrayscale(p=0.2),   # 图像依概率随机灰度化
                                transforms.RandomAffine(5),  # 图像中心保持不变的随机仿射变换
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), # 归一化
                                ]),
    'val':transforms.Compose([transforms.ToTensor(),  # 图像转换成pytorch中的张量
                                transforms.RandomHorizontalFlip(p=0.5),  # 图像依概率随机翻转
                                transforms.RandomGrayscale(p=0.2),   # 图像依概率随机灰度化
                                transforms.RandomAffine(5),  # 图像中心保持不变的随机仿射变换
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)), # 归一化
                                ])
}


# 加载训练集
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,  # 若本地没有数据集，下载
    transform=transforms.ToTensor()
) 
# 构建训练集数据载入器 并提供给定数据集的可迭代对象。
training_loader = torch.utils.data.DataLoader(training_data,batch_size=16,shuffle=True)
# 加载测试集
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
# 构建测试集数据载入器 
test_loader = torch.utils.data.DataLoader(test_data,batch_size=16,shuffle=False)

# 查看一下图片
img,label = next(iter(training_loader))
# img 中有4张图片
# print("img:",img)
# print(f"img.size:{img.size()}")
# print("label:",label)
# print(f"lebel.size:",label.size())

# 如果有GPU 使用GPU训练，否则CPU
device = 'cuda' if torch.cuda.is_available() else "CPU"
# print(f"using: {device}")

# 搭建神经网络
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

class myModule3(nn.Module):
    # 定义卷积神经网络结构
    def __init__(self):
        "__init__函数对神经网络初始化"
        super().__init__()
        self.flatten = nn.Flatten()
        # 搭建全连接层,在容器中构建
        self.fc_stack = nn.Sequential(
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            # 正则化 防止过拟合 Dropout
            # nn.Dropout(),
            nn.Linear(128,64),
            nn.ReLU(),
            # nn.Dropout(),
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
# 实例化模型
my_model = myModule2().to(device) 
# my_model = myModule() 
# my_model = myModule2()
# 定义损失函数
criterion = nn.CrossEntropyLoss()   # 多分类,使用交叉熵损失函数
# 定义优化方法 SGD 随机梯度下降
optimizer = optim.SGD(my_model.parameters(),lr=0.005,momentum=0.1)

# 训练神经网络
epochs = 20
# 训练误差和测试误差存储在这里,最后进行可视化
training_losses = []
test_losses = []
# print("training start...")
# 记录训练时间
e1 = cv2.getTickCount()
for epoch in range(epochs):
    total_loss = 0.0
    # 遍历训练集中所有数据
    for images,labels in training_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()   # 将优化器中的所有求导结果设置位0
        output = my_model(images)   # 神经网络的输出
        loss = criterion(output,labels) # 损失函数
        loss.backward() # 后向传播
        optimizer.step()    # 更新参数
        total_loss += loss.item()   # 损失函数求和
    else:
        # 测试数据集
        test_loss = 0
        accuracy = 0
        # 测试的时候不需要自动求导和反向传播
        with torch.no_grad():
            # 关闭Dorpout
            my_model.eval() # 预测结果前必须要做的步骤，其作用为将模型转为evaluation模式
            # 遍历测试集
            for images,labels in test_loader:
                # 对传入的图片进行正向推断
                # 将数据也送入GPU
                images,labels = images.to(device),labels.to(device)
                output = my_model(images)
                test_loss += criterion(output,labels)
                ps = torch.exp(output)
                top_p,top_class = ps.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # 恢复Dropout
        my_model.train()
        # 将训练误差和测试误差存储在列表中
        training_losses.append(total_loss/len(training_loader))
        test_losses.append(test_loss/len(test_loader))
        
        print("训练集训练次数:{}/{}:".format((epoch+1),epochs),
              "训练误差:{:.3f}".format(total_loss/len(training_loader)),
              "测试误差:{:.3f}".format(test_loss/len(test_loader)),
              "模型分类准确率:{:.3f}".format(accuracy/len(test_loader)))

# 可视化训练误差和测试误差      
# 将训练误差和测试误差数据从GPU转回CPU 并且将tensor->numpy (因为numpy 是cup only 的数据类型)
training_losses = np.array(torch.tensor(training_losses,device='cpu'))
test_losses = np.array(torch.tensor(test_losses,device='cpu'))
# 可视化
plt.plot(training_losses,label="training_losses")
plt.plot(test_losses,label="test_losses")
plt.legend()
plt.show()

# 模型保存
torch.save(my_model.state_dict(),'FashionMnist_weight.pth')


    

    
    
    
        



 

        
        
        
        
        
        
        
        
    

