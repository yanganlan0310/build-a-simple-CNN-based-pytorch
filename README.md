#### READ.ME

##### 0.前言

我们使用FashionMnist 数据集进行训练和分类，它是由 Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自 10 种类别的共 7 万个不同商品的正面图片。Fashion-MNIST数据集包含了10个类别的图像，分别是：t-shirt（T恤），trouser（牛仔裤），pullover（套衫），dress（裙子），coat（外套），sandal（凉鞋），shirt（衬衫），sneaker（运动鞋），bag（包），ankle boot（短靴）。 FashionMNIST 的大小、格式和训练集/测试集划分与原始的 MNIST 完全一致。60000/10000 的训练测试数据划分，28x28 的灰度图片。

使用pytorch搭建神经网络，网络为四层，使用交叉熵损失函数，使用SGD优化器。

##### 1.数据预处理

使用torchvision中的transforms模块对数据进行预处理：

```python
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
```


官方文档： transforms:[Transforming and augmenting images — Torchvision 0.12 documentation (pytorch.org)](https://pytorch.org/vision/stable/transforms.html)

 我们使用FashionMnist 数据集进行训练和分类，它是由 Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自 10 种类别的共 7 万个不同商品的正面图片。Fashion-MNIST数据集包含了10个类别的图像，分别是：t-shirt（T恤），trouser（牛仔裤），pullover（套衫），dress（裙子），coat（外套），sandal（凉鞋），shirt（衬衫），sneaker（运动鞋），bag（包），ankle boot（短靴）。 FashionMNIST 的大小、格式和训练集/测试集划分与原始的 MNIST 完全一致。60000/10000 的训练测试数据划分，28x28 的灰度图片。

##### 2.加载训练集和测试集

```python
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
```

##### 3.查看迭代器中的图片

```python
# 查看一下图片
img,label = next(iter(training_loader))
# img 中有4张图片
# print("img:",img)
# print(f"img.size:{img.size()}")
# print("label:",label)
# print(f"lebel.size:",label.size())
```

##### 4.使用GPU训练

```python
# 如果有GPU 使用GPU训练，否则CPU
device = 'cuda' if torch.cuda.is_available() else "CPU"
# print(f"using: {device}")
```

##### 5.搭建自己的网络

```python
class myModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        """
        在forward方法中进行数据操作
        parameter:
        x: 输入的数据
        """
        x = x.view(x.shape[0], -1)    # make sure input tensor is flattened
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
```

##### 6.定义损失函数

因为是多分类问题，所以我们用交叉熵损失函数：

```python
# 定义损失函数

criterion = nn.CrossEntropyLoss()   # 多分类,使用交叉熵损失函数
```

##### 7.定义优化方法 SGD 随机梯度下降 

```python
# 定义优化方法 SGD 随机梯度下降
optimizer = optim.SGD(my_model.parameters(),lr=0.005,momentum=0.1)
```

##### 8.开始训练

```python
# 训练神经网络
epochs = 20
# 训练误差和测试误差存储在这里,最后进行可视化
training_losses = []
test_losses = []
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
```

##### 9.可视化训练误差和测试误差   

使用 matplotlib.pyplot 可视化训练误差和测试误差

```python
# 可视化训练误差和测试误差      
# 将训练误差和测试误差数据从GPU转回CPU 并且将tensor->numpy (因为numpy 是cup only 的数据类型)
training_losses = np.array(torch.tensor(training_losses,device='cpu'))
test_losses = np.array(torch.tensor(test_losses,device='cpu'))
# 可视化
plt.plot(training_losses,label="training_losses")
plt.plot(test_losses,label="test_losses")
plt.legend()
plt.show()
```

![img](https://img-blog.csdnimg.cn/1e136a4b7b8b4fafb9f449813530e1b8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5piv5a6J5r6c5ZWK,size_20,color_FFFFFF,t_70,g_se,x_16)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

![img](https://img-blog.csdnimg.cn/b586fb8af43b48148bc9ade6961dd49e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5piv5a6J5r6c5ZWK,size_20,color_FFFFFF,t_70,g_se,x_16)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

##### 10.模型的保存

```python
# 模型保存
torch.save(my_model.state_dict(),'FashionMnist_weight.pth')
```

##### 11.测试

```python
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
```

![img](https://img-blog.csdnimg.cn/b262770d4b844fe78c053611653795cb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5piv5a6J5r6c5ZWK,size_20,color_FFFFFF,t_70,g_se,x_16)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

