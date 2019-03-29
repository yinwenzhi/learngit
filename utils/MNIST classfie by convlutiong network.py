import torch.nn
import torch
import torch.utils.data
import torch.optim
import torchvision.datasets
import torchvision.transforms

#构建网络结构 

#第一种形式
class Net(torch.nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.conv0 = torch.nn.Conv2d(1,64,kernel_size = 3, padding= 1)
    self.relu1 = torch.nn.ReLU()
    self.conv2 = torch.nn.Conv2d(64, 128, kernel_size = 3, padding= 1)
    self.relu3 = torch.nn.ReLU()
    self.pool4 = torch.nn.MaxPool2d(stride= 2, kernel_size= 2)
    self.fc5 = torch.nn.Linear(128 * 14 * 14, 1024)
    self.relu6 = torch.nn.ReLU()
    self.drop7 = torch.nn.Dropout(p = 0.5)
    self.fc8 = torch.nn.Linear(1024,10)

  def forward(self,x):
    x = self.conv0(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.relu3(x)
    x = self.pool4(x)
    x = x.view(-1, 128 * 14 * 14)
    x = self.fc5(x)
    x = self.relu6(x)
    x = self.drop7(x)
    x = self.fc8(x)
    return x 
net = Net()
net 

# #第二种形式 把部分层整合在一起
# class Net(torch.nn.Module):

#   def __init__(self):
#     super(Net, self).__init__()
#     self.conv = torch.nn.Sequential(
#       torch.nn.Conv2d(1, 64,kernel_size = 3,padding= 1),
#       torch.nn.ReLU(),
#       torch.nn.Conv2d(64,128 ,kernel_size = 3, padding = 1),
#       torch.nn.ReLU(),
#       torch.nn.MaxPool2d(stride=2, kernel_size = 2)
#     )
#     self.dense = torch.nn.Sequential(
#       torch.nn.Linear(128 * 14 * 14 ,1024 ),
#       torch.nn.ReLU(),
#       torch.nn.Dropout(p = 0.5),
#       torch.nn.Linear(1024,10)
#     )
#   def forward(self,x):
#     x = self.conv(x)
#     x = x.view(-1, 128 * 14 *14)
#     x = self.dense(x)
#     return x

# net = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

#数据读取
train_dataset = torchvision.datasets.MNIST(
  root= '.\\data\\mnist',train= True, 
  transform= torchvision.transforms.ToTensor(),
  download= True)
test_dataset = torchvision.datasets.MNIST(
  root = '.\\data\\mnist',train = False,
  transform= torchvision.transforms.ToTensor(),
  download= True)

batch_size = 100
train_loader = torch.utils.data.DataLoader(
  dataset = train_dataset,
  batch_size = batch_size
)
test_loader = torch.utils.data.DataLoader(
  dataset = test_dataset,
  batch_size = batch_size
)
# print('tranin_dataset',train_dataset)
#训练网络
num_epochs= 5
for epoch in range(num_epochs): #loop over the dataset mutiple times
    for idx ,(input_images, labels) in enumerate(train_loader):
      #get the inputs 
      # input_images, labels = data

      #zero the parameter gradients
      optimizer.zero_grad()

      #forward + backward + optimize
      preds = net(input_images)
      # print("preds",preds)
      # print("labels",labels)

      loss = criterion(preds,labels)
      loss.backward()
      optimizer.step()

      if idx % 100 == 0 : #print every 100 mini-batches
        print('epoch:%d- batch:%5d  loss: %.3f ' %(epoch +1,idx+1,loss.item()))
        running_loss= 0.0

#测试
correct = 0
total = 0
for images , labels in test_loader:
  preds = net(images)
  predicted = torch.argmax(preds,1)
  total += labels.size(0)
  print('labels:',labels)
  print('labels.size(0):',labels.size(0))
  correct += (predicted == labels).sum ().item()

accuracy = correct/total
print('测试数据准确率：{:.1%}'.format(accuracy))