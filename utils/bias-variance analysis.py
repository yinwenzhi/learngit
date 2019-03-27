import torch
import torch.nn as nn
import torch.optim 
import matplotlib.pyplot as plt

#生成有噪声数据
torch.manual_seed(seed = 0) #固定随机数种子，这样生成的数据是确定的
sample_num = 1000  #生成样本数
features = torch.rand(sample_num,2) * 12 - 6  #生成均匀分布的特征 -6~6
noises = torch.randn(sample_num) #生成标准正态分布的噪声 

def himmelblau(x):
  return (x[:,0] ** 2 + x[:,1] -11 )**2 +(x[:,0] +x[:,1]**2 -7 )**2


hims = himmelblau(features)
print('features: ',features)
print('noises: ',noises)
print('hims: ',hims )

labels = hims + noises #标签数据

#接下来将数据分隔为训练集，验证集，测试集，查看各数据集上的噪声大小
train_num , validate_num , test_num = 600 ,200, 200 # 分割数据
train_mse = (noises[:train_num] ** 2 ).mean() 
validate_mse = (noises[train_num: -test_num] ** 2 ).mean()
test_mse = (noises[-test_num:] ** 2 ).mean()

# print('真实：训练集MSE = {:g}， 验证集 = {:g} ，测试集 = {:g}'.format(train_mse,validate_mse,test_mse))

#确定网络结构并训练网络
#试探着选择某个网咯结构，绘制学习曲线和验证曲线 
# 再通过减小网络或者增大网络，来判断是否出现欠拟合或者过拟合
hidden_features = [6,2] #指定隐含层数 将torch.nn.Sequential类的构造参数放在一个列表里  方便后续修改层数和每层的神经元个数
layers = [nn.Linear(2, hidden_features[0]), ]
for idx, hidden_feature in enumerate(hidden_features):
  layers.append(nn.Sigmoid())
  next_hidden_feature = hidden_features[idx + 1]\
    if idx + 1 < len(hidden_features) else 1 
  layers.append(nn.Linear(hidden_feature,next_hidden_feature))
net = nn.Sequential(*layers ) #前馈神经网络
print('神经网络为{} '.format(net)) 

#计算特定结构的神经网络再某训练数据条目下的训练差错和验证差错
optimizer = torch.optim.Adam(net.parameters())
criterion = nn.MSELoss()

train_entry_num = 600 # 选择训练样本数

n_iter = 100000 # 最大迭代次数
loss_train_list = []
loss_validate_list = []

#设置绘图初始值
xs = [0, 0] 
ys_train = [0, 0] 
ys_validate = [0, 0] 

fig, ax = plt.subplots()
plt.ion()   #将画图模式改为交互模式

for step in range(n_iter):
  outputs = net(features)
  preds = outputs.squeeze()

  loss_train = criterion(preds[:train_entry_num], labels[:train_entry_num])
  loss_validate = criterion(preds[train_num : -test_num], labels[train_num : -test_num])
  
  if step % 500 == 0 :
    #如果需要取值 已知的一种方法是转换为numpy,然后转换为List 
    # loss_train_list.append(loss_train.detach().numpy().tolist())
    # loss_validate_list.append(loss_validate.detach().numpy().tolist())
    # print('#{} 训练集 MSE = {:g} , 验证集 MSE = {:g}'.format(step, loss_train, loss_validate ))
  
    xs[0] = xs[1] 
    ys_train[0] = ys_train[1] 
    ys_validate[0] = ys_validate[1] 
    xs[1] = step

    ys_train[1] = loss_train.detach().numpy().tolist()
    ys_validate[1] = loss_validate.detach().numpy().tolist()
    plt.plot(xs, ys_train,c='red') 
    # plt.plot(xs, ys_validate,c='blue') 

    plt.pause(0.1)

    # y1.append(step)  # 每迭代一次，将i放入y1中画出来
    # ax.cla()   # 清除键
    # ax.bar(y1, label='test', height=y1, width=0.3)
    # ax.legend()
    # plt.pause(0.1)


  optimizer.zero_grad() #梯度值归零
  loss_train.backward() #计算偏导值
  optimizer.step() #更新自变量
plt.ioff()
print('训练集 MSE = {:g} ,验证集 MSE = {:g} '.format(loss_train,loss_validate))

#计算测试集上的差错
#经过以上循坏迭代后 训练好的网络存在类实例net中 调用这个类实例进行预测，就可以得到训练集上的预测值了。
outputs = net(features)
preds = outputs.squeeze()
loss = criterion(preds[-test_num: ], labels[-test_num: ])
print( loss )
# print('loss_train_list:', loss_train_list)
# print('loss_validate_list:', loss_validate_list)

