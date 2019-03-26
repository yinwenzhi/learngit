from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import torch.optim as optim

#defin the network
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    # 1 input image channel, 6 output channels, 5x5 square convolution kernel 
    self.conv1 = nn.Conv2d(1,6,5)
    self.conv2 = nn.Conv2d(6,16,5)
    #an affine operation: y = Wx +b
    self.fc1 = nn.Linear(16*5*5,120)
    self.fc2 = nn.Linear(120,84)
    self.fc3 = nn.Linear(84,10)

  def forward(self,x):
    #Max poling over a (2,2) window
    x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
    # If the size is a square you can only specfy a single number
    x = F.max_pool2d(F.relu(self.conv2(x)),2)
    x = x.view(-1,self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x 

  def num_flat_features(self,x):
    size = x.size()[1:] #all dimensions except the batc dimension
    num_features = 1 
    for s in size:
      num_features *= s
    return num_features

net = Net()
print(net)

# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

input = torch.randn(1,1,32,32)
out = net(input)
print(out)

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1,10))


#Loss Function
#for example
output = net(input)
target = torch.randn(10) # a dummy target, for example
target = target.view(1,-1) # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)

# for illustration let us follow a few steps backward:
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  #ReLU


#to backpropagate the error all we have to do is to loss.backward()
# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients
#Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.
net.zero_grad() #zeros the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print( net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# Update the weights
# learning_rate = 0.01 
# for f in net.parameters():
#   f.data.sub_(g.grad.data* learning_rate)
# but we can use different update rules such as SGD ...

#create your optimizer
optimizer = optim.SGD(net.parameters(),lr = 0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update