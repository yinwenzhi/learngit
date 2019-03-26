from __future__ import print_function
import torch
import numpy as np 

#Construct a 5x3 matrix, uninitialized:
#x = torch.empty(5,3) 
#Construct a randomly initialized matrix:
x = torch.rand(5,3) 
#Construct a matrix filled zeros and of dtype long:
# x = torch.zeros(5,3,dtype = torch.long ) 

#or create a tensor based on an existing tensor. These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user
# x = x.new_ones(5,3, dtype = torch.double )
# print(x)

# x = torch.randn_like(x, dtype = torch.float ) # override dtype
# print(x)   # result has the same size
# print(x.size()) #torch.Size is in fact a tuple, so it supports all tuple operations.

x = torch.rand(5,3) 
y = torch.rand(5,3)
print(x+y)
print(torch.add(x,y))

# providing an output tensor as argument
result = torch.empty(5,3)
torch.add(x,y, out = result)
print(result)

#adds x to y 
y.add_(x)  #Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
print(y)

#if you want to reize/reshape tensor, you anc use torch,view
x = torch. randn(4,4)
y = x.view(16)
z = x.view(-1, 2) #the size -1 is intferred推断 from other dimensions
print(x.size(),y.size(), z.size())

#If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())


# The Torch Tensor and NumPy array will share their underlying memory locations, and changing one will change the other.
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

# converting Numpy Array to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,2,out = a )
print("a",a)
print("b",b)