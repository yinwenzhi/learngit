import torch


def himmelblau(x):
  return (x[0] ** 2 + x[1] -11 )**2 +(x[0] +x[1] **2 -7 )**2

# 1 构造tensor实例 用关键字requires_grad = True告诉pytorch那个张量是求梯度时需要考虑的自变量。同时设置自变量的初始值
x = torch.tensor([0. , 0. ],requires_grad = True)
# 2 选择优化器 告诉优化器哪个是要优化的决策变量
optimimizer = torch.optim.Adam([x,])
print('[x,]is: ',[x,])
print('x is: ',x)
for step in range(20001):
  
  if step:
    optimimizer.zero_grad()# 自变量的偏导归零
    f.backward() # 调用后自变量的成员数据 grad就存储了求得的偏导的数值
    optimimizer.step() #调用step（）方法改变自变量的值
  f = himmelblau(x)  #更新函数值
  # print('step ',step,': ',  f )
  if step % 1000 == 0:
    print('step{}:x = {} ,f(x) = {} '.format(step,x.tolist(),f ))