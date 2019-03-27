import numpy as np 
import matplotlib.pyplot as plt

plt.axis([0, 100, 0, 1]) 
plt.ion()
 
xs = [0, 0] 
ys = [1, 1] 

for i in range(100): 
  y = np.random.random() 
  xs[0] = xs[1] 
  ys[0] = ys[1] 
  xs[1] = i 
  ys[1] = y 
  plt.plot(xs, ys) 
  plt.pause(0.1)

# --------------------- 
# 作者：Reacubeth 
# 来源：CSDN 
# 原文：https://blog.csdn.net/xyisv/article/details/80651334 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
# 基本原理是使用一个长度为2的数组，每次替换数据并在原始图像后追加。