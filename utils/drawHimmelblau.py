#%matplotlib inline #//如果是在juppyter下运行这个文件 需要取消这行注释
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm 
from matplotlib.colors import LinearSegmentedColormap
import os

def himmelblau(x):
  return (x[0] ** 2 + x[1] -11 )**2 +(x[0] +x[1] **2 -7 )**2
x = np.arange(-6 ,6, 0.1)
y = np.arange(-6, 6, 0.1)
X,Y = np.meshgrid(x,y)
Z = himmelblau([X,Y])

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60, -30)
ax.set_xlabel('x[0]')
ax.set_ylabel('y[0]')
plt.show()
# os.system("pause")

