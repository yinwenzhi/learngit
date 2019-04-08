import torch as t
import time

#BasicModule  是对NN.Module的简易封装 提供快速加载和保存模型的接口
#实际使用时，直接调用model.save()及 model.load(opt.load_path)即可
# 其它自定义模型一般继承BasicModule，然后实现自己的模型。
# 其中AlexNet.py实现了AlexNet，ResNet34实现了ResNet34。在models/__init__py中，
class BasicModule(t.nn.Module):
  '''
  封装了nn.Module ，主要提供save和load两个方法
  '''
  def __init__(self,opt= None):
    super(BasicModule,self).__init__()
    self.model_name = str(type(self)) #模型的默认名字

  def load(self,path):
    '''
    可加载指定路径的模型
    '''
    self.load_state_dict(t.load(path))

  def save(self,name=None):
    '''
    保存模型，默认使用“模型名字+时间”作为文件名   
    如AlexNet_0710_23:57:29.pth
    '''
    if name is None:
      prefix = 'checkpoints\\' + self.model_name + '_'
      name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
    t.save(self.state_dict(),name)
    return name 

class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)