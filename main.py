# ├── checkpoints/
# ├── data/
# │   ├── __init__.py
# │   ├── dataset.py
# │   └── get_data.sh
# ├── models/
# │   ├── __init__.py
# │   ├── AlexNet.py
# │   ├── BasicModule.py
# │   └── ResNet34.py
# └── utils/
# │   ├── __init__.py
# │   └── visualize.py
# ├── config.py
# ├── main.py
# ├── requirements.txt
# ├── README.md

# 使用谷歌的开源工具fire使得程序可以解析命令行参数 pip install fire即可安
# fire 的基础用法
# 如
# import fire
# def add(x, y):
#  return x + y
 
# def mul(**kwargs):
#    a = kwargs['a']
#    b = kwargs['b']
#    return a * b
 
# if __name__ == '__main__':
#  fire.Fire()
# 则可以这样嗲用
# python example.py add 1 2 # 执行add(1, 2)
# python example.py mul --a=1 --b=2 # 执行mul(a=1, b=2),kwargs={'a':1, 'b':2}
# python example.py add --x=1 --y=2 # 执行add(x=1, y=2)

# 更多参考 http://link.zhihu.com/?target=https%3A//github.com/google/python-fire/blob/master/doc/guide.md

# 在主程序main.py中，主要包含四个函数，其中三个需要命令行执行，main.py的代码组织结构如下：
from cfg.config import opt #导入配置信息
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.utilsVisulise import Visualizer
from tqdm import tqdm
from cfg.config import DefaultConfig
    

# 主要结构如下
# def train(**kwargs):
#     '''
#     训练
#     '''
#     pass
    
# def val(model, dataloader):
#     '''
#     计算模型在验证集上的准确率等信息，用以辅助训练
#     '''
#     pass
  
# def test(**kwargs):
#     '''
#     测试（inference）
#     '''
#     pass
  
# def help():
#     '''
#     打印帮助的信息 
#     '''
#     print('help')
  

# 训练的主要步骤如下：

#     定义网络
#     定义数据
#     定义损失函数和优化器
#     计算重要指标
#     开始训练
#         训练网络
#         可视化各种指标
#         计算在验证集上的指标

# 训练函数的代码如下：
def train(**kwargs):   
   
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    vis = Visualizer(opt.env)
    
    # step1: 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    
    # step2: 数据
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,
                        num_workers=opt.num_workers)
    
    # step3: 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                            lr = lr,
                            weight_decay = opt.weight_decay)
        
    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100
    
    # 训练
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()
        
        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
    
            # 训练模型
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            
            print('epoch:numerate={}:{}'.format(epoch,ii))
            # 更新统计指标以及可视化
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)
            # print('confusion_matrix',confusion_matrix)
            # print('score.data',score.data)
            # print('target.data',target.data)
            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss', loss_meter.value()[0])
                
                # 如果需要的话，进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()
    
        model.save()
        print('model saved ')
        # 计算验证集上的指标及可视化
        val_cm,val_accuracy = val(model,val_dataloader)
        vis.plot('val_accuracy',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
            .format(
                        epoch = epoch,
                        loss = loss_meter.value()[0],
                        val_cm = str(val_cm.value()),
                        train_cm=str(confusion_matrix.value()),
                        lr=lr))
            
        # 如果损失不再下降，则降低学习率
        print('loss_meter.value()[0]',loss_meter.value()[0])
        if float(loss_meter.value()[0]) > previous_loss:          
        # if loss_meter.value().numpy().tolist[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        previous_loss = loss_meter.value()[0]
# 验证相对来说比较简单，但要注意需将模型置于验证模式(model.eval())，验证完成后还需要将其置回为训练模式(model.train())，这两句代码会影响BatchNorm和Dropout等层的运行模式。代码如下。
def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    # 把模型设为验证模式
    model.eval()
    
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
            input, label = data
            with t.no_grad():
                val_input = Variable(input)
                val_label = Variable(label.type(t.LongTensor))
            if opt.use_gpu:
                val_input = val_input.cuda()
                val_label = val_label.cuda()
            score = model(val_input)
            # print('*********************data',data)
            # print('*********************score',score)
            # print('*********************score.data',score.data)
            # print('*********************score.data.squeeze()',score.data.squeeze())
            # print('*********************label',label)
            # print('*********************label.type(t.LongTensor)',label.type(t.LongTensor))

            confusion_matrix.add(score.data, label.type(t.LongTensor))
        
    # 把模型恢复为训练模式
    model.train()
    
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) /\
                    (cm_value.sum())
    return confusion_matrix, accuracy

def test(**kwargs):
    opt.parse(kwargs)
    # 模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    
    # 数据
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,\
                                batch_size=opt.batch_size,\
                                shuffle=False,\
                                num_workers=opt.num_workers)
    
    results = []
    for ii,(data,path) in enumerate(test_dataloader):
        with t.no_grad():
            input = t.autograd.Variable(data)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax\
            (score)[:,1].data.tolist()      
        batch_results = [(path_,probability_) \
            for path_,probability_ in zip(path,probability) ]
        results += batch_results
    write_csv(results,opt.result_file)
    return results

def help():
    '''
    打印帮助的信息： python file.py help
    '''
    
    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))
    
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

# 当用户执行python main.py help的时候，会打印如下帮助信息：


if __name__=='__main__':
    import fire
    fire.Fire()


# 根据fire的使用方法，可通过python main.py <function> --args=xx的方式来执行训练或者测试。

# 正如help函数的打印信息所述，可以通过命令行参数指定变量名.下面是三个使用例子，fire会将包含-的命令行参数自动转层下划线_，也会将非数值的值转成字符串。所以--train-data-root=data/train和--train_data_root='data/train'是等价的

# 训练模型
# python main.py train 
#         --train-data-root=data/train/ 
#         --load-model-path='checkpoints/resnet34_16:53:00.pth' 
#         --lr=0.005 
#         --batch-size=32 
#         --model='ResNet34'  
#         --max-epoch = 20
 
# # 测试模型
# python main.py test
#        --test-data-root=data/test1 
#        --load-model-path='checkpoints/resnet34_00:23:05.pth' 
#        --batch-size=128 
#        --model='ResNet34' 
#        --num-workers=12
 
# # 打印帮助信息
# python main.py help