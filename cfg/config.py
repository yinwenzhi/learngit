import warnings

class DefaultConfig(object):
    # 默认参数
    env = 'default' # visdom 环境
    model = 'AlexNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
   
    train_data_root = './data/train/' # 训练集存放路径
    test_data_root = './data/test1' # 测试集存放路径
    # load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载
    load_model_path = False
    batch_size = 1 # batch size
    use_gpu = False # use GPU or not
    num_workers = 0 # how many workers for loading data
    print_freq = 20 # print info every N batch
    
    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
        
    max_epoch = 2
    lr = 0.01 # initial learning rate
    lr_decay = 0.095 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数

# 根据字典更新配置参数
def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 警告还是报错，取决于你个人的喜好
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self, k, v)
        
    # 打印配置信息	
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


#  可配置的参数主要包括：

#     数据集参数（文件路径、batch_size等）
#     训练参数（学习率、训练epoch等）
#     模型参数

# 这样我们在程序中就可以这样使用：
  # import models
  # from config import DefaultConfig
  
  # opt = DefaultConfig()
  # lr = opt.lr
  # model = getattr(models, opt.model)
  # dataset = DogCat(opt.train_data_root)

# 对于字典更新配置 可以这样使用
    # opt = DefaultConfig()
    # new_config = {'lr':0.1,'use_gpu':False}
    # opt.parse(new_config)
    # opt.lr == 0.1

opt =DefaultConfig()
DefaultConfig.parse = parse
