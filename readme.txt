practice pytorch 201903
这个文件是联系pytorch的目录
	https://blog.csdn.net/wzy_zju/article/details/78509662
	checkpoints/： 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练
	data/：数据相关操作，包括数据预处理、dataset实现等
	models/：模型定义，可以有多个模型，例如上面的AlexNet和ResNet34，一个模型对应一个文件
	utils/：可能用到的工具函数，在本次实验中主要是封装了可视化工具
					其中也有一些其他学习pytorch时的小练习
	config.py：配置文件，所有可配置的变量都集中在此，并提供默认值
	main.py：主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数
	requirements.txt：程序依赖的第三方库
	README.md：提供程序的必要说明

pytorch错误记录

VSCode中pytorch出现’torch’ has no member 'xxx’的错误
	Win10 中使用Anaconda安装完pytorch后
	在VSCode中输入代码时发现，对于很多需要用到的torch里的函数，VSCode均会红线报错。
	‘torch’ has no member ‘xxx’
	经过搜索资料，找到了解决办法。
	在VSCode setting中搜索python.linting.pylintPath
	并且修改值为(你的Anaconda安装路径)\pkgs\pylint-1.8.4-py36_0\Scripts\pylint，修改完后就不会再报错了。
	最终在用户设置中添加pylintpath解决了问题
	"python.linting.pylintPath"下得目录改为"C:\\Users\\ROD\\Anaconda3\\pkgs\\pylint-1.8.4-py36_0\\Scripts\\pylint"
提示no module named torchvision 
	https://blog.csdn.net/Candy_GL/article/details/81234975