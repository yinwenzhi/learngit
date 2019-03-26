practice pytorch 201903

VSCode中pytorch出现’torch’ has no member 'xxx’的错误
	Win10 中使用Anaconda安装完pytorch后
	在VSCode中输入代码时发现，对于很多需要用到的torch里的函数，VSCode均会报错。
	‘torch’ has no member ‘xxx’
	经过搜索资料，找到了解决办法。
	在VSCode setting中搜索python.linting.pylintPath
	并且修改值为(你的Anaconda安装路径)\pkgs\pylint-1.8.4-py36_0\Scripts\pylint，修改完后就不会再报错了。
	最终在用户设置中添加pylintpath解决了问题
	"python.linting.pylintPath"下得目录改为"C:\\Users\\ROD\\Anaconda3\\pkgs\\pylint-1.8.4-py36_0\\Scripts\\pylint"