## PYCHARM连接Github
0. 连接GITHUB: File->Settings->Version Control->Git 找到Git，输入Git的路径，也可以点击test看看能否连接成功。
1. 上传文件：工具栏：VCS->Import Into Version Control->Share Project On Github
参照：
[https://www.cnblogs.com/lidyan/p/6538877.html]
2. 暴力链接：
[https://blog.csdn.net/qq_42330205/article/details/111308528?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&dist_request_id=9a2c5677-ef11-45a2-9673-7b2fc8fe209b&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control]

## VSCode连接Github
1 在 Github 上创建远程仓;
2 在 VSCode 上初始化本地仓;
3 在 VSCode 上绑定远程仓;
4 在 VSCode 上拉取远程仓内容;
5 在 VSCode 上新建文件并提交到本地仓;
6 在 VSCode 上将本地仓更改推送到 Github 远程仓;

## VS code markdown预览
目前应该用ctrl + shift + v吧
#### vs code如何查看已经安装的插件
点击插件，在输入窗口输入“@installed”即可

## git知识点
git是世界上最先进的分布式版本控制系统，很多公司原来都是使用 SVN 进行版本控制管理，但越来越多的公司选择将代码迁移至 Git（最具标志性的就是使用 SVN 做版本控制的 Google Code 因为干不过 GitHub，无奈关门大吉-_-)
[https://blog.csdn.net/qq_24531389/article/details/81330054?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&dist_request_id=9699549e-1a4c-429b-b4a1-a3aac4b12509&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control]()

## git add出现问题
“The file will have its original line endings in your working directory”
#### 出现原因：
1. 原因是路径中存在 / 的符号转义问题，false就是不转换符号默认是true，相当于把路径的 / 符号进行转义，这样添加的时候就有问题。  
2. 这是因为文件中换行符的差别导致的。这个提示的意思是说：会把windows格式（CRLF（也就是回车换行））转换成Unix格式（LF），这些是转换文件格式的警告，不影响使用。
git默认支持LF。windows commit代码时git会把CRLF转LF，update代码时LF换CRLF。
#### 解决方法：
1. 
```
 git config --global core.autocrlf false
```
2. 
```
git rm -r --cached .
git config core.autocrlf false
git add .
git commit -m ''
git push
```

## git 提交
[出现这个错误fatal: Unable to create 'project_path/.git/index.lock': File exists.](https://blog.csdn.net/yy1300326388/article/details/44943985)
#### 出现原因：
1. 某个文件夹下出现了.git多余的文件
#### 解决方法：
```
rm -f ./.git/index.lock
```
```
del .git\index.lock
```

## mac 上彻底卸载atom
```
rm -rf ~/.atom
rm /usr/local/bin/atom
rm /usr/local/bin/apm
rm ~/Library/Preferences/com.github.atom.plist
rm ~/Library/Application Support/com.github.atom.ShipIt
rm -rf ~/Library/Application Support/Atom/
```
具体原因：
[https://discuss.atom.io/t/how-to-completely-uninstall-atom-for-mac/9084](https://discuss.atom.io/t/how-to-completely-uninstall-atom-for-mac/9084)

## 在Markdown中粘贴图片
#### 方法一：markdown-preview-enhanced：
插入图片很方便，支持本地图片路径，网上图片链接，也支持上传图片，上传的图片插件已经给你内置上传的服务器了，不用自己整。
参照：
[https://zhidao.baidu.com/question/924667924520417099.html]
Q:Can I paste the image into atom and save it as a file automatically?
[https://github.com/shd101wyy/markdown-preview-enhanced/issues/71]
#### 方法二：markdown-img-paste：
在 atom 设置里的 install 搜索markdown-img-paste,安装后使用快捷键ctrl+shift+v就可以将复制到系统剪切板的图片粘贴到 markdown 。也可以设置将图片上传至服务器，使用如图；
不支持复制 GIF 等动画，有这些 GIF 地址只不过多写几个 markdown 字符。
参照：
[https://atom-china.org/t/atom-markdown/2337]
[https://blog.csdn.net/Vincent_69/article/details/80274981?utm_medium=distribute.pc_relevant.none-task-blog-searchFromBaidu-8.control&dist_request_id=30ed2201-7c01-4c08-9ebb-3e693a30d0c0&depth_1-utm_source=distribute.pc_relevant.none-task-blog-searchFromBaidu-8.control]
#### 方法三：qiniu-uploader + markdown-assistant
参照：
[https://segmentfault.com/a/1190000012291863]
#### 方法四：markdown-image-assistant
将所需图片上传至图床，生成图片外链，然后将图片以Markdown语句的形式插入文中的对应位置。
此时再将Markdown文本复制到平台编辑器时，就可以自动加载已经上传到网络上的图片，不需要再在本地文件和各种软件、网页之间频繁切换了，无疑可以节省大量时间。
[https://yanxiaocn.github.io/2020/04/07/%E5%9C%A8Markdown%E4%B8%AD%E8%87%AA%E5%8A%A8%E6%8F%92%E5%85%A5%E5%9B%BE%E7%89%87%E7%9A%84%E6%96%B9%E6%B3%95/]
实际操作：
#### 失误
![12](D:\0-Notebook\pics\aHR0cHM6Ly9naXQtc2NtLmNvbS9ib29rL2VuL3YyL2ltYWdlcy9hcmVhcy5wbmc.png)
![1](./pics\aHR0cHM6Ly9naXQtc2NtLmNvbS9ib29rL2VuL3YyL2ltYWdlcy9hcmVhcy5wbmc.png)
![pic](.\pics\222.png)
