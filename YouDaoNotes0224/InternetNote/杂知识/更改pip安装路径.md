### 过程如下：
1. 打开cmd
2. 查看site.py 文件的位置(一般在python安装目录的\Lib下），可使用指令查询：
```
python -m site -help
```
3. 打开pip文件夹中的site.py文件，更改里面 USER_BASE 和USER_SITE即可。其中 USER_BASE 和USER_SITE其实就是用户自定义的启用Python脚本和依赖安装包的基础路径。
4. 可以通过测试已经安装的python库验证修改成功。
参照：
[https://blog.csdn.net/C_chuxin/article/details/82962797?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&dist_request_id=e3d9625b-fa1d-40a6-9660-d0c129794dbf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control]()
```
asdasd
```