一、写入文档  
代码模块如下：
```
# encoding:utf-8

import json

dic = {'a': 1, 'b': 2, 'c': 3}
js = json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':'))
print(js)
```
二、读入数据
```
# 保存
data = [1, 2, 3]
with open('data.json', 'w') as pf:
	json.dump(data, pf)
# 读取
with open('data.json', 'r') as sf:
	data = json.load(sf)
```