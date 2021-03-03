一、str.replace的用法
```
title = "img name 。end，six  ?"

cx = title.replace(" ", "_").replace("，", "_").replace("。", "_")
print(cx)
```
结果:  
img_name__end_six__?

二、re.sub的用法  
```
df = "my name is h2o2"

cc = re.sub(" ", "_", df)
print(cc)
```
结果：  
my_name_is_h2o2  

*re.sub框架：  
```
re.sub(`pattern`, `repl`, `string`, `count=0`, `flags=0`)

`pattern`, `repl`, `string` 为必选参数
`count`, `flags` 为可选参数
`pattern`正则表达式
`repl`被替换的内容，可以是字符串，也可以是函数
`string`正则表达式匹配的内容
`count`由于正则表达式匹配的结果是多个，使用count来限定替换的个数从左向右，默认值是0，替换所有的匹配到的结果
`flags`是匹配模式，`re.I`忽略大小写，`re.L`表示特殊字符集\w,\W,\b,\B,\s,\S，`re.M`表示多行模式，`re.S` ‘.’包括换行符在内的任意字符，`re.U`表示特殊字符集\w,\W,\b,\B,\d,\D,\s,\D
```