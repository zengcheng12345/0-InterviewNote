# 定义一个外部函数outer
def outer():
    print('我是外部函数')
    x = 2
    # 定义一个内部函数inner
    def inner():
        print('我是内部函数')
        # 在内部函数如何修改外部函数的局部变量
        nonlocal x # 此时这里的x不再是新增的变量，而是外部的局部变量x
        y = x * 1.5
        print(x)
        print(y)
        x = 100986 # 不是修改外部的局部变量，而是在内部新定义一个变量
        # return y
        print(x)
        return x
    return inner
outer()()

# 执行结果：
# 我是外部函数
# 我是内部函数
# 3
