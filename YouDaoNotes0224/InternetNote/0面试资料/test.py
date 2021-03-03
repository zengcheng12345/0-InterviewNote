def f1():
    print("in  f1..")
    num=111
    def f2():
        nonlocal num
        num=222
        print(num)
    f2()
    print(num)
f1()
