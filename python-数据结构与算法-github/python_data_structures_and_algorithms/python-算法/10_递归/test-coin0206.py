# def coinChange(self, coins: List[int], amount: int) -> int:
def coinChange(coins, amount):
    # -----------
    # 递归算法：
    # 1. 定义转换的DP函数/数组：
    # res = 0
    def dp(n):
        # 0. Base Case：
        # base case
        if n == 0:
            return 0
        elif n < 0:  # elif n == 1:  这里写错，我草！！！！！！！！！！！！！！！！！！！
            return -1
        res = float("inf")
        # print("1111111111111111111111111")
        # 2. 确定选择,导致状态变换的行为：依次选择不同面额的大小
        for coin in coins:
            # res = min(res, 1 + dp(n - coin))
            subproblem = dp(n - coin)
            if subproblem == -1:
                print("2222222222222")
                continue

            res = min(res, subproblem + 1)
        if res != float("inf"):
            return res
            print(res)
            # print("111111111111111111")
        else:
            return -1
        # return res

    # 返回结果所需的金额大小
    return dp(amount)
    print(dp(amount))

if __name__ == '__main__':
    coinChange([1,2,5], 11)
    print("6554esaffff")