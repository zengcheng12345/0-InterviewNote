def check(self, nums: List[int]) -> bool:
    sorted_nums = sorted(nums)
    # 方法一：拼接：
    # nums2 = nums
    # 2nums = nums + nums2
    2nums = nums * 2
    for i in range(len(nums) - 1):
        j = i + len(nums)
        if 2nums[i:j] == sorted_nums:
        return True
    return False