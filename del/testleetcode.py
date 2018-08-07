def nextPermutation( nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    def swap(a,b):
        temp = a
        a = b
        b = temp
        return a, b

    def move(l, idx):
        temp = l[idx]
        for i in range(idx,-1,-1):
            if i > 0 :
                l[i],l[i-1] = swap(l[i],l[i-1])
        l[0] = temp
        return l

    length = len(nums)
    temp_nums = sorted(nums)

    for i in range(length - 1, -1, -1):

        if i == 1 and nums[1] > nums[0]:
            now_idx = temp_nums.index(nums[0])
            while now_idx < length and temp_nums[now_idx] == temp_nums[now_idx+1]:
                now_idx += 1
            nums = move(temp_nums, now_idx+1)
            break

        if i == 0:
            nums = temp_nums
            break

        if i > 0 and nums[i] > nums[i - 1]:
            nums[i], nums[i-1] = swap(nums[i], nums[i-1])
            idx_ = i

            for j in range(idx_, length):
                if j < length - 1 and nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = swap(nums[j], nums[j + 1])
            break

    return nums

print(nextPermutation([1,2]))