import numpy as np

# print(5*np.random.random_sample(200,)-2.5)


# vector_np = np.zeros(shape=[10,4])
# for w in range(10):
#     vec = 5*np.random.random_sample(4,)-2.5  # 随机生成一个向量范围在【-2.5,2,5】 200维
#     vector_np[w]=vec
# print(vector_np)

# abc =[1,2,3,4,5]
# print(' '.join(str(abc)))
# print([0 for _ in range(5)])
#
# import numpy as np
# a = [[1,2,3],[4,5,6],[7,8,9]]
#
# import random
# random.shuffle(a)
# print(a)
#
# a = np.array([[[1,2,3,4],[4,5,6]],
#                 [[1, 2, ], [4, 9, 12]]])
#
# print(list(a))

# -*- coding: utf-8 -*-
import heapq

nums = [1, 8, 2, 23, 7, -4, 18, 23, 24, 37, 2]

# 最大的3个数的索引
max_num_index_list = map(nums.index, heapq.nlargest(3, nums))

# 最小的3个数的索引
min_num_index_list = map(nums.index, heapq.nsmallest(3, nums))

print(list(max_num_index_list))
print(list(min_num_index_list))

