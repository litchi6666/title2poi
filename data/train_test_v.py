import numpy as np
import pandas as pd
from data.data_helper import idx_to_hot

num_class = 1054


ff = pd.read_csv('../file/title_idx.csv')
length = len(ff['title'])

all_list = []
for i in range(len(ff['title'])):
    title = eval(ff['title'][i])
    poi = eval(ff['poi'][i])
    poi = idx_to_hot(poi, num_class)

    all_list.append([title,poi])

print('开始打乱')
import random
random.shuffle(all_list)
valid = all_list[:10000]
test = all_list[10000:20000]
train = all_list[10000:]


train_test = all_list[:1000]
with open('../file/train_test.txt','w',encoding='utf-8') as t:
    for i in range(len(train_test)):
        t.write(str(train_test[i]))
        t.write('\n')

# with open('../file/train.txt','w',encoding='utf-8') as t:
#     for i in range(len(train)):
#         t.write(str(train[i]))
#         t.write('\n')
# with open('../file/test.txt','w',encoding='utf-8') as t:
#     for i in range(len(test)):
#         t.write(str(test[i]))
#         t.write('\n')
# with open('../file/valid.txt','w',encoding='utf-8') as t:
#     for i in range(len(valid)):
#         t.write(str(valid[i]))
#         t.write('\n')