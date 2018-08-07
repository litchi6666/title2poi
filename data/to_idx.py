import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec


ff = pd.read_csv('../file/video_pre.csv',encoding='utf-8-sig')

# word_idx
words_dict = {}
itetator = 1  # word idx 从1开始，第0个表示0
for line in ff['title']:
    for word in line.split(','):
        if word not in words_dict:
            words_dict[word] = itetator
            itetator += 1

word_idx_pairs = sorted(words_dict.items(),key=lambda x:x[1], reverse=False)
word,idx = zip(*word_idx_pairs)

idx_file = pd.DataFrame({'idx':idx,'word':word})
idx_file.to_csv('../file/word_idx.csv',encoding='utf-8',index=False)
print('word idx save success')
# sentence_idx

poi = list(pd.read_csv('../file/poi.csv')['word'])
title_idx,poi_idx = [],[]
for i in range(len(ff['title'])):
    title = ff['title'][i].split(',')
    t_ = [words_dict[w] for w in title]
    title_idx.append(t_)

    poi_ = ff['poi'][i].split(',')
    p_ = [poi.index(w) for w in poi_]
    poi_idx.append(p_)

    if i > 0 and i % 1000 == 0:
        print(i)

title_idx_file = pd.DataFrame({'title': title_idx,'poi':poi_idx})
title_idx_file.to_csv('../file/title_idx.csv',encoding='utf-8',index=False)
print('title idx save success')


# embedding
w2c_model = 'D:\\data\\wiki_embedding\\model\\wiki.zh.model'
model =Word2Vec.load(w2c_model)
print('w3c load success')

word_size = len(word)
vector_np = np.zeros(shape=[word_size+1,200])
for i in range(len(word)):
    w = word[i]
    if w in model:
        vec = model[w]
    else:
        vec = 5*np.random.random_sample(200,)-2.5  # 随机生成一个向量范围在【-2.5,2,5】 200维
        print(w)
    vector_np[i+1] = vec
np.save('../file/word_embedding',vector_np)
print('embeddinng sava success, size %s' % word_size)


