import pandas as pd
import numpy as np
import jieba


def del_repet(sen_list):
    dict_words = {}
    iterator = 0
    for w in sen_list:
        word = w.strip()
        if word not in dict_words and word in poi_l:
                dict_words[word] = iterator
                iterator += 1

    if len(dict_words) == 0:
        return ''
    else:
        sorted_w = sorted(dict_words.items(), key=lambda x: x[1], reverse=True)
        words,_ = list(zip(*sorted_w))
        dict_words.clear()
        return ','.join(words)


def title_clean(title):
    cleaned = []
    words = jieba.cut(title)
    for w in words:
        if w.strip() not in stop_words:
            cleaned.append(w.strip())
    return ','.join(cleaned)


poi_file = pd.read_csv('../file/poi.csv',encoding='utf-8-sig')
poi_l = [word for word in poi_file['word']]

with open('../file/stopwords.txt','r',encoding='utf-8-sig') as sw:
    stop_words = [line.strip() for line in sw.readlines()]


ff = pd.read_csv('../file/videos.csv',encoding='utf-8-sig',header=0)

id_list, cate_list, title_list, poi_list = [], [], [], []

for i in range(len(ff['poi'])):
    if ff['poi'][i] is not np.nan and ff['title'][i] is not np.nan:
        # poi
        words = del_repet(ff['poi'][i].split(','))
        title = title_clean(ff['title'][i])
        if not words == '' and not title == '':
            poi_list.append(words)

            # title

            title_list.append(title)

            # two id
            id_list.append(ff['video_id'][i])
            cate_list.append(ff['category_id'][i])

    if i % 1000 == 0 and i > 0:
        print(i)


new_file = pd.DataFrame({'video_id':id_list,
                         'category_id':cate_list,
                         'title':title_list,
                         'poi':poi_list})
new_file.to_csv('../file/video_pre.csv',encoding='utf-8',index=False,
                columns=['video_id','category_id','title','poi'])