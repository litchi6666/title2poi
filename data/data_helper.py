import numpy as np
import math
import pandas as pd

def load_data_and_labels(file,max_title_length):
    x, y = [], []
    with open(file, 'r', encoding='utf-8-sig') as ff:
        for line in ff.readlines():
            line = eval(line.strip())
            line_x = [0 for _ in range(max_title_length)]
            for i in range(len(line[0])):
                line_x[i] = line[0][i]

            x.append(line_x)
            y.append(line[1])
    return x, y


def load_idx_to_poi(file):
    dict = {}
    ff = pd.read_csv(file,encoding='utf-8-sig')
    for i in range(len(ff['word'])):
        dict[i] = ff['word'][i]
    return dict


def load_embedding(file):
    em = np.load(file)
    em = em.astype(np.float32)
    return em


def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)

    num_batches = int((len(data)-1)/batch_size)+1

    for epoch in range(num_epochs):
        for batch_num in range(num_batches):
            start = batch_num*batch_size
            end = min(start+batch_size, data_size)
            yield (data[start:end],epoch+1,batch_num+1,num_batches)


def idx_to_hot(list_idx, num_class):
    sentence_hot = [0 for _ in range(num_class)]
    for idx in list_idx:
        sentence_hot[idx] = 1
    return sentence_hot


def abc(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return (precision * recall) / (precision + recall)


if __name__ == '__main__':
    print(idx_to_hot([1,2,3,4],10))

    print(abc([([1,2,0,0,0],[4,5,6,7,0])]))

    # x, y = load_data_and_labels('../file/test.txt',25)
    # print(x[:10])
    embedding = load_embedding('../file/data/word_embedding.npy')
    print(embedding.dtype)
    print(embedding[-1])

