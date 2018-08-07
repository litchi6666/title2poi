import tensorflow as tf
import numpy as np


class TextCNN(object):

    def __init__(
            self, seq_length, num_class, embedding_dim,
            filter_sizes, filter_nums, words_embedding, l2_reg_lambed=0.001):

        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_class], name='input_y')
        ### 加载词典 ojbk
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('embedding'):
            # self.embedding = tf.get_variable('embedding', [vacab_size, embedding_dim])
            self.embedding = tf.Variable(tf.constant(words_embedding), name='embedding', trainable=True)
            # self.embedding = tf.Variable(tf.random_uniform([vacab_size, embedding_dim], -1.0, 1.0))
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_inputs_expend = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope('cnn_pooling'):
            pooled_outputs = []
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                filter_shape = [filter_size, embedding_dim, 1, filter_nums]

                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='w')
                b = tf.Variable(tf.constant(0.1, shape=[filter_nums]), name='b')

                conv = tf.nn.conv2d(input=self.embedding_inputs_expend,
                                    filter=w,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(h, ksize=[1, seq_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                pooled_outputs.append(pooled)

        with tf.name_scope('concat'):
            num_filters_totle = filter_nums * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_totle])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        with tf.name_scope('decens'):
            w1 = tf.get_variable('w1', shape=[num_filters_totle, num_class],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[num_class]), name='b1')
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(w1)
            l2_loss += tf.nn.l2_loss(b1)

            self.scores = tf.nn.xw_plus_b(self.h_drop, w1, b1, name='scores')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                             labels=self.input_y)

            self.loss = tf.reduce_mean(losses) + l2_reg_lambed * l2_loss

        with tf.name_scope('result'):
            _, self.pridict_topk = tf.nn.top_k(self.scores,5,name='predicts')
            _, self.input_topK = tf.nn.top_k(self.input_y,5,name='input_top5')

        with tf.name_scope('accuracy'):
            self.predict = tf.argmax(self.scores,1,name='temp_prediciton')
            self.input_max_y = tf.argmax(self.input_y,1,name='temp_input')

            yes_or_no = tf.equal(self.predict,self.input_max_y)

            self.accuracy = tf.reduce_mean(tf.cast(yes_or_no,'float'), name='accuracy')