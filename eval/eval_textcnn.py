from data import data_helper
import tensorflow as tf
import numpy as np
import pandas as pd
import heapq


tf.flags.DEFINE_string('checkpoint_dir','../checkpoint/textcnn','')
tf.flags.DEFINE_string('checkpoint_meta','../checkpoint/textcnn/.meta','')
tf.flags.DEFINE_integer('batch_size',64,'')
tf.flags.DEFINE_string('outfile','../file/predict/textcnn_predict.csv','')
tf.flags.DEFINE_integer('num_epochs',1,'')


FLAGS = tf.flags.FLAGS


def data_pre():
    eval_x, eval_y = data_helper.load_data_and_labels('../file/data/valid.txt',25)
    idx_to_poi = data_helper.load_idx_to_poi('../file/poi.csv')

    return eval_x, eval_y, idx_to_poi


def predict(x, y):
    #checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph('d:/workspace/title2poi/checkpoint/textcnn/dev.meta')
            saver.restore(sess,tf.train.latest_checkpoint('d:/workspace/title2poi/checkpoint/textcnn/'))

            input_x = graph.get_operation_by_name('input_x').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]

            # input_y = graph.get_operation_by_name('input_top5').outputs[0]
            predictions = graph.get_operation_by_name('decens/scores').outputs[0]

            batches = data_helper.batch_iter(
                list(zip(x, y)), FLAGS.batch_size, FLAGS.num_epochs)
            all_prediction_y = []

            for batch,epoch,batch_num,num_batches in batches:
                x,_ = zip(*batch)
                batch_pridiction = sess.run([predictions],feed_dict={input_x:x,dropout_keep_prob:1.0})

                all_ = batch_pridiction[0].tolist()
                for p in all_:
                    max_num_index_list = map(p.index, heapq.nlargest(5, p))
                    all_prediction_y.append(list(max_num_index_list))

                print("{} // {}".format(batch_num,num_batches))

    pd.DataFrame({'predict':all_prediction_y}).to_csv(FLAGS.outfile,index=False,encoding='utf-8')


def main(argv=None):
    eval_x, eval_y, idx_to_poi = data_pre()
    predict(eval_x, eval_y)
    print('done!')


if __name__ == '__main__':
    tf.app.run()