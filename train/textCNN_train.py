import tensorflow as tf
from data import data_helper
from tensorflow.contrib import learn
import numpy as np
from models import textCNN
import datetime


# datafile
tf.flags.DEFINE_string('data_file','../file/data/train.txt','train or test file')
tf.flags.DEFINE_string('checkpoint','../checkpoint/textcnn/','save-file-dir')
tf.flags.DEFINE_string('embedding_file','../file/data/word_embedding.npy','embedding-file')

tf.flags.DEFINE_integer('embedding_dim',200,'')
tf.flags.DEFINE_string('filter_sizes','2,3,4','')
tf.flags.DEFINE_integer('filter_nums',128,'')
tf.flags.DEFINE_integer('batch_size',64,'')
tf.flags.DEFINE_integer('num_epochs',10,'')

tf.flags.DEFINE_integer('max_title_length',25,'')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
    print('Loading data...')
    x, y = data_helper.load_data_and_labels(FLAGS.data_file,FLAGS.max_title_length)
    x = np.array(x)
    y = np.array(y)

    embedding = data_helper.load_embedding(FLAGS.embedding_file)
    print('Loading success!')
    return x, y,embedding


def train(x, y, embedding):

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = textCNN.TextCNN(seq_length=x.shape[1],
                          num_class=y.shape[1],
                          embedding_dim=FLAGS.embedding_dim,
                          filter_sizes=FLAGS.filter_sizes.split(','),
                          filter_nums=FLAGS.filter_nums,
                          words_embedding=embedding)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(5e-5)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            '''使用tensorboard记录训练效果'''
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar('acc',cnn.accuracy)

            train_summary_op = tf.summary.merge([loss_summary,acc_summary, grad_summaries_merged])
            train_summary_writer = tf.summary.FileWriter(FLAGS.checkpoint, sess.graph)

            '''训练过程'''
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {cnn.input_x:x_batch,
                             cnn.input_y:y_batch,
                             cnn.keep_prob: 0.5}

                _, step, summaries, loss,acc = sess.run(
                    [train_op, global_step,train_summary_op, cnn.loss,cnn.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                train_summary_writer.add_summary(summaries,step)

            batches = data_helper.batch_iter(
                list(zip(x, y)),FLAGS.batch_size,FLAGS.num_epochs)

            for batch,epoch,batch_num,num_batches in batches:
                print("epoch:{}/{},batch:{}/{}".format(epoch,FLAGS.num_epochs,batch_num,num_batches))
                x_batch,y_batch = zip(*batch)
                train_step(x_batch,y_batch)
                # current_step = tf.train.global_step(sess, global_step)

            saver = tf.train.Saver()
            path = saver.save(sess, FLAGS.checkpoint)
            print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x, y, embedding = preprocess()
    train(x, y, embedding)


if __name__ == '__main__':
    tf.app.run()