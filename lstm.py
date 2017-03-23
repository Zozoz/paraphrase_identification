#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from process_data import load_word_embedding, batch_index, load_train_test, load_sentence
from nn_layer import bi_dynamic_rnn, dynamic_rnn, softmax_layer


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 2000, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.01, 'random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_string('t1', 'last', 'training file')
tf.app.flags.DEFINE_float('keep_prob1', 0.8, 'random base')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'random base')

tf.app.flags.DEFINE_string('train_file_path', 'data/train.txt', 'training file')
tf.app.flags.DEFINE_string('test_file_path', 'data/test.txt', 'test file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/embedding_840b_300d.txt', 'embedding file')


class LSTM(object):

    def __init__(self, embedding_dim=100, batch_size=200, n_hidden=200, learning_rate=0.01, n_class=2,
            max_sentence_len=80, l2_reg=0., display_step=4, n_iter=100, random_base=0.01, t1='last'):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.display_step = display_step
        self.n_iter = n_iter
        self.random_base = random_base
        self.t1 = t1
        self.word_id_mapping, self.w2v = load_word_embedding(FLAGS.embedding_file_path, self.embedding_dim)
        self.word_embedding = tf.constant(self.w2v, name='word_embedding')
        # self.word_embedding = tf.Variable(self.w2v, name='word_embedding')
        # self.word_id_mapping = load_word_id_mapping(FLAGS.word_id_file_path)
        # self.word_embedding = tf.Variable(
        #     tf.random_uniform([len(self.word_id_mapping), self.embedding_dim], -0.1, 0.1), name='word_embedding')

        with tf.name_scope('inputs'):
            self.x1 = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.x2 = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.len1 = tf.placeholder(tf.int32, None)
            self.len2 = tf.placeholder(tf.int32, [None])
            self.y = tf.placeholder(tf.float32, [None, self.n_class])
            self.keep_prob1 = tf.placeholder(tf.float32)
            self.keep_prob2 = tf.placeholder(tf.float32)

    def bi_dynamic_lstm(self, x1, x2, len1, len2):
        x1 = tf.nn.dropout(x1, keep_prob=self.keep_prob1)
        x2 = tf.nn.dropout(x2, keep_prob=self.keep_prob2)

        cell = tf.nn.rnn_cell.LSTMCell
        output1 = bi_dynamic_rnn(cell, x1, self.n_hidden, len1, self.max_sentence_len, 'q1', self.t1)
        output2 = bi_dynamic_rnn(cell, x2, self.n_hidden, len2, self.max_sentence_len, 'q2', self.t1)

        output = tf.concat(1, [output1, output2])  # batch_size * 4n_hidden
        predict = softmax_layer(output, 4 * self.n_hidden, self.random_base, self.keep_prob2, self.l2_reg, self.n_class)
        return predict

    def run(self):
        self.print_para()
        inputs_1 = tf.nn.embedding_lookup(self.word_embedding, self.x1)
        inputs_2 = tf.nn.embedding_lookup(self.word_embedding, self.x2)
        prob = self.bi_dynamic_lstm(inputs_1, inputs_2, self.len1, self.len2)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, self.y)) + sum(reg_loss)
            cost = - tf.reduce_mean(self.y * tf.log(prob)) + sum(reg_loss)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            acc_ = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            true_y = tf.argmax(self.y, 1)
            pred_y = tf.argmax(prob, 1)

        with tf.Session() as sess:
            summary_loss = tf.scalar_summary('loss', cost)
            summary_acc = tf.scalar_summary('acc', acc_)
            train_summary_op = tf.merge_summary([summary_loss, summary_acc])
            test_loss = tf.placeholder(tf.float32)
            test_acc = tf.placeholder(tf.float32)
            summary_loss = tf.scalar_summary('loss', test_loss)
            summary_acc = tf.scalar_summary('acc', test_acc)
            validate_summary_op = tf.merge_summary([summary_loss, summary_acc])
            test_summary_op = tf.merge_summary([summary_loss, summary_acc])
            import time
            timestamp = str(int(time.time()))
            _dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
            train_summary_writer = tf.train.SummaryWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.train.SummaryWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.train.SummaryWriter(_dir + '/validate', sess.graph)

            # x1, x2, len1, len2, y, te_x1, te_x2, te_len1, te_len2, te_y = load_train_test(
            #     FLAGS.train_file_path,
            #     self.word_id_mapping,
            #     self.max_sentence_len
            # )

            x1, x2, len1, len2, y = load_sentence(
                FLAGS.train_file_path,
                self.word_id_mapping,
                self.max_sentence_len
            )

            te_x1, te_x2, te_len1, te_len2, te_y  = load_sentence(
                FLAGS.test_file_path,
                self.word_id_mapping,
                self.max_sentence_len
            )

            init = tf.initialize_all_variables()
            sess.run(init)

            max_acc = 0.
            max_ty, max_py = None, None
            for i in xrange(self.n_iter):
                for train, _ in self.get_batch_data(x1, x2, len1, len2, y, self.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                    _, step, summary = sess.run([optimizer, global_step, train_summary_op], feed_dict=train)
                    train_summary_writer.add_summary(summary, step)
                acc, loss, cnt, summary  = 0., 0., 0, None
                ty, py = [], []
                for test, num in self.get_batch_data(te_x1, te_x2, te_len1, te_len2, te_y, 2000, 1.0, 1.0):
                    _loss, _acc, _ty, _py = sess.run([cost, accuracy, true_y, pred_y], feed_dict=test)
                    ty += list(_ty)
                    py += list(_py)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                print 'all samples={}, correct prediction={}'.format(cnt, acc)
                acc = acc /cnt
                loss = loss / cnt
                print 'Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(step, loss, acc)
                summary = sess.run(test_summary_op, feed_dict={test_loss: loss, test_acc: acc})
                test_summary_writer.add_summary(summary, step)
                if acc > max_acc:
                    max_acc = acc
                    max_ty = ty
                    max_py = py
            print 'P:', precision_score(max_ty, max_py, average=None)
            print 'R:', recall_score(max_ty, max_py, average=None)
            print 'F:', f1_score(max_ty, max_py, average=None)

            print 'Optimization Finished! Max acc={}'.format(max_acc)

    def print_para(self):
            print 'Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}, keep_prob1={}, keep_prob2={}, train_file_path={}, test_file_path={}'.format(
                self.learning_rate,
                self.n_iter,
                self.batch_size,
                self.n_hidden,
                self.l2_reg,
                FLAGS.keep_prob1,
                FLAGS.keep_prob2,
                FLAGS.train_file_path,
                FLAGS.test_file_path
            )

    def get_batch_data(self, x1, x2, len1, len2, y, batch_size, keep_prob1, keep_prob2):
        for index in batch_index(len(y), batch_size, 1):
            feed_dict = {
                self.x1: x1[index],
                self.x2: x2[index],
                self.y: y[index],
                self.len1: len1[index],
                self.len2: len2[index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
            }
            yield feed_dict, len(index)


def main(_):
    lstm = LSTM(
        embedding_dim=FLAGS.embedding_dim,
        batch_size=FLAGS.batch_size,
        n_hidden=FLAGS.n_hidden,
        learning_rate=FLAGS.learning_rate,
        n_class=FLAGS.n_class,
        max_sentence_len=FLAGS.max_sentence_len,
        l2_reg=FLAGS.l2_reg,
        display_step=FLAGS.display_step,
        n_iter=FLAGS.n_iter,
        random_base=FLAGS.random_base,
        t1=FLAGS.t1
    )
    lstm.run()


if __name__ == '__main__':
    tf.app.run()
