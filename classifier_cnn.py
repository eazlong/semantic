#!/usr/bin/evn python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import data_prepare as dp
import random
import jieba
import os
import logging

model = "classifier"


class Classifier(object):
    def __init__(self, lex, appid, ckpt_dir, labels):
        self.lex = lex
        self.appid = appid
        self.labels = labels
        self.num_classes = len(labels)
        self.ckpt_dir = ckpt_dir
        self.inited = False
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.g = g = tf.Graph()

        with g.as_default():
            with tf.variable_scope(self.appid, reuse=None):
                logging.debug(tf.get_default_graph())
                input_size = len(self.lex)
                self.X = X = tf.placeholder(tf.int32, [None, input_size])
                self.Y = Y = tf.placeholder(tf.float32, [None, self.num_classes])
                with g.name_scope("embedding"):
                    embedding_size = 128
                    W = tf.Variable(tf.random_uniform([input_size, embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(W, X)
                    embedded_chars_expanded = tf.expand_dims(embed, -1)

                self.dropout_keep_prob = tf.placeholder(tf.float32)
                self.batch_size = 32
                num_filters = 128
                filter_sizes = [2, 3, 4, 5]
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with g.name_scope("conv-maxpool-%s" % filter_size):
                        #[filter_size, embedding_size] batch size, 1 input channel, num_filters out put channel
                        filter_shape = [filter_size, embedding_size, 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                        conv = tf.nn.conv2d(embedded_chars_expanded, W,
                                            strides=[1, 1, 1, 1], padding="VALID")
                        h = tf.nn.relu(tf.nn.bias_add(conv, b))
                        pooled = tf.nn.max_pool(
                            h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
                        pooled_outputs.append(pooled)

                num_filters_total = num_filters * len(filter_sizes)
                h_pool = tf.concat(3, pooled_outputs)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
                # dropout
                with g.name_scope("dropout"):
                    h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
                # output
                with g.name_scope("output"):
                    W = tf.get_variable("W", shape=[num_filters_total, self.num_classes],
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
                    self.output = tf.nn.xw_plus_b(h_drop, W, b)

                optimizer = tf.train.AdamOptimizer(1e-3, name="adam")
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, Y))
                grads_and_vars = optimizer.compute_gradients(self.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars)

    # 将句子转化为向量
    def __sentense2feature(self, sentense):
        words = jieba.lcut(sentense)
        logging.debug("words %s" % words)
        features = np.zeros(len(self.lex))
        for word in words:
            if word.isdigit():
                word = '_NUM'
            if word in self.lex:
                features[self.lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return features

    def train(self, data):
        with self.g.as_default():
            sess = tf.InteractiveSession()
            logging.debug(tf.get_default_graph())
            tf.initialize_all_variables().run()
            pos = 0
            epoch_size = 20

            while pos < epoch_size:
                batch_x = []
                batch_y = []
                rarray = np.random.randint(0, high=len(data) - 1, size=self.batch_size)
                for i in list(rarray):
                    k = sorted(data)[i]
                    features = self.__sentense2feature(k)
                    label = np.zeros(self.num_classes)
                    label[self.labels.index(data[k])] = 1
                    batch_y.append(list(label))
                    batch_x.append(list(features))

                    _, loss_ = sess.run([self.train_op, self.loss], feed_dict={
                                        self.X: batch_x, self.Y: batch_y, self.dropout_keep_prob: 0.8})
                pos += 1
                logging.debug("epoch size %d, loss %.3f" % (pos, loss_))
            # 训练完成，保存模型
            tf.train.Saver().save(sess, os.path.join(self.ckpt_dir, model + ".ckpt"))
            sess.close()

    def __load(self, sess):
        logging.debug(self.ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("check point file %s not found. please train model first" %
                             self.ckpt_dir)

    def predict(self, sentense, reload=False):
        with self.g.as_default():
            if (not self.inited) or reload:
                sess = self.session = tf.InteractiveSession()
                tf.initialize_all_variables().run()
                self.__load(sess)
                self.inited = True
                logging.debug(self.lex)
            sess = self.session
            features = self.__sentense2feature(sentense)
            label = np.zeros(self.num_classes)
            out = sess.run(self.output, feed_dict={self.X: [list(features)], self.Y: [
                           list(label)], self.dropout_keep_prob: 1.0})
            predictions = tf.argmax(out, 1)
            data = sess.run(predictions)
            return self.labels[data[0]]


if __name__ == "__main__":
    import json

    data = {"我要买1台手机": "order",
            "帮我定2颗核弹": "order",
            "我要买3台手机": "order",
            "我要买4台坦克": "order",
            "我要买5台手机": "order",
            "我要买6台手机": "order",
            "帮我定7台手机": "order",
            "我要买8台手机": "order",
            "我要买9台手机": "order",
            "帮我看看最贵的坦克": "query",
            "我要看销量最好的核弹": "query",
            "帮我看看销量最好的核弹": "query",
            "我要看销量最好的核弹": "query",
            "我要看最贵的的核弹": "query",
            "我要看销量最好的核弹": "query",
            "我要看质量最好的手机": "query"}
    # data = {'帮我订购100台坦克': 'order',
    #         '给我看下最好的航母': 'query',
    #         '帮我看看销量最好的核弹': 'query',
    #         '帮我订购6台BB': 'order',
    #         '帮我订购7只坦克': 'order',
    #         '帮我订购9台坦克': 'order',
    #         '帮我看看最好的电视': 'query',
    #         '我要买10000颗核弹': 'order',
    #         '帮我订购4台导航': 'order',
    #         '给我看下最牛的航母': 'query',
    #         '我要买1辆坦克': 'order',
    #         '我要买8台核弹': 'order',
    #         '帮我订购3台电视': 'order',
    #         '帮我订购2台坦克': 'order',
    #         '帮我看看最好的坦克': 'query',
    #         '我要买1台坦克': 'order',
    #         '我要买5台核弹': 'order',
    #         '我要买1台手机': 'order',
    #         '我要买1000台坦克': 'order',
    #         '给我看下最差的航母': 'query',
    #         '帮我订购10台电视': 'order',
    #         '帮我看看销量最差的核弹': 'query'}
    vocab_file = "data/v.txt"
    d = ".".join(list(data.keys()))
    lex, v = dp.create_vocabulary_from_data(vocab_file, d, True)
    logging.debug(lex, v)

    c = Classifier(v, "aaa", "data/aaa/classifier", ['order', 'query'])
    c.train(data)
    logging.debug(c.predict("帮我看最贵的手机"))
    logging.debug(c.predict("帮我订购5台手机"))
    logging.debug(c.predict("我要买销量最好的核弹"))
    logging.debug(c.predict("帮我看看销量最好的核弹"))

    #c1 = Classifier(v, 2, "bbb", "data/bbb/classifier")
    # c1.train(data)
    # logging.debug(c1.predict("我要看最贵的手机"))
