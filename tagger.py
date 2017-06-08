#_*_ coding:utf-8 _*_
import data_prepare as dp
import model
import jieba
import json
import tensorflow as tf
import os
import re
import logging

# 标注器。


class Tagger(object):
    def __init__(self, data_file, vocab_file, category, ckpt_path, user_dict, step=1):
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.ckpt_path = ckpt_path
        self.init = False
        self.keys = dp.get_keys(data_file)  # 从数据文件中获取标签
        self.category = category
        self.model = model.Predictor(category, len(self.keys), ckpt_path, step)
        self.user_dict = user_dict

    # 训练
    def train(self, retrain=False):
        # 是否存在checkpoint的保存目录，如果不存在则创建
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        # 考虑到训练量比较小，词汇表更改比较频繁，每次都重新训练
        # 从数据文件生成词汇表
        self.word_to_id, self.vocabs = dp.build_vocabulary(self.vocab_file)
        train_data = dp.genarate_train_data(
            self.data_file, self.word_to_id, self.keys, self.vocab_file, self.user_dict)

        # 考虑到train_data的数据比较少，如果太少训练效果可能会很差，所以这里将数据重复训练
        # 但可能会产生过拟合的问题
        if len(train_data) < 100:
            train_data = train_data * (100 // len(train_data))

        self.model.train(train_data, self.vocabs, retrain)
        self.trained = True

    def determine(self, sentense, reload=False):
        if (not self.init) or reload:
            self.word_to_id, self.vocabs = dp.build_vocabulary(self.vocab_file)
            self.init = True
        sentense = re.sub(
            r"[\t\r\n\u3000+\.\!\/_,x$%^*(+\"\']+|[·+——！，。：；》《？、?~@#￥%……&*（）【】”“]+|(\[.*?\])", "", sentense)

        print(sentense)
        data = jieba.lcut(sentense)
        ids = dp.data_to_word_ids(data, self.word_to_id, self.user_dict)
        logging.debug("%s,%s,%s" % (self.vocab_file, data, ids))
        tag_ids = self.model.predict(ids, reload)
        pairs = {}
        i = 0
        for tag_id in tag_ids:
            if tag_id != 0:
                key = self.keys[tag_id]
                if key in self.user_dict:
                    datas = self.user_dict[key]['dict']
                    if data[i] in datas:
                        if datas[data[i]] != '':
                            pairs[key] = datas[data[i]]
                            i = i + 1
                            continue

                pairs[key] = data[i]
            i = i + 1

        logging.debug("pairs %s" % pairs)
        return pairs

if __name__ == "__main__":
    data_file = "data/shopping.data"
    vocab_file = "data/shopping_vocab.txt"
    category = "shopping"

    word_to_id, v = dp.create_vocabulary_from_data_file(vocab_file, data_file)

    tagger = Tagger(data_file, vocab_file, category)
    tagger.train()
    tagger.determine("帮我采购3台BB")
    tagger.determine("我要买5000台iphone9")

    # tagger1 = Tagger(data_file, vocab_file, "category")
    # tagger1.train()
    # tagger1.determine("帮我采购3台BB")
