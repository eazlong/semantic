#_*_ coding:utf-8 _*_
import data_prepare as dp
import model
import jieba
import tensorflow as tf
import os

# 标注器。


class Tagger(object):
    def __init__(self, data_file, vocab_file, category, ckpt_path):
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.ckpt_path = ckpt_path
        self.init = False
        self.keys = dp.get_keys(data_file)  # 从数据文件中获取标签
        self.category = category
        self.model = model.Predictor(category, len(self.keys), ckpt_path)

    # 训练
    def train(self, retrain=False):
        # 是否存在checkpoint的保存目录，如果不存在则创建
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)

        # 考虑到训练量比较小，词汇表更改比较频繁，每次都重新训练
        # 从数据文件生成词汇表
        self.word_to_id, self.vocabs = dp.build_vocabulary(self.vocab_file)
        train_data = dp.genarate_train_data(self.data_file, self.word_to_id, self.keys)

        self.model.train(train_data, self.vocabs, retrain)
        self.trained = True

    def determine(self, sentense):
        if not self.init:
            self.word_to_id, self.vocabs = dp.build_vocabulary(self.vocab_file)
            self.init = True

        data = jieba.lcut(sentense)
        ids = dp.data_to_word_ids(data, self.word_to_id)
        tag_ids = self.model.predict(ids)
        pairs = {}
        i = 0
        for tag_id in tag_ids:
            if tag_id != 0:
                pairs[self.keys[tag_id]] = data[i]
            i = i + 1

        print("pairs", pairs)
        return pairs

if __name__ == "__main__":
    data_file = "data/shopping.data"
    vocab_file = "data/shopping_vocab.txt"
    category = "shopping"

    word_to_id, v = dp.create_vocabulary_from_data_file(vocab_file, data_file)

    tagger = Tagger(data_file, vocab_file, category)
    # tagger.train()
    tagger.determine("帮我采购3台BB")
    tagger.determine("我要买5000台iphone9")

    # tagger1 = Tagger(data_file, vocab_file, "category")
    # tagger1.train()
    # tagger1.determine("帮我采购3台BB")
