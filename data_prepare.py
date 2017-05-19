# -*- coding:utf-8 -*-

import re
import jieba
import collections
from tensorflow.python.platform import gfile

_UNK = "_UNK"

''' 整理训练文件，提取需要的关键词。'''


def get_keys(file_name):
    keys = ['unk']
    with open(file_name, encoding='utf-8') as f:
        data = f.read()
        pattern = re.compile(r"\[([^/]+)\]")
        for item in re.findall(pattern, data):
            if item != '/' and item not in keys:
                keys.append(item)
    return keys

'''生成去掉关键词的纯文本,用于生成词向量'''


def del_keys(file_name, out_file_name):
    with open(file_name, encoding='utf-8') as f:
        data = f.read()
        pattern = re.compile(r'\[(.*?)\]')
        str = re.sub(pattern, '', data)
        with open(out_file_name, 'w') as fw:
            fw.write(str)

''' 生成训练标签对 '''


def genarate_train_data(file_name, word_to_ids, labels):
    train_data = []
    with open(file_name, encoding='utf-8') as f:
        for line in f.readlines():
            pattern = re.compile(r'(.*?)\[(.*?)\](.*?)\[/\]')
            items = re.findall(pattern, line)
            vocab = []
            l = []
            for item in items:
                if item[0] != '':
                    val = jieba.lcut(item[0])
                    vocab.extend([word_to_ids[word] for word in val])
                    l.extend([0] * len(val))
                val = jieba.lcut(item[2])
                vocab.extend([word_to_ids[word] for word in val])
                # jieba.add_word
                temp = labels.index(item[1])
                l.extend([temp] * len(val))
            train_data.append((vocab, l))

    return train_data

''' read data and seg to vocab list '''


def read_data(file_name):
    with open(file_name, encoding='utf-8') as f:
        all_cab = f.read()
    return all_cab

# write vocabularys to file


def write_vocabulary(vocab_file, words):
    with gfile.GFile(vocab_file, mode="wb") as vocab_file:
        for w in words:
            vocab_file.write(bytes(w, 'utf-8') + b"\n")

# build vocabulary file from data file


def create_vocabulary_from_data_file(vocab_file, data_file):
    vocabs = read_data(data_file)
    return create_vocabulary_from_data(vocab_file, vocabs)


def create_vocabulary_from_data(vocab_file, vocabs):
    vocabs = re.sub(
        "[\t\r\n\u3000+\.\!\/_,x$%^*(+\"\']+|[·+——！，。：；》\{\}《？、?~@#￥%……&*（）【】”“]+", "", vocabs)
    cabs = jieba.lcut(vocabs)
    counter = collections.Counter(cabs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    v = [_UNK] + list(words)
    write_vocabulary(vocab_file, v)
    word_to_id = dict(zip(v, range(len(v))))
    return word_to_id, v

    # build vocabulary list from file


def build_vocabulary(vocab_file):
    if gfile.Exists(vocab_file):
        rev_vocab = []
        with gfile.GFile(vocab_file, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found." % vocab_file)

# data file to word ids


def file_to_word_ids(filename, word_to_id):
    data = read_data(filename)
    return [word_to_id[word] for word in data]

# data to word ids


def data_to_word_ids(data, word_to_id):
    ids = []
    for word in data:
        print(word)
        if word in word_to_id:
            ids.append(word_to_id[word])
        else:
            ids.append(word_to_id[_UNK])
    return ids

if __name__ == '__main__':
    keys = get_keys("data/shopping.data")
    word_to_ids, _ = create_vocabulary_from_data_file(
        "data/shopping_vocab.txt", "data/shopping.data")
    train_data = genarate_train_data("data/shopping.data", word_to_ids, keys)
    print(train_data)
