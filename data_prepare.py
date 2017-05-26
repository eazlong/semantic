# -*- coding:utf-8 -*-

import re
import jieba
import collections
from tensorflow.python.platform import gfile

_UNK = "_UNK"
_PAD = "_PAD"
_NUM = "_NUM"

''' 整理训练文件，提取需要的关键词。'''


def get_keys(file_name):
    keys = ['none', 'unk']
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

#


def add_new_vacab(vocab_file, new_vocab):
    with open(vocab_file, 'a+') as f:
        f.write(new_vocab + '\n')
        jieba.add_word(new_vocab)
''' 生成训练标签对 '''


def genarate_train_data(file_name, word_to_ids, labels, vocab_file):
    train_data = []
    with open(file_name, encoding='utf-8') as f:
        a = []
        for line in f.readlines():
            line = line.strip()
            pattern = re.compile(r'(.*?)\[(.*?)\](.*?)\[/\]')
            items = re.findall(pattern, line)
            vocab = []
            l = []
            for item in items:
                if item[0] != '':
                    val = jieba.lcut(item[0])
                    a.extend(val)
                    vocab.extend([word_to_ids[word] for word in val])
                    l.extend([labels.index('none')] * len(val))

                data = item[2]
                if data.isdigit():
                    data = _NUM
                if data not in word_to_ids:
                    print("add %s to words" % data)
                    add_new_vacab(vocab_file, data)
                    word_to_ids[data] = len(word_to_ids)
                a.append(data)
                vocab.append(word_to_ids[data])
                l.append(labels.index(item[1]))

            pos = line.rindex(']')
            if pos != -1 and pos != len(line) - 1:
                val = jieba.lcut(line[pos + 1:])
                a.extend(val)
                vocab.extend([word_to_ids[word] for word in val])
                l.extend([labels.index('none')] * len(val))

            train_data.append((vocab, l))
        print(a, labels)
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


def create_vocabulary_from_data(vocab_file, data, cut=False):
    data = re.sub(
        r"[\t\r\n\u3000+\.\!\/_,x$%^*(+\"\']+|[·+——！，。：；》《？、?~@#￥%……&*（）【】”“]+|(\[.*?\])", "", data)
    cabs = jieba.lcut(data)
    cabs = [cab if not cab.isdigit() else _NUM for cab in cabs]
    counter = collections.Counter(cabs)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if cut:
        for word in count_pairs:
            if float(word[1]) / float(len(cabs)) > 0.05:
                count_pairs.remove(word)
    words, _ = list(zip(*count_pairs))
    v = [_PAD, _UNK] + list(words)
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


def data_to_word_ids(data, word_to_id, numbs=[]):
    ids = []
    for word in data:
        print(word)
        if word.isdigit():
            numbs.extend(word)
            word = _NUM
        if word in word_to_id:
            ids.append(word_to_id[word])
        else:
            ids.append(word_to_id[_UNK])
    return ids

if __name__ == '__main__':
    df = "data/test9/connector.data"
    vf = "data/test9/test9_connector.vocabs"
    keys = get_keys(df)
    word_to_ids, _ = create_vocabulary_from_data_file(vf, df)
    train_data = genarate_train_data(df, word_to_ids, keys, vf)
    print(train_data)
