# -*- coding:utf-8 -*-
from flask import Flask, jsonify, request
import json
import tagger
import re
import os
import data_prepare as dp
import classifier_cnn as cc
import threading

app = Flask(__name__)

engines = {}

CLASSIFIER = 'classifier'


def data_dir(appid):
    return "data/" + appid + "/"


def data_file(appid, operation):
    return data_dir(appid) + operation + ".data"


def vocab_file(appid):
    return data_dir(appid) + "vocabs.txt"


def labels_file(appid):
    return data_dir(appid) + "labels.txt"


def load_labels(appid):
    lf = labels_file(appid)
    with open(lf, encoding='utf-8') as f:
        data = f.readlines()
        return [x.strip() for x in data]


def train(data):
    json_data = json.loads(data.decode('utf-8'))
    appid = json_data['appid']
    train_data = json_data['data']

    # ce
    vocab_dir = data_dir(appid)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    vf = vocab_file(appid)
    word_to_id, v = dp.create_vocabulary_from_data(
        vf, json.dumps(train_data, ensure_ascii=False))

    if appid not in engines:
        engines[appid] = {}
    e = engines[appid]
    classifer_data = {}
    labels = list(train_data.keys())
    print("labels", labels)

    if CLASSIFIER not in e:
        e[CLASSIFIER] = cc.Classifier(v, appid, data_dir(appid) + 'classifier', labels)
    else:
        e[CLASSIFIER].lex = v
        e[CLASSIFIER].num_classes = len(train_data)
        e[CLASSIFIER].labels = labels

    lf = labels_file(appid)
    with open(lf, "w") as f:
        data = train_data.keys()
        result = map(lambda x: x.strip() + "\n", data)
        f.writelines(result)

    for d in train_data.items():
        operation = d[0]
        t_data = d[1]
        df = data_file(appid, operation)

        with open(df, "w") as f:
            for line in t_data:
                f.write(line + "\r\n")
                pattern = re.compile(r'\[.*?\]')
                line = re.sub(pattern, '', line)
                classifer_data[line] = operation

        if operation not in e:
            print("create tagger for %s" % operation)
            category = appid + "_" + operation
            e[operation] = tagger.Tagger(df, vf, category, data_dir(appid) + category)
        e[operation].train()

    e[CLASSIFIER].train(classifer_data)

    return "200 OK"


@app.route('/train', methods=['POST'])
def tream_thread():
    threading.Thread(target=train, args=[request.get_data()]).start()

    return '200 OK'


@app.route('/predict', methods=['POST'])
def predict():
    print("----------------------------------------------------------------")
    print(request.data)
    print("----------------------------------------------------------------")
    data = request.get_data()
    json_data = json.loads(data.decode('utf-8'))
    appid = json_data['appid']
    setense = json_data['setense']

    if appid not in engines:
        engines[appid] = {}
    e = engines[appid]
    vf = vocab_file(appid)
    _, v = dp.build_vocabulary(vf)
    labels = load_labels(appid)

    if CLASSIFIER not in e:
        e[CLASSIFIER] = cc.Classifier(v, appid, data_dir(appid) + 'classifier', labels)

    operation = e[CLASSIFIER].predict(setense)
    if operation == '':
        return 'operation not gotten'

    print("the operation is %s" % operation)

    if operation not in e:
        category = appid + "_" + operation
        e[operation] = tagger.Tagger(data_file(appid, operation), vf,
                                     category, data_dir(appid) + category)

    pairs = e[operation].determine(setense)
    response = {}
    response['appid'] = appid
    response['operation'] = operation
    response['data'] = pairs
    return jsonify(response)

if __name__ == "__main__":
    app.run(port=8001, host='0.0.0.0')
