# -*- coding:utf-8 -*-
from flask import Flask, jsonify, request, make_response, Response
import json
import tagger
import re
import os
import data_prepare as dp
import classifier_cnn as cc
import threading
import jieba
from werkzeug.datastructures import Headers
# from functools import wraps

app = Flask(__name__)

engines = {}
thread_pool = []

CLASSIFIER = 'classifier'
MAX_THREAD = 2


# def allow_cross_domain(fun):
#     @wraps(fun)
#     def wrapper_fun(*args, **kwargs):
#         rst = make_response(fun(*args, **kwargs))
#         rst.headers['Access-Control-Allow-Origin'] = '*'
#         rst.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
#         allow_headers = "Referer,Accept,Origin,User-Agent"
#         rst.headers['Access-Control-Allow-Headers'] = allow_headers
#         return rst
#     return wrapper_fun


def data_dir(appid):
    return "data/" + appid + "/"


def data_file(appid, operation):
    return data_dir(appid) + operation + ".data"


def vocab_file(appid, operation):
    return data_dir(appid) + operation + ".vocabs"


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

    jieba.initialize()
    # ce
    base_dir = data_dir(appid)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 创建标签文件，保存分类标签
    classifer_data = {}
    labels = list(train_data.keys())
    print("labels", labels)

    lf = labels_file(appid)
    with open(lf, "w") as f:
        data = train_data.keys()
        result = map(lambda x: x.strip() + "\n", data)
        f.writelines(result)

    # 创建分类后的数据文件
    data_list = []
    for d in train_data.items():
        operation = d[0]
        t_data = d[1]
        df = data_file(appid, operation)
        category = appid + "_" + operation
        print("create tagger for %s" % category)
        data_list.append((df, category))

        with open(df, "w") as f:
            for i in range(10):
                for line in t_data:
                    f.write(line + "\r\n")
                    pattern = re.compile(r'\[.*?\]')
                    line = re.sub(pattern, '', line)
                    classifer_data[line] = operation

    for data in data_list:
        df, category = data
        vf = vocab_file(appid, category)
        dp.create_vocabulary_from_data_file(vf, df)
        step = 30
        t = tagger.Tagger(df, vf, category, data_dir(appid) + category, step)
        t.train()

    vf = vocab_file(appid, CLASSIFIER)
    d = ".".join(list(classifer_data.keys()))
    lex, v = dp.create_vocabulary_from_data(vf, d, False)
    calssifier = cc.Classifier(v, appid, data_dir(appid) + 'classifier', labels)
    calssifier.train(classifer_data)

    return "200 OK"


@app.route('/login', methods=['POST'])
def login():
    print(request.get_data().decode('utf-8'))
    name = request.form['name']
    pwd = request.form['password']

    with open("data/account.data", 'r') as f:
        lines = f.readlines()
        for l in lines:
            data = l.strip().split(' ')
            if data[0] == name and data[1] == pwd:
                appid = {}
                appid['appid'] = data[2]
                return jsonify(appid)
    return 'user name or password error'


@app.route('/query', methods=['GET'])
def query():
    result = {}
    appid = request.args.get('appid')
    print(appid)
    path = "data/" + appid + "/"
    for _, _, files in os.walk(path):
        for file in files:
            (name, suffix) = os.path.splitext(file)
            print(name, suffix)
            if suffix == '.data':
                with open(path + file) as f:
                    l = f.readlines()
                    l = list(set(l))
                    result[name] = l
    return jsonify(result)


@app.route('/train', methods=['POST'])
def traim_thread():
    train(request.get_data())
    return '200 OK'


def predict(json_data, result):
    try:
        appid = json_data['appid']
        sentence = json_data['sentence']
        e = engines[appid]
        vf = vocab_file(appid, CLASSIFIER)
        _, v = dp.build_vocabulary(vf)
        labels = load_labels(appid)
        jieba.load_userdict(vf)
        if CLASSIFIER not in e:
            e[CLASSIFIER] = cc.Classifier(v, appid, data_dir(appid) + 'classifier', labels)

        operation = e[CLASSIFIER].predict(sentence)

        if operation == '':
            return 'operation not gotten'

        print("the operation is %s" % operation)

        category = appid + "_" + operation
        vf = vocab_file(appid, category)
        if operation not in e:
            e[operation] = tagger.Tagger(data_file(appid, operation), vf,
                                         category, data_dir(appid) + category)

        pairs = e[operation].determine(sentence)

    except Exception as e:
        print(e)
        return e

    result['appid'] = appid
    result['operation'] = operation
    result['data'] = pairs
    return result


def predict_thread(cond, args):
    while True:
        cond.acquire()
        print("wait for singal!")
        cond.wait()
        predict(args[0], args[1])
        cond.notify()


@app.route('/predict', methods=['POST'])
def request_predict():
    print("----------------------------------------------------------------")
    print(request.data)
    print("----------------------------------------------------------------")
    data = request.get_data()
    json_data = json.loads(data.decode('utf-8'))
    appid = json_data['appid']

    if appid not in engines:
        engines[appid] = {}
    e = engines[appid]
    print(appid)
    if 'thread' not in e:
        e['thread'] = thread_pool.pop()
        print(len(thread_pool))

    result = {}
    e['thread'][1].acquire()
    e['thread'][2].extend([json_data, result])
    e['thread'][1].notify()
    print("wait for result!")
    e['thread'][1].wait()
    j = jsonify(result)
    e['thread'][2].clear()

    print("run success!")
    return j


class MyResponse(Response):
    def __init__(self, response=None, **kwargs):
        kwargs['headers'] = ''
        headers = kwargs.get('headers')
        # 跨域控制
        origin = ('Access-Control-Allow-Origin', '*')
        allow_headers = ('Access-Control-Allow-Headers',
                         'Referer, Accept, Origin, User-Agent, Content-Type')
        methods = ('Access-Control-Allow-Methods', 'HEAD, OPTIONS, GET, POST, DELETE, PUT')
        if headers:
            headers.add(*origin)
            headers.add(*methods)
            headers.add(*allow_headers)
        else:
            headers = Headers([origin, methods, allow_headers])
        kwargs['headers'] = headers
        return super().__init__(response, **kwargs)

if __name__ == "__main__":
    for i in range(MAX_THREAD):
        args = []
        cond = threading.Condition()
        t = threading.Thread(target=predict_thread, args=[cond, args])
        t.start()
        print(cond)
        thread_pool.append((t, cond, args))
    app.response_class = MyResponse
    app.run(port=8001, host='0.0.0.0')
