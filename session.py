"""
    this class is for session manage.
"""
from enum import Enum

State = Enum("State", ("Init", "Start", "Run", "Finish"))


class Session(object):
    def __init__(self, id):
        self.category = ""
        self.state = State.Init
        self.param = {}
        self.id = id

    @property
    def id(self):
        return self.id

    def run(self, args):
        content = args.get('content')

if __name__ == "__main__":
    # sentence = "我要看小区景观"
    # logging.debug(jieba.lcut(sentence))
    # jieba.add_word("小区景观")
    # logging.debug(jieba.lcut(sentence))
    # jieba.del_word("小区景观")
    # logging.debug(jieba.lcut(sentence))
    # l = [(1, 2), (3, 4)]
    # print(l * 2)
    import numpy as np
    print(1 + np.array(False, float))
