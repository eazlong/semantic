"""
    this class is for session manage.
"""
for enum import Enum

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
