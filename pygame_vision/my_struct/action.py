import const

class Action:
    def __init__(self,type = const.NONE,tile34 = -1,sequence34 = []):
        self.type = type
        self.tile34 = tile34
        self.sequence34 = sequence34