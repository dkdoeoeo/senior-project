import AI
import const

class game():
    def __init__(self):
        self.game_state = {
            "hands": [
                [],  # 玩家 0
                [],  # 玩家 1
                [],  # 玩家 2
                []   # 玩家 3
            ],
            "discards": [
                [],  # 玩家 0
                [],  # 玩家 1
                [],  # 玩家 2
                []  # 玩家 3
            ],
            "opened_hands": [
                [], # 玩家 0
                [], # 玩家 1
                [], # 玩家 2
                [] # 玩家 3
            ],
            "wall_count": 0,
            "current_player": 0,
            "tenpai_info": [None, None, None, None],
            "who_info" : [False,False,False,False],
            "opened_hand_info" : [False,False,False,False],
            "winning_info": None,
            "game_over": False,
            "wall": [],
            "last_discarded_tile" : -1
        }