from . import player 
from . import action 

class Game_state:
    def __init__(self):
        self.players = [player.Player(),player.Player(),player.Player(),player.Player()]
        self.round = 0
        self.dealer = 0
        self.player_wind = [0,1,2,3]
        self.score = [250,250,250,250]
        self.dora = [0]
        self.uradora = []
        self.open_dora_num =1
        self.current_player = 0
        self.tenpai_info = [None, None, None, None]
        self.who_info = [False,False,False,False]
        self.riichi_info = [False,False,False,False]
        self.winning_info = None
        self.game_over = False
        self.wall = []
        self.player_behavior = action.Action() #紀錄玩家/ai行為初始化為捨牌
        self.last_discarded_tile = -1