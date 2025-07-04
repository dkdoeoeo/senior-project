from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.shanten import Shanten
from AI import MahjongAI
from model_define import DiscardCNN
from model_define import MyCNN
from model_define import MyGRU
import my_struct
import torch
import const
import torch.nn.functional as F

ai = MahjongAI()
game_state = my_struct.Game_state()
print(ai.get_predictor_score(game_state))

