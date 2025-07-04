from MahjongHelper import MahjongHelper
import torch
import const
import torch.nn.functional as F
from mahjong.hand_calculating.hand_config import HandConfig
from my_struct.action import Action
import numpy as np
import os
import torch.nn as nn
from model_define import DiscardCNN
from model_define import MyCNN
from model_define import MyGRU
import my_struct

class MahjongAI():
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mahjongHelper = MahjongHelper()
        
        self.discard_model_file = 'E:/專題/pygame_vision/models/discard_model.pth'
        self.discard_model=torch.load(self.discard_model_file, weights_only=False).to(device)
        self.discard_model.eval()

        self.pung_model_file = 'E:/專題/pygame_vision/models/pung_model.pth'
        self.pung_model=torch.load(self.pung_model_file, weights_only=False).to(device)
        self.pung_model.eval()

        self.chow_model_file = 'E:/專題/pygame_vision/models/chow_model.pth'
        self.chow_model=torch.load(self.chow_model_file, weights_only=False).to(device)
        self.chow_model.eval()

        self.kong_model_file = 'E:/專題/pygame_vision/models/kong_model.pth'
        self.kong_model=torch.load(self.kong_model_file, weights_only=False).to(device)
        self.kong_model.eval()

        self.riichi_model_file = 'E:/專題/pygame_vision/models/riichi_model.pth'
        self.riichi_model=torch.load(self.riichi_model_file, weights_only=False).to(device)
        self.riichi_model.eval()

        self.predictor_model_file = 'E:/專題/pygame_vision/models/predictor_model.pth'
        self.predictor_model=torch.load(self.predictor_model_file, weights_only=False).to(device)
        self.predictor_model.eval()

        self.last_state = None

    def process_draw(self,game_state: my_struct.Game_state,draw_tile:int):#處理自己摸牌
        model_input = self.mahjongHelper.process_model_input(game_state)
        model_input = model_input.to("cuda")

        config = HandConfig(
            is_tsumo=True,
            is_riichi = (game_state.riichi_info[game_state.current_player] == 1),
            round_wind = game_state.dealer,
            player_wind=game_state.player_wind[game_state.current_player] == 1
        )

        #if self.mahjongHelper.can_long(game_state.players[game_state.current_player].hand,draw_tile,config):
            #return Action(const.SELF_LONG,draw_tile)
        
        if self.mahjongHelper.can_kong(game_state.players[game_state.current_player].hand,game_state.players[game_state.current_player].meld,game_state.last_discarded_tile,True):
            kong_point = self.kong_model(model_input)
            #print("kong分數:",kong_point)

            if kong_point >= const.KONG_THRESHOLD:
                return self.mahjongHelper.best_kong_choice(game_state.players[game_state.current_player].hand,
                                                                 game_state.players[game_state.current_player].meld,
                                                                 game_state.last_discarded_tile,
                                                                 True)

        if self.mahjongHelper.can_riichi(game_state.players[game_state.current_player].hand,game_state.players[game_state.current_player].meld):
            riichi_point = self.riichi_model(model_input)
            #print("riichi_point分數:",riichi_point)

            if riichi_point >= const.RIICHI_THRESHOLD:
                return Action(const.RIICHI)

        discard_output = self.discard_model(model_input).float()
        discard_probabilities = F.softmax(discard_output, dim=0)
        hand_probs = {tile: discard_probabilities[tile].item() for tile in game_state.players[game_state.current_player].hand}
        discard = max(hand_probs, key=hand_probs.get)

        return Action(const.DISCARD,discard)

    def process_discard(self,game_state:my_struct.Game_state,current_player = -1):#處理別人打牌
        if current_player == -1:
            current_player = game_state.current_player

        model_input = self.mahjongHelper.process_model_input(game_state)
        model_input = model_input.to("cuda")
        config = HandConfig(
            is_tsumo=False,
            is_riichi = (game_state.riichi_info[current_player] == 1),
            round_wind = game_state.dealer,
            player_wind=game_state.player_wind[current_player] == 1
        )

        if self.mahjongHelper.can_long(game_state.players[current_player].hand,game_state.last_discarded_tile,config):
            return Action(const.OTHER_LONG)
        
        chow_point = 0
        pong_point = 0
        kong_point = 0

        if (current_player == (game_state.current_player + 1) % 4) and self.mahjongHelper.can_chow(game_state.players[current_player].hand,game_state.last_discarded_tile):
            chow_point = self.chow_model(model_input)
        
        if self.mahjongHelper.can_pong(game_state.players[current_player].hand,game_state.last_discarded_tile):
            pong_point = self.pung_model(model_input)
        
        if self.mahjongHelper.can_kong(game_state.players[current_player].hand,game_state.players[current_player].meld,game_state.last_discarded_tile,False):
            kong_point = self.kong_model(model_input)

        #print("吃碰槓分數:")
        #print("chow_point:",chow_point)
        #print("pong_point:",pong_point)
        #print("kong_point:",kong_point)

        if chow_point >= const.CHOW_THRESHOLD or pong_point >= const.PONG_THRESHOLD or kong_point >= const.KONG_THRESHOLD:
            
            if max(chow_point,pong_point,kong_point) == kong_point:
                return self.mahjongHelper.best_kong_choice(game_state.players[current_player].hand,
                                                                 game_state.players[current_player].meld,
                                                                 game_state.last_discarded_tile,
                                                                 False)
            if max(chow_point,pong_point,kong_point) == pong_point:
                return Action(const.PONG,game_state.last_discarded_tile,[game_state.last_discarded_tile,game_state.last_discarded_tile,game_state.last_discarded_tile])
            if max(chow_point,pong_point,kong_point) == chow_point:
                return self.mahjongHelper.best_chow_choice(game_state.players[current_player].hand,game_state.last_discarded_tile)
        return Action(const.NONE)
    
    def just_discard(self,game_state:my_struct.Game_state):
        model_input = self.mahjongHelper.process_model_input(game_state)
        model_input = model_input.to("cuda")
        discard_output = self.discard_model(model_input).float()
        discard_probabilities = F.softmax(discard_output, dim=0)
        hand_probs = {tile: discard_probabilities[tile].item() for tile in game_state.players[game_state.current_player].hand}
        discard = max(hand_probs, key=hand_probs.get)

        return Action(const.DISCARD,discard)
    
    def get_predictor_score(self,game_state:my_struct.Game_state):
        model_input = self.mahjongHelper.process_predictor_input(game_state)
        model_input = model_input.to("cuda")
        return self.predictor_model(model_input)

    def _reinforce_discard(self, game_state:my_struct.Game_state):
        model_input = self.mahjongHelper.process_model_input(game_state)
        model_input = model_input.to("cuda")

        current_predctor_score = self.predictor_model(model_input).item()
        
