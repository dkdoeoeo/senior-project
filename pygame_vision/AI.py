# -*- coding: utf-8 -*-
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
import time
torch.autograd.set_detect_anomaly(True)


class MahjongAI():
    def __init__(self,buffer_capacity=10000, batch_size = 2):
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
        for param in self.predictor_model.parameters():
            param.requires_grad = False

        self.last_state = None
        self.last_predctor_score = 0
        self.last_log_prob = None
        self.last_action = None

        self.optimizer = torch.optim.Adam(self.discard_model.parameters(), lr=1e-4)

        self.pending_transition = None
        self.buffer = my_struct.ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.running_avg_reward = 0.0
        self.train_step = 0

        self.start_time = time.time()
        self.last_saved_time = self.start_time

    def process_draw(self,game_state:my_struct.Game_state,draw_tile:int, RL_flag = False): #處理自己摸牌
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

        if RL_flag:
            return self._reinforce_discard(game_state)
        
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
    
    def just_discard(self,game_state:my_struct.Game_state, RL_flag = False):
        if RL_flag:
            return self._reinforce_discard(game_state)
        
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
        return self.predictor_model(model_input).item()

    #決策與暫存強化學習用資料
    def _reinforce_discard(self, game_state:my_struct.Game_state):

        model_input = self.mahjongHelper.process_model_input(game_state).clone().detach().to("cuda")
        discard_output = self.discard_model(model_input).float()
        discard_probabilities = F.softmax(discard_output, dim=0)

        hand_tiles = game_state.players[game_state.current_player].hand
        hand_tile_indices = torch.tensor(hand_tiles, device="cuda")

        legal_probs = discard_probabilities[hand_tile_indices]
        m = torch.distributions.Categorical(legal_probs)
        sampled_idx = m.sample()
        log_prob = m.log_prob(sampled_idx)
        entropy = m.entropy()

        discard = hand_tile_indices[sampled_idx].item()
        current_predctor_score = self.get_predictor_score(game_state)

        self.pending_transition = my_struct.Transition()
        self.pending_transition.state = model_input.detach()
        self.pending_transition.discard = sampled_idx.item()
        self.pending_transition.current_score = current_predctor_score
        self.pending_transition.legal_indices = hand_tile_indices

        return Action(const.DISCARD,discard)
    
    #更新遊戲後補全 transition 並存入 buffer
    def store_transition(self, next_game_state:my_struct.game_state):
        if self.pending_transition is None:
            return
        next_state = self.mahjongHelper.process_model_input(next_game_state).clone().detach().to("cuda")
        next_score = self.get_predictor_score(next_game_state)

        self.pending_transition.next_state = next_state.detach()
        self.pending_transition.next_score = next_score

        self.buffer.push(self.pending_transition)
        self.pending_transition = None

    def train_from_buffer(self, beta=0.01):
        if len(self.buffer) < self.batch_size:
            return
        
        self.discard_model.train()
        batch = self.buffer.sample(self.batch_size)

        states = torch.stack([t.state for t in batch])
        discards = torch.tensor([t.discard for t in batch], device="cuda")
        rewards = torch.tensor([t.next_score - t.current_score for t in batch], dtype=torch.float32, device="cuda")
        legal_indices_list = [t.legal_indices for t in batch]

        #模型重新 forward
        discard_outputs = self.discard_model(states).float()
        discard_probabilities = F.softmax(discard_outputs, dim=0)

        log_probs = []
        entropies = []

        for i, legal_indices in enumerate(legal_indices_list):
            legal_probs = discard_probabilities[i][legal_indices]
            m = torch.distributions.Categorical(legal_probs)
            action_idx = discards[i]
            log_probs.append(m.log_prob(torch.tensor(action_idx, device="cuda")))
            entropies.append(m.entropy())
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        batch_mean = rewards.mean().item()
        self.running_avg_reward = 0.9 * self.running_avg_reward + 0.1 * batch_mean
        advantages = rewards - self.running_avg_reward

        RL_loss = -(advantages * log_probs).mean() - beta * entropies.mean()

        self.optimizer.zero_grad()
        RL_loss.backward(retain_graph=True)
        self.optimizer.step()

        current_time = time.time()
        if current_time - self.last_saved_time >= 21600:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            model_path = f'E:/專題/discard_model/RL/CNN_discard_model_{timestamp}.pth'
            torch.save(self.discard_model, model_path)
            self.last_saved_time = current_time
            print(f'模型已基於時間間隔存儲，時間: {timestamp}')
        
        self.discard_model.eval()