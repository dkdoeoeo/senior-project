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
from discard_model_validator import Discard_model_validator
torch.autograd.set_detect_anomaly(True)


class MahjongAI():
    def __init__(self,buffer_capacity=300, batch_size = 128,discard_model_file_pth = 'E:/專題/discard_model/RL/best_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mahjongHelper = MahjongHelper()
        
        self.discard_model_file = discard_model_file_pth
        self.discard_model=torch.load(self.discard_model_file, weights_only=False).to(self.device)
        self.discard_model.eval()

        self.pung_model_file = 'E:/專題/pygame_vision/models/pung_model.pth'
        self.pung_model=torch.load(self.pung_model_file, weights_only=False).to(self.device)
        self.pung_model.eval()

        self.chow_model_file = 'E:/專題/pygame_vision/models/chow_model.pth'
        self.chow_model=torch.load(self.chow_model_file, weights_only=False).to(self.device)
        self.chow_model.eval()

        self.kong_model_file = 'E:/專題/pygame_vision/models/kong_model.pth'
        self.kong_model=torch.load(self.kong_model_file, weights_only=False).to(self.device)
        self.kong_model.eval()

        self.riichi_model_file = 'E:/專題/pygame_vision/models/riichi_model.pth'
        self.riichi_model=torch.load(self.riichi_model_file, weights_only=False).to(self.device)
        self.riichi_model.eval()

        self.predictor_model_file = 'E:/專題/pygame_vision/models/predictor_model.pth'
        self.predictor_model=torch.load(self.predictor_model_file, weights_only=False).to(self.device)
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
        self.last_print_info_time = self.start_time
        self.time_interval = 7200

        self.best_model_path = 'E:/專題/discard_model/RL/best_model.pth'

    def process_draw(self,game_state:my_struct.Game_state,draw_tile:int, RL_flag = False): #處理自己摸牌
        model_input = self.mahjongHelper.process_model_input(game_state)
        model_input = model_input.to("cuda")

        config = HandConfig(
            is_tsumo=True,
            is_riichi = (game_state.riichi_info[game_state.current_player] == 1),
            round_wind = game_state.dealer,
            player_wind=game_state.player_wind[game_state.current_player] == 1
        )

        if self.mahjongHelper.can_long(game_state.players[game_state.current_player].hand,game_state.players[game_state.current_player].meld):
            print("胡牌了")
            return Action(const.SELF_LONG,draw_tile)
        
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

        #棄牌
        if RL_flag and self.should_Defend(game_state):
            max_opponent_tile_safety = self.calculate_Tile_Safety(game_state)
            discard = max(game_state.players[game_state.current_player].hand, key=lambda t: (max_opponent_tile_safety[t], t))
            return Action(const.DISCARD,discard)


        if RL_flag:
            return self._reinforce_discard(game_state)
        
        model_input = model_input.unsqueeze(0).unsqueeze(0)   # [1, 1, 29, 34]
        model_input = model_input.permute(0, 1, 3, 2)   # [1, 1, 34, 29]
        discard_output = self.discard_model(model_input).float()
        discard_output = discard_output.squeeze(0)

        hand_tiles = game_state.players[game_state.current_player].hand
        hand_tile_indices = torch.tensor(hand_tiles, device="cuda")

        legal_logits  = discard_output[hand_tile_indices]
        legal_probs = F.softmax(legal_logits, dim=0)
        m = torch.distributions.Categorical(legal_probs)
        sampled_idx = m.sample()
        discard = hand_tile_indices[sampled_idx].item()

        #print("discard_output:", discard_output)
        #print("hand_tile_indices:", hand_tile_indices)
        #print("legal_probs:", legal_probs)

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

        if self.mahjongHelper.can_long(game_state.players[game_state.current_player].hand,game_state.players[game_state.current_player].meld):
            print("胡牌了")
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
        #棄牌
        if RL_flag and self.should_Defend(game_state):
            max_opponent_tile_safety = self.calculate_Tile_Safety(game_state)
            discard = max(game_state.players[game_state.current_player].hand, key=lambda t: (max_opponent_tile_safety[t], t))
            return Action(const.DISCARD,discard)
        
        if RL_flag:
            return self._reinforce_discard(game_state)
        
        model_input = self.mahjongHelper.process_model_input(game_state)
        model_input = model_input.to("cuda")
        model_input = model_input.unsqueeze(0).unsqueeze(0)   # [1, 1, 29, 34]
        model_input = model_input.permute(0, 1, 3, 2)   # [1, 1, 34, 29]
        discard_output = self.discard_model(model_input).float()
        discard_output = discard_output.squeeze(0)

        hand_tiles = game_state.players[game_state.current_player].hand
        hand_tile_indices = torch.tensor(hand_tiles, device="cuda")

        legal_logits  = discard_output[hand_tile_indices]
        legal_probs = F.softmax(legal_logits, dim=0)
        m = torch.distributions.Categorical(legal_probs)
        sampled_idx = m.sample()
        discard = hand_tile_indices[sampled_idx].item()

        #print("discard_output:", discard_output)
        #print("hand_tile_indices:", hand_tile_indices)
        #print("legal_probs:", legal_probs)

        return Action(const.DISCARD,discard)
    
    def get_predictor_score(self,game_state:my_struct.Game_state):
        model_input = self.mahjongHelper.process_predictor_input(game_state)
        model_input = model_input.to("cuda")
        return self.predictor_model(model_input).item()

    #決策與暫存強化學習用資料
    def _reinforce_discard(self, game_state:my_struct.Game_state):

        model_input = self.mahjongHelper.process_model_input(game_state).clone().detach().to("cuda")
        model_input = model_input.unsqueeze(0).unsqueeze(0)   # [1, 1, 29, 34]
        model_input = model_input.permute(0, 1, 3, 2)   # [1, 1, 34, 29]
        discard_output = self.discard_model(model_input).float()
        discard_output = discard_output.squeeze(0)

        hand_tiles = game_state.players[game_state.current_player].hand
        hand_tile_indices = torch.tensor(hand_tiles, device="cuda")

        legal_logits  = discard_output[hand_tile_indices]
        legal_probs = F.softmax(legal_logits, dim=0)

        #print("discard_output:", discard_output)
        #print("hand_tile_indices:", hand_tile_indices)
        #print("legal_probs:", legal_probs)
        
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

    def train_from_buffer(self,train_times, ai_win_times):

        if len(self.buffer) < self.buffer.capacity:
            return train_times, False, ''

        beta=0.01 * (0.995 ** train_times)
        self.discard_model.train()
        batch = self.buffer.sample(self.batch_size)

        states = torch.stack([t.state for t in batch]).squeeze(2)
        discards = torch.tensor([t.discard for t in batch], device="cuda")
        rewards = torch.tensor([t.next_score - t.current_score for t in batch], dtype=torch.float32, device="cuda")
        legal_indices_list = [t.legal_indices for t in batch]

        #模型重新 forward
        log_probs = []
        entropies = []

        discard_outputs = self.discard_model(states).float()

        for i, legal_indices in enumerate(legal_indices_list):
            legal_logits  = discard_outputs[i][legal_indices]
            legal_probs = F.softmax(legal_logits, dim=0)
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

        if current_time - self.last_saved_time >= self.time_interval:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            model_path = f'E:/專題/discard_model/RL/CNN_discard_model_{timestamp}.pth'
            torch.save(self.discard_model, model_path)
            self.last_saved_time = current_time
            #print(f'模型已基於時間間隔存儲，時間: {timestamp}')

            discard_model_validator = Discard_model_validator(model_path, best_model_path = self.discard_model_file)
            discard_model_validator.start()

            """if os.path.exists(self.best_model_path):
                self.discard_model=torch.load(self.best_model_path, weights_only=False).to(self.device)
                print(f"[AI] 已切換至最佳模型 {self.best_model_path}")"""
            
            if (ai_win_times[0]/sum(ai_win_times)) >= 0.3:
                self.discard_model.eval()
                return train_times + 1,True,model_path
        
        self.discard_model.eval()
        #print("結束訓練")
        #print("訓練次數:",train_times + 1)
        return train_times + 1,False,''
    
    #棄牌時判斷是否要啟動防守模式
    def should_Defend(self, game_state:my_struct.Game_state):
        current_player = game_state.current_player
        current_player_shanten = self.mahjongHelper.calculate_shanten(game_state.players[current_player].hand)

        if len(game_state.wall) < 25 and current_player_shanten>=3:
            return True
        
        if len(game_state.wall) < 15 and current_player_shanten>=2:
            return True
        
        if len(game_state.wall) < 10 and current_player_shanten>=1:
            return True
        
        #三個對手中，最大的副露數
        maxMeldCount = max(len(game_state.players[(current_player + 1) % 4].meld),len(game_state.players[(current_player + 2) % 4].meld),len(game_state.players[(current_player + 3) % 4].meld))

        if maxMeldCount == 2 and current_player_shanten >= 2:
            return True
        
        if maxMeldCount >= 3 and current_player_shanten >= 1:
            return True
        
        return False
    
    def calculate_Tile_Safety(self, game_state:my_struct.Game_state):
        current_player = game_state.current_player
        opponent_tile_safety_matrix = [[0 for _ in range(34)] for _ in range(3)]

        for player_count in range(3):
            for tile_number in range(34):
                #目標玩家棄牌
                aim_player_discard = game_state.players[(current_player + 1 + player_count) % 4].discards

            #字牌出現數量
                word_tile_count = 0
                if tile_number >= 27:
                    word_tile_count += game_state.players[current_player].hand.count(tile_number)
                    word_tile_count += game_state.players[(current_player + 1) % 4].discards.count(tile_number)
                    word_tile_count += game_state.players[(current_player + 2) % 4].discards.count(tile_number)
                    word_tile_count += game_state.players[(current_player + 3) % 4].discards.count(tile_number)

                    for i in range(4):
                        for meld_count in range(len(game_state.players[(current_player + i) % 4].meld)):
                            word_tile_count += game_state.players[(current_player + i) % 4].meld[meld_count].tiles34.count(tile_number)

            #完全安牌:9

                #該玩家打過
                if tile_number in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 9
                    continue
                #上家剛打過
                elif tile_number == game_state.players[(current_player - 1) % 4].discards[-1]:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 9
                    continue
                #第四張字牌
                elif tile_number >= 27 and word_tile_count == 4:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 9
                    continue

                #現2字牌:8
                if tile_number >= 27 and word_tile_count >= 3:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 8
                    continue

            #筋19、現1字牌:7
                
                #筋一萬
                if tile_number == 0 and 3 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 7
                    continue
                #筋九萬
                if tile_number == 8 and 5 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 7
                    continue
                #筋一筒
                if tile_number == 9 and 12 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 7
                    continue
                #筋九筒
                if tile_number == 17 and 14 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 7
                    continue
                #筋一索
                if tile_number == 18 and 21 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 7
                    continue
                #筋九索
                if tile_number == 26 and 23 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 7
                    continue
                #現1字牌
                if tile_number >= 27 and word_tile_count >= 2:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 7
                    continue

            #筋28、兩筋456:6
                #筋二萬
                if tile_number == 1 and 4 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #筋八萬
                if tile_number == 7 and 4 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #筋二筒
                if tile_number == 10 and 13 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #筋八筒
                if tile_number == 16 and 13 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #筋二索
                if tile_number == 19 and 22 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #筋八索
                if tile_number == 25 and 22 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋四萬
                if tile_number == 3 and 0 in aim_player_discard and 6 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋五萬
                if tile_number == 4 and 1 in aim_player_discard and 7 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋六萬
                if tile_number == 5 and 2 in aim_player_discard and 8 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋四筒
                if tile_number == 12 and 9 in aim_player_discard and 15 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋五筒
                if tile_number == 13 and 10 in aim_player_discard and 16 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋六筒
                if tile_number == 14 and 11 in aim_player_discard and 17 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋四索
                if tile_number == 21 and 18 in aim_player_discard and 24 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋五索
                if tile_number == 22 and 19 in aim_player_discard and 25 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue
                #兩筋六索
                if tile_number == 23 and 20 in aim_player_discard and 26 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 6
                    continue

            #筋37:5
                #筋三萬
                if tile_number == 2 and 5 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 5
                    continue
                #筋七萬
                if tile_number == 6 and 3 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 5
                    continue
                #筋三筒
                if tile_number == 11 and 14 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 5
                    continue
                #筋七筒
                if tile_number == 15 and 12 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 5
                    continue
                #筋三索
                if tile_number == 20 and 23 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 5
                    continue
                #筋七索
                if tile_number == 24 and 21 in aim_player_discard:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 5
                    continue
            
            #生張字牌:4
                if tile_number >= 27 and word_tile_count >= 1:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 4
                    continue

            #無筋19:3
                #無筋一萬
                if tile_number == 0:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 3
                    continue
                #無筋九萬
                if tile_number == 8:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 3
                    continue
                #無筋一筒
                if tile_number == 9:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 3
                    continue
                #無筋九筒
                if tile_number == 17:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 3
                    continue
                #無筋一索
                if tile_number == 18:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 3
                    continue
                #無筋九索
                if tile_number == 26:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 3
                    continue
            
            #無筋2837、半筋456:2
                #無筋二萬
                if tile_number == 1:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋八萬
                if tile_number == 7:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋二筒
                if tile_number == 10:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋八筒
                if tile_number == 16:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋二索
                if tile_number == 19:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋八索
                if tile_number == 25:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋三萬
                if tile_number == 2:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋七萬
                if tile_number == 6:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋三筒
                if tile_number == 11:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋七筒
                if tile_number == 15:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋三索
                if tile_number == 20:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #無筋七索
                if tile_number == 24:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋四萬
                if tile_number == 3 and( 0 in aim_player_discard or 6 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋五萬
                if tile_number == 4 and( 1 in aim_player_discard or 7 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋六萬
                if tile_number == 5 and( 2 in aim_player_discard or 8 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋四筒
                if tile_number == 12 and( 9 in aim_player_discard or 15 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋五筒
                if tile_number == 13 and( 10 in aim_player_discard or 16 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋六筒
                if tile_number == 14 and( 11 in aim_player_discard or 17 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋四索
                if tile_number == 21 and(18 in aim_player_discard or 24 in aim_player_discard) :
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋五索
                if tile_number == 22 and( 19 in aim_player_discard or 25 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue
                #半筋六索
                if tile_number == 23 and( 20 in aim_player_discard or 26 in aim_player_discard):
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 2
                    continue

            #無筋456:1
                #半筋四萬
                if tile_number == 3:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋五萬
                if tile_number == 4:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋六萬
                if tile_number == 5:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋四筒
                if tile_number == 12:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋五筒
                if tile_number == 13:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋六筒
                if tile_number == 14:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋四索
                if tile_number == 21:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋五索
                if tile_number == 22:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue
                #半筋六索
                if tile_number == 23:
                    opponent_tile_safety_matrix[(current_player + 1 + player_count) % 3][tile_number] = 1
                    continue

        max_opponent_tile_safety = []
        for tile_number in range(34):
            max_opponent_tile_safety.append(min(opponent_tile_safety_matrix[0][tile_number],opponent_tile_safety_matrix[1][tile_number],opponent_tile_safety_matrix[2][tile_number]))
        return max_opponent_tile_safety
    

