# -*- coding: utf-8 -*-
from AI import MahjongAI
import const
import random
from meld import Meld
import time
import keyboard
from model_define import DiscardCNN
from model_define import MyCNN
from model_define import MyGRU
import my_struct

class Game():
    def __init__(self):
        self.game_state = my_struct.Game_state()

        self.mahjong_AIs = [MahjongAI(),MahjongAI(),MahjongAI(),MahjongAI()]
        self.game_mode = const.AI_ONLY
        self.is_game_running = True
    
    def init_game(self,game_mode = const.AI_ONLY):
        self.generate_wall()
        self.deal_tiles()
        self.deal_dora()
        self.game_mode = game_mode

        draw_tile = self.draw_tiles()
        self.game_state.players[self.game_state.current_player].hand.append(draw_tile)
        self.sort_hands_list()
        self.game_state.player_behavior = self.get_self_draw_player_action(draw_tile)
    
    def draw_tiles(self):
        self.game_state.round += 1
        return self.game_state.wall.pop(0)

    def generate_wall(self):
        self.game_state.wall = []

        for i in range(136):
            self.game_state.wall.append(int(i/4))
        
        random.shuffle(self.game_state.wall)
    
    def deal_tiles(self):
        for i in range(4):
            self.game_state.players[i].hand = self.game_state.wall[:13]
            self.game_state.wall = self.game_state.wall[13:]
        self.sort_hands_list()
    
    def deal_dora(self):
        self.game_state.dora = (self.game_state.wall[:5])
        self.game_state.uradora = (self.game_state.wall[5:10])
        self.game_state.wall = self.game_state.wall[10:]
    
    def sort_hands_list(self):#對hands排序
        for i in range(4):
            self.game_state.players[i].hand.sort()

    def next_player(self):
        self.game_state.current_player = (self.game_state.current_player + 1) % 4

    def resume_the_game(self):
        self.is_game_running  = True
    
    def pause_the_game(self):
        self.is_game_running  = False

    def check_other_ai_action(self):#棄牌時檢查其他三家ai動作
        current_player = self.game_state.current_player
        
        right_AI = self.mahjong_AIs[(current_player + 1) % 4]
        right_action = right_AI.process_discard(self.game_state,(current_player + 1) % 4)
        
        opposite_AI = self.mahjong_AIs[(current_player + 2) % 4]
        opposite_action = opposite_AI.process_discard(self.game_state,(current_player + 2) % 4)
        
        left_AI = self.mahjong_AIs[(current_player + 3) % 4]
        left_action = left_AI.process_discard(self.game_state,(current_player + 3) % 4)

        if right_action.type == const.PONG or right_action.type == const.ADD_KONG or right_action.type == const.KONG or right_action.type == const.CONCEALED_KONG:
            return right_action, 1
        elif opposite_action.type == const.PONG or opposite_action.type == const.ADD_KONG or opposite_action.type == const.KONG or opposite_action.type == const.CONCEALED_KONG:
            return opposite_action, 2
        elif left_action.type == const.PONG or left_action.type == const.ADD_KONG or left_action.type == const.KONG or left_action.type == const.CONCEALED_KONG:
            return left_action, 3
        elif right_action.type == const.CHOW:
            return right_action, 1
        elif opposite_action.type == const.CHOW:
            return opposite_action, 2
        elif left_action.type == const.CHOW:
            return left_action, 3

        return None,-1#沒有人吃、碰、槓
    
    def get_self_draw_player_action(self,draw_tile:int):
        current_player = self.game_state.current_player
        cur_AI = self.mahjong_AIs[current_player]
        return cur_AI.process_draw(self.game_state,draw_tile)

    def get_other_discard_player_action(self):
        current_player = self.game_state.current_player
        cur_AI = self.mahjong_AIs[current_player]
        return cur_AI.process_discard(self.game_state)
    
    def get_just_discard_player_action(self):
        current_player = self.game_state.current_player
        cur_AI = self.mahjong_AIs[current_player]
        return cur_AI.just_discard(self.game_state)
    
    def process_player_actions(self):
        if not self.is_game_running:
            return
        
        current_player = self.game_state.current_player

        if self.game_state.player_behavior.type == const.CHOW:
            new_meld = Meld(self.game_state.player_behavior.type,self.game_state.player_behavior.sequence34)
            self.game_state.players[current_player].hand.append(self.game_state.last_discarded_tile)
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[0])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[1])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[2])
            self.game_state.players[current_player].meld.append(new_meld)
        elif self.game_state.player_behavior.type == const.PONG:
            new_meld = Meld(self.game_state.player_behavior.type,self.game_state.player_behavior.sequence34)
            self.game_state.players[current_player].hand.append(self.game_state.last_discarded_tile)
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[0])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[1])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[2])
            self.game_state.players[current_player].meld.append(new_meld)
        elif self.game_state.player_behavior.type == const.ADD_KONG:
            if self.game_state.open_dora_num < 5:
                self.game_state.open_dora_num += 1
            new_meld = Meld(self.game_state.player_behavior.type,self.game_state.player_behavior.sequence34)

            #移除之前碰的紀錄
            target_type = const.PONG
            target_tile34 = int(self.game_state.player_behavior.tile34 / 4)
            for meld in  self.game_state.players[current_player].meld:
                if meld.type == target_type and meld.tiles34[0] == target_tile34:
                    self.game_state.players[current_player].meld.remove(meld)
                    break
            
            self.game_state.players[current_player].append(new_meld)

        elif self.game_state.player_behavior.type == const.KONG:
            if self.game_state.open_dora_num < 5:
                self.game_state.open_dora_num += 1
            new_meld = Meld(self.game_state.player_behavior.type,self.game_state.player_behavior.sequence34)
            self.game_state.players[current_player].hand.append(self.game_state.last_discarded_tile)
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[0])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[1])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[2])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[3])
            self.game_state.players[current_player].meld.append(new_meld)

        elif self.game_state.player_behavior.type == const.CONCEALED_KONG:
            if self.game_state.open_dora_num < 5:
                self.game_state.open_dora_num += 1
            new_meld = Meld(self.game_state.player_behavior.type,self.game_state.player_behavior.sequence34)
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[0])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[1])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[2])
            self.game_state.players[current_player].hand.remove(new_meld.tiles34[3])
            self.game_state.players[current_player].meld.append(new_meld)

        elif self.game_state.player_behavior.type == const.DISCARD:
            self.game_state.players[current_player].hand.remove(self.game_state.player_behavior.tile34)
            self.game_state.players[current_player].discards.append(self.game_state.player_behavior.tile34)
            self.game_state.last_discarded_tile = self.game_state.player_behavior.tile34
        
        elif self.game_state.player_behavior.type == const.RIICHI:
            self.game_state.riichi_info[current_player] = True
        
        elif self.game_state.player_behavior.type == const.SELF_LONG:
            self.game_state.game_over = True
        
        elif self.game_state.player_behavior.type == const.OTHER_LONG:
            self.game_state.game_over = True
    
    def get_next_player_behavior(self):
        if not self.is_game_running:
            return
        
        current_player = self.game_state.current_player

        if self.game_state.player_behavior.type == const.CHOW:
            self.game_state.player_behavior = self.get_just_discard_player_action()
        elif self.game_state.player_behavior.type == const.PONG:
            self.game_state.player_behavior = self.get_just_discard_player_action()
        elif self.game_state.player_behavior.type == const.ADD_KONG:
            draw_tile = self.draw_tiles()
            self.game_state.players[current_player].hand.append(draw_tile)
            self.sort_hands_list()
            self.game_state.player_behavior = self.get_self_draw_player_action(draw_tile)
        elif self.game_state.player_behavior.type == const.KONG:
            draw_tile = self.draw_tiles()
            self.game_state.players[current_player].hand.append(draw_tile)
            self.sort_hands_list()
            self.game_state.player_behavior = self.get_self_draw_player_action(draw_tile)
        elif self.game_state.player_behavior.type == const.CONCEALED_KONG:
            draw_tile = self.draw_tiles()
            self.game_state.players[current_player].hand.append(draw_tile)
            self.sort_hands_list()
            self.game_state.player_behavior = self.get_self_draw_player_action(draw_tile)
        elif self.game_state.player_behavior.type == const.DISCARD:
            other_ai_action, position_offset =  self.check_other_ai_action()

            if other_ai_action != None:
                self.game_state.player_behavior = other_ai_action
                self.game_state.current_player = (current_player + position_offset) % 4
            else:
                self.next_player()
                draw_tile = self.draw_tiles()
                self.game_state.players[self.game_state.current_player].hand.append(draw_tile)
                self.sort_hands_list()
                self.game_state.player_behavior = self.get_self_draw_player_action(draw_tile)
        elif self.game_state.player_behavior.type == const.RIICHI:
            draw_tile = self.draw_tiles()
            self.game_state.players[current_player].hand.append(draw_tile)
            self.sort_hands_list()
            self.game_state.player_behavior = self.get_self_draw_player_action(draw_tile)
    
    def start_game_loop(self):#這後端測試用
        while not self.game_state.game_over:
            self.process_player_actions()
            self.test()
            self.get_next_player_behavior()
            self.pause_the_game()
            while not self.is_game_running:
                if keyboard.is_pressed('a'):
                    self.resume_the_game()
                    continue
                elif keyboard.is_pressed('esc'):
                    self.game_state.game_over = True
                    break
            time.sleep(0.1)

    def test(self): 
        print("current_player:",self.game_state.current_player)
        print("player_behavior",self.game_state.player_behavior.type, self.game_state.player_behavior.tile34, self.game_state.player_behavior.sequence34)
        
        for player in range(4):
            print("palyer",player,":",end=" ")
            print("hand:",self.game_state.players[player].hand, end=" ")
            print("discard:",self.game_state.players[player].discards, end=" ")
            print("meld:", end=" ")
            for meld in self.game_state.players[player].meld:
                print(meld.type,meld.tiles34,end = " ")
            print("")

        print("")
        print("last_discarded_tile:",self.game_state.last_discarded_tile) 
        print("wall_count",len(self.game_state.wall))
        print("dora:",self.game_state.dora,self.game_state.uradora)
        print("")

#測試區
if __name__ == "__main__":
    game = Game()
    game.init_game()
    game.start_game_loop()