from mahjong.shanten import Shanten
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.agari import Agari
from my_struct.action import Action
import const
from mahjong.hand_calculating.hand_config import HandConfig
import numpy as np
import torch
from meld import Meld
import my_struct
TilesConverter.string_to_34_array

class MahjongHelper:
    def __init__(self):
        self.calculator = HandCalculator()
        self.shanten_calculator = Shanten()
    
    def can_long(self, hand_tiles, melds):
        agari = Agari()
        hand_copy = hand_tiles.copy()
        input_melds = []

        for meld in melds:
            hand_copy.extend(meld.tiles34)
            input_melds.append(meld.tiles34)

        input_hand = self.decode_to_tile34(hand_copy)
        return agari.is_agari(input_hand, input_melds)
    
    def can_kong(self, hand_tiles, melds, tile, if_self_round):
        if if_self_round and hand_tiles.count(tile) == 4:#暗槓
            return True
        
        if not if_self_round and hand_tiles.count(tile) == 3:#明槓
            return True
        
        if if_self_round:#加槓
            for meld in melds:
                if meld.type == 'pung' and tile in meld.tiles:
                    return True
        return False

    def calculate_shanten(self,hand_tiles):
        return self.shanten_calculator.calculate_shanten(self.decode_to_tile34(hand_tiles))
    
    def calculate_value(self,original_shanten: int,hand_tiles:list ,action:Action):
        value = 0

        # 檢查槓後的向聽數是否變大 
        new_hand = [t for t in hand_tiles if t not in action.sequence34]
        new_shanten = self.calculate_shanten(new_hand)

        if new_shanten > original_shanten:
            value -= 3  # 槓後胡牌變遠，降低優先度
        elif new_shanten < original_shanten:
            value += 1  # 槓後更接近胡牌，增加優先度

        return value
    
    def best_kong_choice(self,hand_tiles, melds, last_discard, if_self_round):
        kong_candidates = []

        # 暗槓 (手牌中 4 張相同)
        if if_self_round:
            for tile in set(hand_tiles):
                if hand_tiles.count(tile) == 4:
                    kong_candidates.append(Action(const.CONCEALED_KONG,tile,[tile,tile,tile,tile]))

        # 明槓 (手上 3 張相同，場上有人打出)
        if not if_self_round and hand_tiles.count(last_discard) == 3:
            kong_candidates.append(Action(const.KONG, last_discard,[last_discard,last_discard,last_discard,last_discard]))

        # 加槓 (已有碰，且手中還有 1 張)
        if if_self_round:
            for meld in melds:
                if meld.type == const.PONG:
                    pung_tile = meld.tiles34[0]
                    if pung_tile in hand_tiles:
                        kong_candidates.append(Action(const.ADD_KONG,pung_tile,[pung_tile,pung_tile,pung_tile,pung_tile]))

        if len(kong_candidates) == 1:
            return kong_candidates[0]
        
        original_shanten = self.calculate_shanten(hand_tiles)
        
        best_action = max(kong_candidates, key=lambda action: self.calculate_value(original_shanten,hand_tiles,action))
        return best_action

    def can_chow(self, hand_tiles, tile):
        if tile >= 27:  # 字牌不能吃
            return False
        
        suit = tile // 9
        if tile - 2 in hand_tiles and tile - 1 in hand_tiles:
            if (tile - 2) // 9 == suit and (tile - 1) // 9 == suit:
                return True

        if tile - 1 in hand_tiles and tile + 1 in hand_tiles:
            if (tile - 1) // 9 == suit and (tile + 1) // 9 == suit:
                return True
        
        if tile + 1 in hand_tiles and tile + 2 in hand_tiles:
            if (tile + 1) // 9 == suit and (tile + 2) // 9 == suit:
                return True
        
        return False
    
    def best_chow_choice(self, hand_tiles, discard):
        chow_candidates = []

        suit = discard // 9
        #大
        if discard - 2 in hand_tiles and discard - 1 in hand_tiles:
            if (discard - 2) // 9 == suit and (discard - 1) // 9 == suit:
                chow_candidates.append(Action(const.CHOW,discard - 2,[discard - 2,discard - 1,discard]))
        #中間
        if discard - 1 in hand_tiles and discard + 1 in hand_tiles:
            if (discard - 1) // 9 == suit and (discard + 1) // 9 == suit:
                chow_candidates.append(Action(const.CHOW,discard - 1,[discard - 1,discard,discard + 1]))
        #小
        if discard + 1 in hand_tiles and discard + 2 in hand_tiles:
            if (discard + 1) // 9 == suit and (discard + 2) // 9 == suit:
                chow_candidates.append(Action(const.CHOW,discard,[discard,discard + 1,discard + 2]))

        # 如果沒有可以吃的牌，回傳 None
        if not chow_candidates:
            return None

        if len(chow_candidates) == 1:
            return chow_candidates[0]
        
        original_shanten = self.calculate_shanten(hand_tiles)
        
        best_action = max(chow_candidates, key=lambda action: self.calculate_value(original_shanten,hand_tiles,action))
        return best_action
    
    def can_pong(self, hand_tiles, tile):
        return hand_tiles.count(tile) >= 2
    
    def can_riichi(self, hand_tiles, melds):
        if len(melds) != 0:  # 副露過則不能立直
            return False
        shanten = self.calculate_shanten(hand_tiles)
        return shanten == 0
    
    def decode_to_tile34(self,hand_tiles34):
        tile34 = [0]*34
        for tile in hand_tiles34:
            tile34[tile] += 1
        return tile34

    def compute_remain_tiles(self,game_state: my_struct.Game_state, random_player:int):
        remain_tile_count = [4]*34

        #減去寶牌
        for open_dora in range(game_state.open_dora_num):
            remain_tile_count[game_state.dora[open_dora]] -= 1

        #減去手牌
        self_Hai_34_List = self.tileNumberTrans(game_state.players[random_player].hand)
        for tile in self_Hai_34_List:
            remain_tile_count[tile] -= 1

        #減去棄牌
        all_discards = [game_state.players[0].discards, game_state.players[1].discards, game_state.players[2].discards, game_state.players[3].discards]
        for discards in all_discards:
            for tile in discards:
                remain_tile_count[tile] -= 1
        
        #減去副露
        for player in game_state.players:
            for meld in player.meld:
                for tile in meld.tiles34:
                    remain_tile_count[tile] -= 1
        return remain_tile_count

    def decode_to_tile34(self,hand_tiles34):
        tile34 = [0]*34
        for tile in hand_tiles34:
            tile34[tile] += 1
        return tile34
    
    def calc_shanten_change(self,hand_tile:list):
        shanten_calculator = Shanten()
        hand_tile_copy = hand_tile[:]

        shanten_change_34 = [0]*34

        for i in range(34):
            if i in hand_tile_copy:
                hand_tile_copy.remove(i)
                new_shanten = shanten_calculator.calculate_shanten(self.decode_to_tile34(hand_tile_copy))
                shanten_change_34[i] = new_shanten
                hand_tile_copy.append(i)
            else:
                shanten_change_34[i] = 5  # 不在手牌中的牌，設為一個較大的值

        for i in range(34):
            shanten_change_34[i] = shanten_change_34[i] - min(shanten_change_34)

        return shanten_change_34

    def calc_pon_potential(self,hand_tile:list,remain_count:list):
        pon_potential = [0]*34

        hand_tile_count = self.decode_to_tile34(hand_tile)

        for i in range(34):
            if hand_tile_count[i]==3:#已有pon
                pon_potential[i] = 1
            elif hand_tile_count[i] == 2 and remain_count[i] >= 1:#已有pair，且還有剩
                pon_potential[i] = 0.7
            elif hand_tile_count[i] == 1 and remain_count[i] >= 2:#只有單張，且還有剩
                pon_potential[i] = 0.4
            elif hand_tile_count[i] == 0 and remain_count[i] >= 3:#只有單張，且還有剩
                pon_potential[i] = 0.1
            else:
                pon_potential[i] = 0
        return pon_potential


    def calc_chi_potential(self,hand_tile:list,remain_count:list):
        chi_potential_34 = [0]*34

        for i in range(27):
            point = 0.0
            if i % 9 >=2:#(i-2.i-1.i)組合
                if i-1 in hand_tile:
                    point += 1
                else:
                    point += remain_count[i-1] * 0.1

                if i-2 in hand_tile:
                    point += 0.7
                else:
                    point += remain_count[i-2] * 0.05

            if i % 9 >=1 and i % 9 <= 7:#(i-1.i.i+1)組合
                if i-1 in hand_tile:
                    point += 1
                else:
                    point += remain_count[i-1] * 0.1

                if i+1 in hand_tile:
                    point += 1
                else:
                    point += remain_count[i+1] * 0.1

            if i % 9 <=6:#(i.i+1.i+2)組合
                if i+1 in hand_tile:
                    point += 1
                else:
                    point += remain_count[i+1] * 0.1

                if i+2 in hand_tile:
                    point += 0.7
                else:
                    point += remain_count[i+2] * 0.05
        
            chi_potential_34[i] = point

        return chi_potential_34
    
    def decode_Dora_to_136(self,Dora_Hai_list: list, open_dora_num:int):
        Dora_Hai_136_list = [0]*136
        for i in range(open_dora_num):
            dora = Dora_Hai_list[i]
            while(Dora_Hai_136_list[dora] == 1):
                dora = dora + 34
            Dora_Hai_136_list[dora] = 1
        return Dora_Hai_136_list
    
    def decode_4playersDis_to_136(self,game_state: my_struct.Game_state):
        Discard_Hai_136_list = [0]*136
        for dis in game_state.players[0].discards:
            while(Discard_Hai_136_list[dis] == 1):
                dis = dis + 34
            Discard_Hai_136_list[dis] = 1
        for dis in game_state.players[1].discards:
            while(Discard_Hai_136_list[dis] == 1):
                dis = dis + 34
            Discard_Hai_136_list[dis] = 1
        for dis in game_state.players[2].discards:
            while(Discard_Hai_136_list[dis] == 1):
                dis = dis + 34
            Discard_Hai_136_list[dis] = 1
        for dis in game_state.players[3].discards:
            while(Discard_Hai_136_list[dis] == 1):
                dis = dis + 34
            Discard_Hai_136_list[dis] = 1
        return Discard_Hai_136_list
    
    def process_model_input(self,game_state: my_struct.Game_state):

        self_Hai_34_List = self.tileNumberTrans(game_state.players[game_state.current_player].hand)
        right_Hai_34_list = self.tileNumberTrans(self.merge_meld(game_state.players[(game_state.current_player + 1) % 4].meld))
        Opposite_Hai_34_list = self.tileNumberTrans(self.merge_meld(game_state.players[(game_state.current_player + 2) % 4].meld))
        Left_Hai_34_list = self.tileNumberTrans(self.merge_meld(game_state.players[(game_state.current_player + 3) % 4].meld))
        self_Hai_136_List = self.decode_34_to_136(self_Hai_34_List)
        right_Hai_136_List = self.decode_34_to_136(right_Hai_34_list)
        Opposite_Hai_136_List = self.decode_34_to_136(Opposite_Hai_34_list)
        Left_Hai_136_list = self.decode_34_to_136(Left_Hai_34_list)
        Dora_Hai_136_list = self.decode_Dora_to_136(game_state.dora,game_state.open_dora_num)
        Discard_Hai_136_list = self.decode_4playersDis_to_136(game_state)
        rounds_34_list = self.decode_rounds_to_34(game_state.round)
        self_score_34_List = self.decode_score_to_34(game_state.score[game_state.current_player])
        right_score_34_List = self.decode_score_to_34(game_state.score[(game_state.current_player + 1) % 4])
        Opposite_score_34_List = self.decode_score_to_34(game_state.score[(game_state.current_player + 2) % 4])
        Left_score_34_List = self.decode_score_to_34(game_state.score[(game_state.current_player + 3) % 4])

        feature_maps = np.array([
            self_Hai_136_List[0:34],
            self_Hai_136_List[34:68],
            self_Hai_136_List[68:102],
            self_Hai_136_List[102:136],
            right_Hai_136_List[0:34],
            right_Hai_136_List[34:68],
            right_Hai_136_List[68:102],
            right_Hai_136_List[102:136],
            Opposite_Hai_136_List[0:34],
            Opposite_Hai_136_List[34:68],
            Opposite_Hai_136_List[68:102],
            Opposite_Hai_136_List[102:136],
            Left_Hai_136_list[0:34],
            Left_Hai_136_list[34:68],
            Left_Hai_136_list[68:102],
            Left_Hai_136_list[102:136],
            Dora_Hai_136_list[0:34],
            Dora_Hai_136_list[34:68],
            Dora_Hai_136_list[68:102],
            Dora_Hai_136_list[102:136],
            Discard_Hai_136_list[0:34],
            Discard_Hai_136_list[34:68],
            Discard_Hai_136_list[68:102],
            Discard_Hai_136_list[102:136],
            self_score_34_List,
            right_score_34_List,
            Opposite_score_34_List,
            Left_score_34_List,
            rounds_34_list
        ])

        return torch.tensor(feature_maps,dtype=torch.float32)
    
    def new_process_model_input(self,game_state: my_struct.Game_state,random_player: int):

        self_Hai_34_List = self.tileNumberTrans(game_state.players[game_state.current_player].hand)
        right_Hai_34_list = self.tileNumberTrans(self.merge_meld(game_state.players[(game_state.current_player + 1) % 4].meld))
        Opposite_Hai_34_list = self.tileNumberTrans(self.merge_meld(game_state.players[(game_state.current_player + 2) % 4].meld))
        Left_Hai_34_list = self.tileNumberTrans(self.merge_meld(game_state.players[(game_state.current_player + 3) % 4].meld))
        self_Hai_136_List = self.decode_34_to_136(self_Hai_34_List)
        right_Hai_136_List = self.decode_34_to_136(right_Hai_34_list)
        Opposite_Hai_136_List = self.decode_34_to_136(Opposite_Hai_34_list)
        Left_Hai_136_list = self.decode_34_to_136(Left_Hai_34_list)
        Dora_Hai_34_list = self.decode_Dora_to_34(game_state.dora,game_state.open_dora_num)
        Discard_Hai_102_list = self.decode_3playersDis_to_102(game_state,random_player)
        rounds_34_list = self.decode_rounds_to_34(game_state.round)
        self_score_34_List = self.decode_score_to_34(game_state.score[game_state.current_player])
        right_score_34_List = self.decode_score_to_34(game_state.score[(game_state.current_player + 1) % 4])
        Opposite_score_34_List = self.decode_score_to_34(game_state.score[(game_state.current_player + 2) % 4])
        Left_score_34_List = self.decode_score_to_34(game_state.score[(game_state.current_player + 3) % 4])
        remain_tile_count_34 = self.compute_remain_tiles(game_state,random_player)
        shanten_change_34 = self.calc_shanten_change(game_state.players[game_state.current_player].hand)
        pon_potential_34 = self.calc_pon_potential(self_Hai_34_List,remain_tile_count_34)
        chi_potential_34 = self.calc_chi_potential(self_Hai_34_List,remain_tile_count_34)

        feature_maps = np.array([
            self_Hai_136_List[0:34],
            self_Hai_136_List[34:68],
            self_Hai_136_List[68:102],
            self_Hai_136_List[102:136],
            right_Hai_136_List[0:34],
            right_Hai_136_List[34:68],
            right_Hai_136_List[68:102],
            right_Hai_136_List[102:136],
            Opposite_Hai_136_List[0:34],
            Opposite_Hai_136_List[34:68],
            Opposite_Hai_136_List[68:102],
            Opposite_Hai_136_List[102:136],
            Left_Hai_136_list[0:34],
            Left_Hai_136_list[34:68],
            Left_Hai_136_list[68:102],
            Left_Hai_136_list[102:136],
            Dora_Hai_34_list[0:34],
            Discard_Hai_102_list[0:34],
            Discard_Hai_102_list[34:68],
            Discard_Hai_102_list[68:102],
            self_score_34_List,
            right_score_34_List,
            Opposite_score_34_List,
            Left_score_34_List,
            rounds_34_list,
            remain_tile_count_34,
            shanten_change_34,
            pon_potential_34,
            chi_potential_34
        ])

        return torch.tensor(feature_maps,dtype=torch.float32)
    
    def process_predictor_input(self,game_state: my_struct.Game_state):

        self_Hai_34_List = self.tileNumberTrans(game_state.players[game_state.current_player].hand)
        self_discard_34_List = self.tileNumberTrans(game_state.players[game_state.current_player].discards)
        self_score = game_state.score[game_state.current_player]
        self_riichi_info = game_state.riichi_info[game_state.current_player]
        
        right_Hai_34_List = self.tileNumberTrans(game_state.players[(game_state.current_player + 1) % 4].hand)
        right_discard_34_List = self.tileNumberTrans(game_state.players[(game_state.current_player + 1) % 4].discards)
        right_score = game_state.score[(game_state.current_player + 1) % 4]
        right_riichi_info = game_state.riichi_info[(game_state.current_player + 1) % 4]

        opposite_Hai_34_List = self.tileNumberTrans(game_state.players[(game_state.current_player + 2) % 4].hand)
        opposite_discard_34_List = self.tileNumberTrans(game_state.players[(game_state.current_player + 2) % 4].discards)
        opposite_score = game_state.score[(game_state.current_player + 2) % 4]
        opposite_riichi_info = game_state.riichi_info[(game_state.current_player + 2) % 4]

        left_Hai_34_List = self.tileNumberTrans(game_state.players[(game_state.current_player + 3) % 4].hand)
        left_discard_34_List = self.tileNumberTrans(game_state.players[(game_state.current_player + 3) % 4].discards)
        left_score = game_state.score[(game_state.current_player + 3) % 4]
        left_riichi_info = game_state.riichi_info[(game_state.current_player + 3) % 4]

        dora_5_list = self.decode_Dora_to_5(game_state.dora,game_state.open_dora_num)

        feature_maps = np.array([
            0,#場風一律為東風
            game_state.player_wind[game_state.current_player],
            game_state.round
        ])

        feature_maps = np.append(feature_maps,self_Hai_34_List)
        feature_maps = np.append(feature_maps,self_discard_34_List)
        feature_maps = np.append(feature_maps,self_score)

        feature_maps = np.append(feature_maps,right_Hai_34_List)
        feature_maps = np.append(feature_maps,right_discard_34_List)
        feature_maps = np.append(feature_maps,right_score)

        feature_maps = np.append(feature_maps,opposite_Hai_34_List)
        feature_maps = np.append(feature_maps,opposite_discard_34_List)
        feature_maps = np.append(feature_maps,opposite_score)

        feature_maps = np.append(feature_maps,left_Hai_34_List)
        feature_maps = np.append(feature_maps,left_discard_34_List)
        feature_maps = np.append(feature_maps,left_score)

        feature_maps = np.append(feature_maps,self_riichi_info)
        feature_maps = np.append(feature_maps,right_riichi_info)
        feature_maps = np.append(feature_maps,opposite_riichi_info)
        feature_maps = np.append(feature_maps,left_riichi_info)

        feature_maps = np.append(feature_maps,dora_5_list)

        return torch.tensor(feature_maps,dtype=torch.float32).unsqueeze(0)
    
    #game_state轉換成模型輸入用函數
    def merge_meld(self,melds:list):
            meld_list = []
            for meld in melds:
                for tile in meld.tiles34:
                    meld_list.append(tile)
            return meld_list

    def tileNumberTrans(self,playerHai):#將136張牌轉換成34種
        templist = [0] * 34
        for tile in playerHai:
            templist[tile] = templist[tile] + 1
        return templist
    
    def decode_34_to_136(self,Hai_34_List: list):
        Hai_136_List = [0]*136
        for i in range(4):
            for j in range(34):
                if(Hai_34_List[j] > 0):
                    Hai_136_List[i*34+j] = 1
                    Hai_34_List[j] = Hai_34_List[j] - 1
        return Hai_136_List
    
    def decode_Dora_to_34(self,Dora_Hai_list: list, open_dora_num:int):
        Dora_Hai_34_list = [0]*34
        for i in range(open_dora_num):
            Dora_Hai_34_list[Dora_Hai_list[i]] = 1
        return Dora_Hai_34_list
    
    def decode_Dora_to_5(self,Dora_Hai_list: list, open_dora_num:int):
        Dora_Hai_5_list = [-1]*5
        for i in range(open_dora_num):
            Dora_Hai_5_list[i] = Dora_Hai_list[i]
        return Dora_Hai_5_list
    
    def decode_3playersDis_to_102(self,game_state: my_struct.Game_state, random_player: int):
        Discard_Hai_102_list = [0]*102
        all_discards = [game_state.players[0].discards, game_state.players[1].discards, game_state.players[2].discards, game_state.players[3].discards]
        ordered_players = [(random_player + 1) % 4,(random_player + 2) % 4,(random_player + 3) % 4]

        for seg_idx, player in enumerate(ordered_players):
            base = seg_idx * 34  # 第幾段 (0, 34, 68)
            recent_discards = all_discards[player][-6:]  # 該玩家最後 6 張棄牌

            for dis in recent_discards:
                dis34 = int(dis / 4)
                target_idx = base + dis34  # 將該牌標記到該玩家的區段中

                if target_idx < len(Discard_Hai_102_list):
                    Discard_Hai_102_list[target_idx] = 1
        return Discard_Hai_102_list

    def decode_score_to_34(self,score: int):
        score_34 = [0]*34
        index = int(score/20)
        if index < 0:
            index = 0
        elif index >=34:
            index = 33
        score_34[index] = 1
        return score_34

    def decode_rounds_to_34(self,round:int):
        rounds_34 = [0]*34
        index = int((69 - round)/2)
        if index > 33:
            index = 33
        rounds_34[index] = 1
        return rounds_34
    
    def decode_score_to_34(self,score: int):
        score_34 = [0]*34
        index = int(score/20)
        if index < 0:
            index = 0
        elif index >=34:
            index = 33
        score_34[index] = 1
        return score_34
    
if __name__ == "__main__":
    test = MahjongHelper()
    tiles = [4, 5, 7, 10, 16, 20, 21, 22, 22, 24]
    result = test.best_chow_choice(tiles,23)
    print(result.sequence34)