from mahjong.shanten import Shanten
from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
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
    
    def can_long(self, hand_tiles, win_tile, config):
        result = self.calculator.estimate_hand_value( 
            tiles=hand_tiles,        # 手牌
            win_tile=win_tile,       # 和牌
            config=config            # 設定 (場風、自風等))
            )
        return result.yaku is not None
    
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
        shanten = Shanten()
        return shanten.calculate_shanten(self.decode_to_tile34(hand_tiles))
    
    def calculate_value(self,original_shanten: int,hand_tiles:list ,action:Action):
        value = 0

        # 檢查槓後的向聽數是否變大 
        new_hand = [t for t in hand_tiles if t not in action.sequence34]
        new_shanten = self.calculate_shanten(hand_tiles)

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

        # 如果沒有可以槓的牌，回傳 None
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

    def process_model_input(self,game_state: my_struct.Game_state):

        def merge_meld(melds:list):
            meld_list = []
            for meld in melds:
                for tile in meld.tiles34:
                    meld_list.append(tile)
            return meld_list

        def tileNumberTrans(playerHai):#將136張牌轉換成34種
            templist = [0] * 34
            for tile in playerHai:
                templist[tile] = templist[tile] + 1
            return templist
        
        def decode_34_to_136(Hai_34_List: list):
            Hai_136_List = [0]*136
            for i in range(4):
                for j in range(34):
                    if(Hai_34_List[j] > 0):
                        Hai_136_List[i*34+j] = 1
                        Hai_34_List[j] = Hai_34_List[j] - 1
            return Hai_136_List
        
        def decode_Dora_to_136(Dora_Hai_list: list):
            Dora_Hai_136_list = [0]*136
            for dora in Dora_Hai_list:
                while(Dora_Hai_136_list[dora] == 1):
                    dora = dora + 34
                Dora_Hai_136_list[dora] = 1
            return Dora_Hai_136_list
        
        def decode_4playersDis_to_136(game_state: my_struct.Game_state):
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

        def decode_score_to_34(score: int):
            score_34 = [0]*34
            index = int(score/20)
            if index < 0:
                index = 0
            elif index >=34:
                index = 33
            score_34[index] = 1
            return score_34

        def decode_rounds_to_34(round:int):
            rounds_34 = [0]*34
            index = int((69 - round)/2)
            if index > 33:
                index = 33
            rounds_34[index] = 1
            return rounds_34
        
        def decode_score_to_34(score: int):
            score_34 = [0]*34
            index = int(score/20)
            if index < 0:
                index = 0
            elif index >=34:
                index = 33
            score_34[index] = 1
            return score_34

        self_Hai_34_List = tileNumberTrans(game_state.players[game_state.current_player].hand)
        right_Hai_34_list = tileNumberTrans(merge_meld(game_state.players[(game_state.current_player + 1) % 4].meld))
        Opposite_Hai_34_list = tileNumberTrans(merge_meld(game_state.players[(game_state.current_player + 2) % 4].meld))
        Left_Hai_34_list = tileNumberTrans(merge_meld(game_state.players[(game_state.current_player + 3) % 4].meld))
        self_Hai_136_List = decode_34_to_136(self_Hai_34_List)
        right_Hai_136_List = decode_34_to_136(right_Hai_34_list)
        Opposite_Hai_136_List = decode_34_to_136(Opposite_Hai_34_list)
        Left_Hai_136_list = decode_34_to_136(Left_Hai_34_list)
        Dora_Hai_136_list = decode_Dora_to_136(game_state.dora)
        Discard_Hai_136_list = decode_4playersDis_to_136(game_state)
        rounds_34_list = decode_rounds_to_34(game_state.round)
        self_score_34_List = decode_score_to_34(game_state.score[game_state.current_player])
        right_score_34_List = decode_score_to_34(game_state.score[(game_state.current_player + 1) % 4])
        Opposite_score_34_List = decode_score_to_34(game_state.score[(game_state.current_player + 2) % 4])
        Left_score_34_List = decode_score_to_34(game_state.score[(game_state.current_player + 3) % 4])

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
    

if __name__ == "__main__":
    test = MahjongHelper()
    tiles = TilesConverter.to_34_array([4,5,12,13,14,44,45,46,52,56,60,84,85,86])
    result = test.shanten_calculator.calculate_shanten(tiles)
    print(result)