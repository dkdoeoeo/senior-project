import AI_function
import torch
import const
import torch.nn.functional as F

class MahjongAI():
    def __init__(self):
        self.discard_model = torch.load("discard_model.pth")
        self.pung_model = torch.load("pung_model.pth")
        self.chow_model = torch.load("chow_model.pth")
        self.kong_model = torch.load("kong_model.pth")
        self.riichi_model = torch.load("riichi_model.pth")
    
    def process_draw(self,model_input: list,game_state: dict):#處理自己摸牌
        if AI_function.if_can_long(game_state):
            return const.LONG,const.NONE
        
        if AI_function.if_can_kong(game_state):
            kong_point = self.kong_model(model_input)

            if kong_point >= const.THRESHOLD:
                return const.KONG,const.NONE

        if AI_function.if_can_riichi(game_state):
            riichi_point = self.riichi_model(model_input)

            if riichi_point >= const.THRESHOLD:
                return const.RIICHI,const.NONE

        discard_output = self.discard_model(model_input).float()
        discard_probabilities = F.softmax(discard_output, dim=0)
        discard = torch.argmax(discard_probabilities, dim=0).item()
        return const.DISCARD,discard

    def process_discard(self,model_input: list,game_state: dict):#處理別人打牌
        if AI_function.if_can_long(game_state):
            return const.LONG,const.NONE
        
        chow_point = 0
        pung_point = 0
        kong_point = 0

        if AI_function.if_can_chow(game_state):
            chow_point = self.chow_model(model_input)
        
        if AI_function.if_can_pung(game_state):
            pung_point = self.pung_model(model_input)
        
        if AI_function.if_can_kong(game_state):
            kong_point = self.kong_model(model_input)

        if chow_point >= const.THRESHOLD or pung_point >= const.THRESHOLD or kong_point >= const.THRESHOLD:
            
            if max(chow_point,pung_point,kong_point) == kong_point:
                return const.KONG,const.NONE
            if max(chow_point,pung_point,kong_point) == pung_point:
                return const.PUNG,const.NONE
            if max(chow_point,pung_point,kong_point) == chow_point:
                return const.CHOW,const.NONE
            return const.NONE,const.NONE