#副露/行為類別
SELF_LONG = "self_long"
OTHER_LONG = "other_long"
KONG = "kong"
CONCEALED_KONG = "concealed_kong"
ADD_KONG = "add_kong"
RIICHI = "riichi"
DISCARD = "discard"
CHOW = "chow"
PONG = "pong"
NONE = "none"
THRESHOLD = 0.5

#遊戲模式
AI_ONLY = 1

#呼叫前端行為
NOTHING = 0
DISCARD_AFTER_MELD = 1
REACT_TO_DISCARD = 2
REACT_TO_DRAW_TILE = 3

#牌間距
Tile_width = 50

#前端整體位置
fronted_x_offset = 50
fronted_y_offset = 0

#前端玩家手牌_副露offset
Player0_hand_meld_x_offset = 350 + fronted_x_offset
Player0_hand_meld_y_offset = 800 + fronted_y_offset
Player1_hand_meld_x_offset = 1150 + fronted_x_offset
Player1_hand_meld_y_offset = 780 + fronted_y_offset
Player2_hand_meld_x_offset = 1080 + fronted_x_offset
Player2_hand_meld_y_offset = 100 + fronted_y_offset
Player3_hand_meld_x_offset = 200 + fronted_x_offset
Player3_hand_meld_y_offset = 150 + fronted_y_offset

#前端棄牌位置
Player0_discard_x_offset = 550 + fronted_x_offset
Player0_discard_y_offset = 570 + fronted_y_offset
Player1_discard_x_offset = 830 + fronted_x_offset
Player1_discard_y_offset = 530 + fronted_y_offset
Player2_discard_x_offset = 780 + fronted_x_offset
Player2_discard_y_offset = 290 + fronted_y_offset
Player3_discard_x_offset = 500 + fronted_x_offset
Player3_discard_y_offset = 320 + fronted_y_offset

#前端分數與風位置
Player0_score_and_wind_x_offset = 660 + fronted_x_offset
Player0_score_and_wind_y_offset = 530 + fronted_y_offset
Player1_score_and_wind_x_offset = 790 + fronted_x_offset
Player1_score_and_wind_y_offset = 390 + fronted_y_offset
Player2_score_and_wind_x_offset = 620 + fronted_x_offset
Player2_score_and_wind_y_offset = 330 + fronted_y_offset
Player3_score_and_wind_x_offset = 550 + fronted_x_offset
Player3_score_and_wind_y_offset = 420 + fronted_y_offset

#前端回合數
Round_x_offset = 620 + fronted_x_offset
Round_y_offset = 470 + fronted_y_offset

#前端寶牌位置
Dora_x_offset = 10 + fronted_x_offset
Dora_y_offset = 50 + fronted_y_offset