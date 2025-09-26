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
from game import Game

stop_flag = False
target_ai = 'E:/專題/discard_model/RL/CNN_discard_model_20250919_155448.pth'
opponent_ai = 'E:/專題/discard_model/RL/best_model.pth'
game = Game(enable_rl = False,target_ai=target_ai,opponent_ai=opponent_ai)
num_games = 500

start_time = time.time()

while True and (num_games > 0):
    if keyboard.is_pressed("esc") or stop_flag:
        break
    game.init_game(game_mode=const.VAL_MODE)
    while not game.game_state.game_over:
        if keyboard.is_pressed("esc"):
            stop_flag = True
            break

        game.process_player_actions()
        #game.test()
        game.get_next_player_behavior()
    num_games -= 1
    print(num_games)

current_time = time.time()
print("每個ai贏的次數",game.ai_win_times)
print("花費時間:",current_time-start_time)
print("已暫停")