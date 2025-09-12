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



if __name__ == "__main__":
    game = Game()
    stop_flag = False
    while True:
        if keyboard.is_pressed("esc") or stop_flag:
            break
        game.init_game()
        while not game.game_state.game_over:
            if keyboard.is_pressed("esc"):
                stop_flag = True
                break

            game.process_player_actions()
            #game.test()
            game.get_next_player_behavior()
    print("每個ai贏的次數",game.ai_win_times)
    print("已暫停")