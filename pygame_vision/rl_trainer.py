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
    while True:
        if keyboard.is_pressed("esc"):
            break
        game.init_game()
        while not game.game_state.game_over:
            if keyboard.is_pressed("esc"):
                exit(0)
                
            game.process_player_actions()
            #game.test()
            game.get_next_player_behavior()
    print("已暫停")