import pygame
import random
from model_define import DiscardCNN
from model_define import MyCNN
from game import Game
import frontend_funtion
import const
import random
import time
import keyboard

pygame.init()

screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("日麻前端")

# 設定遊戲窗口
screen_width, screen_height = 1500, 950
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("日麻前端")

# 設定顏色
WHITE = (255, 255, 255)

game = Game()
game.init_game()

# 遊戲循環
running = True
while running and not game.game_state["game_over"]:
    screen.fill(WHITE)  # 清除畫面
    frontend_funtion.display_hands_melds(screen,game)
    frontend_funtion.display_discard(screen,game)
    frontend_funtion.display_score_and_wind(screen,game)
    frontend_funtion.display_round(screen,game)
    frontend_funtion.display_dora(screen,game)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pass
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                game.process_player_actions()
                game.get_next_player_behavior()
                game.pause_the_game()
                game.test()
                game.resume_the_game()
            elif event.key == pygame.K_ESCAPE:
                running = False

    pygame.display.flip()

pygame.quit()