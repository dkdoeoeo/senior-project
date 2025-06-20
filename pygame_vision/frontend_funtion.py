import pygame
from game import Game
import const

# 讀取麻將牌圖像
tiles_images = {
    0: pygame.image.load("pygame_vision/images/1m.png"),
    1: pygame.image.load("pygame_vision/images/2m.png"),
    2: pygame.image.load("pygame_vision/images/3m.png"),
    3: pygame.image.load("pygame_vision/images/4m.png"),
    4: pygame.image.load("pygame_vision/images/5m.png"),
    5: pygame.image.load("pygame_vision/images/6m.png"),
    6: pygame.image.load("pygame_vision/images/7m.png"),
    7: pygame.image.load("pygame_vision/images/8m.png"),
    8: pygame.image.load("pygame_vision/images/9m.png"),
    9: pygame.image.load("pygame_vision/images/1t.png"),
    10: pygame.image.load("pygame_vision/images/2t.png"),
    11: pygame.image.load("pygame_vision/images/3t.png"),
    12: pygame.image.load("pygame_vision/images/4t.png"),
    13: pygame.image.load("pygame_vision/images/5t.png"),
    14: pygame.image.load("pygame_vision/images/6t.png"),
    15: pygame.image.load("pygame_vision/images/7t.png"),
    16: pygame.image.load("pygame_vision/images/8t.png"),
    17: pygame.image.load("pygame_vision/images/9t.png"),
    18: pygame.image.load("pygame_vision/images/1s.png"),
    19: pygame.image.load("pygame_vision/images/2s.png"),
    20: pygame.image.load("pygame_vision/images/3s.png"),
    21: pygame.image.load("pygame_vision/images/4s.png"),
    22: pygame.image.load("pygame_vision/images/5s.png"),
    23: pygame.image.load("pygame_vision/images/6s.png"),
    24: pygame.image.load("pygame_vision/images/7s.png"),
    25: pygame.image.load("pygame_vision/images/8s.png"),
    26: pygame.image.load("pygame_vision/images/9s.png"),
    27: pygame.image.load("pygame_vision/images/East.png"),
    28: pygame.image.load("pygame_vision/images/South.png"),
    29: pygame.image.load("pygame_vision/images/West.png"),
    30: pygame.image.load("pygame_vision/images/North.png"),
    31: pygame.image.load("pygame_vision/images/White.png"),
    32: pygame.image.load("pygame_vision/images/Green.png"),
    33: pygame.image.load("pygame_vision/images/Red.png"),
    34: pygame.image.load("pygame_vision/images/tile_back.png")
}

wind = {
    0:"東",
    1:"南",
    2:"西",
    3:"北",
}

# 顯示玩家手牌的函式
def display_hands_melds(screen,game:Game):
    """
    顯示指定玩家的手牌。
    player_index: 玩家索引 (0: 玩家1, 1: 玩家2, 2: 玩家3, 3: 玩家4)
    hand: 玩家手牌
    screen: 顯示的畫布
    tiles_images: 牌的圖片字典
    """
    # 遍歷每一位玩家（假設有四位玩家）
    for player_index in range(4):
        hand = game.game_state["hands"][player_index]
        melds = game.game_state["melds"][player_index]
        
        # 設定每個玩家手牌的起始位置與旋轉角度
        if player_index == 0:  # 玩家1（底部）
            x_offset, y_offset = const.Player0_hand_meld_x_offset, const.Player0_hand_meld_y_offset
            rotation = 0
            for i, tile in enumerate(hand):
                rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                x_offset += const.Tile_width
                screen.blit(rotated_tile, (x_offset, y_offset))

            x_offset += const.Tile_width / 2

            for meld in melds:
                for tile in meld.tiles34:
                    rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                    x_offset += const.Tile_width
                    screen.blit(rotated_tile, (x_offset, y_offset))
                x_offset += const.Tile_width / 2
        
        elif player_index == 1:  # 玩家2（右邊）
            x_offset, y_offset = const.Player1_hand_meld_x_offset, const.Player1_hand_meld_y_offset
            rotation = 90
            for i, tile in enumerate(hand):
                rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                y_offset -= const.Tile_width
                screen.blit(rotated_tile, (x_offset, y_offset))
            
            y_offset -= const.Tile_width / 2

            for meld in melds:
                for tile in meld.tiles34:
                    rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                    y_offset -= const.Tile_width
                    screen.blit(rotated_tile, (x_offset, y_offset))
                y_offset -= const.Tile_width / 2
            
        
        elif player_index == 2:  # 玩家3（上方）
            x_offset, y_offset = const.Player2_hand_meld_x_offset, const.Player2_hand_meld_y_offset
            rotation = 180
            for i, tile in enumerate(hand):
                rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                x_offset -= const.Tile_width
                screen.blit(rotated_tile, (x_offset, y_offset))
            
            x_offset -= const.Tile_width / 2

            for meld in melds:
                for tile in meld.tiles34:
                    rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                    x_offset -= const.Tile_width
                    screen.blit(rotated_tile, (x_offset, y_offset))
                x_offset -= const.Tile_width / 2
        
        elif player_index == 3:  # 玩家4（左邊）
            x_offset, y_offset = const.Player3_hand_meld_x_offset, const.Player3_hand_meld_y_offset
            rotation = 270
            for i, tile in enumerate(hand):
                rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                y_offset += const.Tile_width
                screen.blit(rotated_tile, (x_offset, y_offset))
            
            y_offset += const.Tile_width / 2

            for meld in melds:
                for tile in meld.tiles34:
                    rotated_tile = pygame.transform.rotate(tiles_images[tile], rotation)
                    y_offset += const.Tile_width
                    screen.blit(rotated_tile, (x_offset, y_offset))
                y_offset += const.Tile_width / 2

def display_discard(screen,game:Game):
    discard_per_row = 6  # 每行顯示的棄牌數量
    discard_size = 40  # 棄牌的縮小大小

    for player_index in range(4):
        discard = game.game_state["discards"][player_index]
        # 設定棄牌堆的位置
        if player_index == 0:  # 玩家1（底部）
            discard_x_offset, discard_y_offset = const.Player0_discard_x_offset, const.Player0_discard_y_offset
            for i, tile in enumerate(discard):
                row = i // discard_per_row  # 計算棄牌所在的行數
                col = i % discard_per_row  # 計算棄牌所在的列數
                # 繪製每張棄牌，並縮小圖片
                discard_tile = pygame.transform.scale(tiles_images[tile], (discard_size, discard_size))
                screen.blit(discard_tile, (discard_x_offset + col * (discard_size + 5), discard_y_offset + row * (discard_size + 5)))
        
        elif player_index == 1:  # 玩家2（右邊）
            discard_x_offset, discard_y_offset = const.Player1_discard_x_offset, const.Player1_discard_y_offset
            rotation = 90
            for i, tile in enumerate(discard):
                row = i % discard_per_row  # 計算棄牌所在的行數
                col = i // discard_per_row  # 計算棄牌所在的列數
                # 繪製每張棄牌，並縮小圖片
                discard_tile = pygame.transform.scale(tiles_images[tile], (discard_size, discard_size))
                rotated_tile = pygame.transform.rotate(discard_tile, rotation)
                screen.blit(rotated_tile, (discard_x_offset + col * (discard_size + 5), discard_y_offset - row * (discard_size + 5)))
        
        elif player_index == 2:  # 玩家3（上方）
            discard_x_offset, discard_y_offset = const.Player2_discard_x_offset, const.Player2_discard_y_offset
            rotation = 180
            for i, tile in enumerate(discard):
                row = i // discard_per_row  # 計算棄牌所在的行數
                col = i % discard_per_row  # 計算棄牌所在的列數
                # 繪製每張棄牌，並縮小圖片
                discard_tile = pygame.transform.scale(tiles_images[tile], (discard_size, discard_size))
                rotated_tile = pygame.transform.rotate(discard_tile, rotation)
                screen.blit(rotated_tile, (discard_x_offset - col * (discard_size + 5), discard_y_offset - row * (discard_size + 5)))
        
        elif player_index == 3:  # 玩家4（左邊）
            discard_x_offset, discard_y_offset = const.Player3_discard_x_offset, const.Player3_discard_y_offset
            rotation = 270
            for i, tile in enumerate(discard):
                row = i % discard_per_row  # 計算棄牌所在的行數
                col = i // discard_per_row  # 計算棄牌所在的列數
                # 繪製每張棄牌，並縮小圖片
                discard_tile = pygame.transform.scale(tiles_images[tile], (discard_size, discard_size))
                rotated_tile = pygame.transform.rotate(discard_tile, rotation)
                screen.blit(rotated_tile, (discard_x_offset - col * (discard_size + 5), discard_y_offset + row * (discard_size + 5)))
            
def display_score_and_wind(screen,game:Game):
    font = pygame.font.Font("pygame_vision/msyh.ttc", 30)  # 設定顯示分數的字型和大小

    for player_index in range(4):
        score = game.game_state["score"][player_index] * 100
        
        if player_index == 0:  # 玩家1（底部）
            score_x_offset, score_y_offset = const.Player0_score_and_wind_x_offset, const.Player0_score_and_wind_y_offset
            score_text = font.render(str(score), True, (0, 0, 0))
            screen.blit(score_text, (score_x_offset, score_y_offset))
            wind_text = font.render(wind[game.game_state["player_wind"][player_index]], True, (0, 0, 0))
            screen.blit(wind_text, (score_x_offset - 40, score_y_offset))
        
        elif player_index == 1:  # 玩家2（右邊）
            score_x_offset, score_y_offset = const.Player1_score_and_wind_x_offset, const.Player1_score_and_wind_y_offset
            rotation = 90
            score_text = font.render(str(score), True, (0, 0, 0))
            rotated_score_text = pygame.transform.rotate(score_text, rotation)
            screen.blit(rotated_score_text, (score_x_offset, score_y_offset))
            wind_text = font.render(wind[game.game_state["player_wind"][player_index]], True, (0, 0, 0))
            rotated_wind_text = pygame.transform.rotate(wind_text, rotation)
            screen.blit(rotated_wind_text, (score_x_offset, score_y_offset + 100))
        
        elif player_index == 2:  # 玩家3（上方）
            score_x_offset, score_y_offset = const.Player2_score_and_wind_x_offset, const.Player2_score_and_wind_y_offset  # 分數顯示在棄牌堆下方
            rotation = 180
            score_text = font.render(str(score), True, (0, 0, 0))
            rotated_score_text = pygame.transform.rotate(score_text, rotation)
            screen.blit(rotated_score_text, (score_x_offset, score_y_offset))
            wind_text = font.render(wind[game.game_state["player_wind"][player_index]], True, (0, 0, 0))
            rotated_wind_text = pygame.transform.rotate(wind_text, rotation)
            screen.blit(rotated_wind_text, (score_x_offset + 100, score_y_offset))
        
        elif player_index == 3:  # 玩家4（左邊）
            score_x_offset, score_y_offset = const.Player3_score_and_wind_x_offset, const.Player3_score_and_wind_y_offset  # 分數顯示在棄牌堆左側
            rotation = 270
            score_text = font.render(str(score), True, (0, 0, 0))
            rotated_score_text = pygame.transform.rotate(score_text, rotation)
            screen.blit(rotated_score_text, (score_x_offset, score_y_offset))
            wind_text = font.render(wind[game.game_state["player_wind"][player_index]], True, (0, 0, 0))
            rotated_wind_text = pygame.transform.rotate(wind_text, rotation)
            screen.blit(rotated_wind_text, (score_x_offset, score_y_offset - 40))

def display_round(screen,game:Game):
    font = pygame.font.Font("pygame_vision/msyh.ttc", 30)

    round_x_offset, round_y_offset = const.Round_x_offset, const.Round_y_offset
    round_text = font.render("剩餘牌山"+str(70 - game.game_state["round"]), True, (0, 0, 0))
    screen.blit(round_text, (round_x_offset, round_y_offset))

def display_dora(screen,game:Game):

    dora_x_offset, dora_y_offset = const.Dora_x_offset, const.Dora_y_offset
    for i in range(5):
        dora_x_offset += const.Tile_width
        if i < game.game_state["open_dora_num"]:
            image = tiles_images[game.game_state["dora"][i]]
        else:
            image = tiles_images[34]
        screen.blit(image,(dora_x_offset,dora_y_offset))
        