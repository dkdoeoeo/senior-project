import pygame
import sys
import copy
import traceback
from model_define import DiscardCNN
from model_define import MyCNN
from model_define import MyGRU

# --- 使用者設定區 (請修改這裡) ---
# 請填入你的模型路徑與類型
"E:\專題\discard_model\final_discard_model\new_feature_15_256.pth"
TARGET_AI_PATH = "E:/專題/discard_model/final_discard_model/new_feature_15_256.pth" 
TARGET_AI_TYPE = "15_256" # 或其他定義的 type

# --- 模擬 Const 模組 ---
class MockConst:
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
    CHOW_THRESHOLD = 0.45
    PONG_THRESHOLD = 0.25
    KONG_THRESHOLD = 0.5
    RIICHI_THRESHOLD = 0.5

# --- 模組匯入區 ---
USER_MODULES_AVAILABLE = False
try:
    import my_struct
    from AI import MahjongAI
    import const
    from MahjongHelper import MahjongHelper
    USER_MODULES_AVAILABLE = True
    print(">> 成功載入 AI 模組與資料結構！")
except ImportError as e:
    print(f"Warning: {e}")
    print(">> 系統將運作於 [GUI 模擬模式] 並使用 Mock 資料結構。")
    const = MockConst()

# --- 全域 UI 設定 ---
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
BG_COLOR = (30, 100, 50) # 深綠色桌面
PANEL_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)
TILE_WIDTH = 36
TILE_HEIGHT = 54
FONT_SIZE = 20

# --- 資料轉換工具 ---
TILE_34_MAP = {}
TILES = [] 
idx = 0
for suit in ['m', 'p', 's']:
    for i in range(1, 10):
        t_str = f"{i}{suit}"
        TILES.append(t_str)
        TILE_34_MAP[idx] = t_str
        idx += 1
for i in range(1, 8):
    t_str = f"{i}z"
    TILES.append(t_str)
    TILE_34_MAP[idx] = t_str
    idx += 1

STRING_TO_34 = {v: k for k, v in TILE_34_MAP.items()}
TILE_DISPLAY_MAP = {
    '1z': '東', '2z': '南', '3z': '西', '4z': '北',
    '5z': '白', '6z': '發', '7z': '中'
}

def get_tile_name(tile34_idx):
    if 0 <= tile34_idx < 34:
        raw = TILE_34_MAP[tile34_idx]
        return TILE_DISPLAY_MAP.get(raw, raw)
    return "??"

def get_tile_str(tile34_idx):
    return TILE_34_MAP.get(tile34_idx, "1m")

# --- 類別定義 ---

class Meld:
    def __init__(self, type=const.NONE, tiles34=None):
        self.type = type
        self.tiles34 = tiles34 if tiles34 is not None else []
    
    def __repr__(self):
        tiles_str = [get_tile_name(t) for t in self.tiles34]
        return f"{self.type}:{tiles_str}"

class Player:
    def __init__(self):
        self.hand = []      
        self.discards = []  
        self.meld = []      

class UIGameState:
    def __init__(self):
        self.players = [Player() for _ in range(4)]
        self.current_editing_player = 0 # 對應 backend.current_player
        
        # 全域資訊
        self.dora_indicators = [] 
        self.my_wind = 0 
        self.scores = [25000, 25000, 25000, 25000]
        self.riichi_info = [False, False, False, False] # 立直狀態

        self.is_dirty = True 

    def mark_dirty(self):
        self.is_dirty = True

    def switch_player(self, player_idx):
        if self.current_editing_player != player_idx:
            self.current_editing_player = player_idx
            self.mark_dirty()

    def add_tile_to_hand(self, player_idx, tile34):
        p = self.players[player_idx]
        if len(p.hand) < 14:
            p.hand.append(tile34)
            self.sort_hand(player_idx)
            self.current_editing_player = player_idx # 自動切換當前玩家
            self.mark_dirty()

    def remove_tile_from_hand(self, player_idx, index):
        p = self.players[player_idx]
        if 0 <= index < len(p.hand):
            p.hand.pop(index)
            self.mark_dirty()

    def sort_hand(self, player_idx):
        self.players[player_idx].hand.sort()

    def add_discard(self, player_idx, tile34):
        self.players[player_idx].discards.append(tile34)
        self.current_editing_player = player_idx # 自動切換當前玩家
        self.mark_dirty()
    
    def remove_last_discard(self, player_idx):
        if self.players[player_idx].discards:
            self.players[player_idx].discards.pop()
            self.mark_dirty()

    def add_meld(self, player_idx, meld_obj):
        p = self.players[player_idx]
        if meld_obj.type == const.ADD_KONG:
            target_tile = meld_obj.tiles34[0]
            found_pong_idx = -1
            for i, m in enumerate(p.meld):
                if m.type == const.PONG and target_tile in m.tiles34:
                    found_pong_idx = i
                    break
            if found_pong_idx != -1:
                p.meld.pop(found_pong_idx)
        p.meld.append(meld_obj)
        self.current_editing_player = player_idx # 自動切換當前玩家
        self.mark_dirty()

    def remove_meld(self, player_idx, meld_index):
        p = self.players[player_idx]
        if 0 <= meld_index < len(p.meld):
            p.meld.pop(meld_index)
            self.mark_dirty()

    def add_dora(self, tile34):
        if len(self.dora_indicators) < 10:
            self.dora_indicators.append(tile34)
            self.mark_dirty()

    def remove_last_dora(self):
        if self.dora_indicators:
            self.dora_indicators.pop()
            self.mark_dirty()

    def set_wind(self, wind):
        self.my_wind = wind
        self.mark_dirty()
    
    def update_score(self, player_idx, new_score):
        self.scores[player_idx] = new_score
        self.mark_dirty()

    def toggle_riichi(self, player_idx):
        self.riichi_info[player_idx] = not self.riichi_info[player_idx]
        self.mark_dirty()

class AIWrapper:
    def __init__(self):
        self.ai = None
        if USER_MODULES_AVAILABLE:
            try:
                print(f"正在載入模型: {TARGET_AI_PATH} ...")
                self.ai = MahjongAI(discard_model_file_pth=TARGET_AI_PATH, ai_type=TARGET_AI_TYPE)
                print("MahjongAI 初始化成功。")
            except Exception as e:
                print(f"初始化 MahjongAI 失敗: {e}")
                traceback.print_exc()
        
    def _sync_to_real_state(self, ui_state, exclude_last_hand_tile=False):
        if USER_MODULES_AVAILABLE:
            real_state = my_struct.Game_state()
        else:
            class MockState:
                def __init__(self):
                    self.players = [Player() for _ in range(4)]
                    self.current_player = 0
                    self.dora = []
                    self.player_wind = [0,1,2,3]
                    self.score = [250,250,250,250]
                    self.riichi_info = [False,False,False,False]
            real_state = MockState()

        # 1. 同步全域資訊
        real_state.dora = copy.deepcopy(ui_state.dora_indicators)
        start_wind = ui_state.my_wind
        real_state.player_wind = [(start_wind + i) % 4 for i in range(4)]
        real_state.score = [int(s // 100) for s in ui_state.scores]
        
        # 關鍵修改：current_player 同步 UI 上的編輯對象
        real_state.current_player = ui_state.current_editing_player
        
        # Riichi info
        real_state.riichi_info = copy.deepcopy(ui_state.riichi_info)

        # 2. 同步玩家
        for i in range(4):
            ui_p = ui_state.players[i]
            real_p = real_state.players[i]
            
            # Hand (全部同步，方便 debug，雖然 AI 可能只看 P0)
            current_hand = copy.deepcopy(ui_p.hand)
            # 如果是 P0 且需要模擬 process_draw 前的狀態
            if i == 0 and exclude_last_hand_tile and len(current_hand) > 0:
                current_hand.pop()
            real_p.hand = current_hand

            # Discards & Melds
            real_p.discards = copy.deepcopy(ui_p.discards)
            real_p.meld = copy.deepcopy(ui_p.meld)
            
        return real_state

    def debug_print_state(self, ui_state):
        """印出 Game_state 所有請求的資訊"""
        try:
            rs = self._sync_to_real_state(ui_state)
            print("\n" + "="*20 + " Backend State Check " + "="*20)
            print(f"[Global Info]")
            print(f"  Current Player : {rs.current_player}")
            print(f"  Dora Indicators: {[get_tile_name(t) for t in rs.dora]}")
            print(f"  Scores (raw/100): {ui_state.scores} -> {rs.score}")
            print(f"  Riichi Info    : {rs.riichi_info}")
            print(f"  Player Winds   : {rs.player_wind} (My Wind: {ui_state.my_wind})")
            
            print("-" * 20)
            for i in range(4):
                p = rs.players[i]
                p_label = f"Player {i}"
                if i == rs.current_player: p_label += " [CURRENT]"
                print(f"{p_label}:")
                
                # 手牌
                hand_str = [get_tile_name(t) for t in p.hand]
                print(f"  Hand     : {hand_str}")
                
                # 副露
                print(f"  Melds    : {p.meld}")
                
                # 棄牌
                disc_str = [get_tile_name(t) for t in p.discards]
                print(f"  Discards : {disc_str}")
                print("-" * 10)
            print("="*60)
        except Exception as e:
            print(f"Debug Print Error: {e}")
            traceback.print_exc()

    def _format_action(self, action):
        if not action: return "No Action"
        try:
            type_str = str(action.type)
            for k, v in vars(const).items():
                if v == action.type and k.isupper():
                    type_str = k
                    break
            t_str = get_tile_name(action.tile34) if action.tile34 != -1 else ""
            return f"[{type_str}] {t_str}"
        except:
            return str(action)

    def process_draw(self, ui_state):
        if not self.ai: return "模擬: 切 1m"
        
        # 修正：不再限制必須 14 張，而是滿足 3n+2 (2, 5, 8, 11, 14) 即可
        hand_len = len(ui_state.players[0].hand)
        if hand_len % 3 != 2: 
            return f"錯誤: 手牌數 {hand_len} 不正確 (應為 2, 5, 8, 11, 14 張)"

        try:
            draw_tile = ui_state.players[0].hand[-1]
            real_state = self._sync_to_real_state(ui_state, exclude_last_hand_tile=True)
            action = self.ai.process_draw(real_state, draw_tile, RL_flag=False)
            return self._format_action(action)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"

    def process_discard(self, ui_state):
        if not self.ai: return "模擬: Pass"
        if len(ui_state.players[0].hand) % 3 != 1: return "警告: 手牌數錯誤"
        try:
            real_state = self._sync_to_real_state(ui_state)
            action = self.ai.process_discard(real_state, current_player=-1)
            return self._format_action(action)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"

    def just_discard(self, ui_state):
        if not self.ai: return "模擬: 切牌"
        try:
            real_state = self._sync_to_real_state(ui_state)
            action = self.ai.just_discard(real_state, RL_flag=False)
            return self._format_action(action)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"

class MahjongGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("四人麻將盤面編輯器 v3.2 - 直觀切換版")
        self.clock = pygame.time.Clock()
        
        # 字體
        possible_fonts = ["Microsoft JhengHei", "SimHei", "Arial", "sans-serif"]
        self.font = pygame.font.Font(None, FONT_SIZE)
        for f in possible_fonts:
            try:
                self.font = pygame.font.SysFont(f, FONT_SIZE)
                break
            except: continue
        self.small_font = pygame.font.SysFont(possible_fonts[0], 16) if possible_fonts else pygame.font.Font(None, 16)

        self.game_state = UIGameState()
        self.ai = AIWrapper()
        
        self.selected_palette_tile_str = "1m"
        self.staging_area_tiles = []
        self.ai_message = "請配置盤面..."
        self.edit_mode = "STAGING" 
        
        self.score_input_focus = -1
        self.input_buffer = ""

        # --- UI 佈局 ---
        self.panel_width = 400
        game_w = SCREEN_WIDTH - self.panel_width
        
        self.zone_p2 = pygame.Rect(150, 10, game_w - 300, 160)
        self.zone_p3 = pygame.Rect(10, 180, 240, 450)
        self.zone_p1 = pygame.Rect(game_w - 250, 180, 240, 450)
        self.zone_p0 = pygame.Rect(150, 640, game_w - 300, 250)
        self.zone_center = pygame.Rect(260, 180, game_w - 520, 450)

        # Buttons
        base_y = 700
        self.btn_draw = pygame.Rect(SCREEN_WIDTH - 380, base_y, 110, 40)
        self.btn_resp = pygame.Rect(SCREEN_WIDTH - 260, base_y, 110, 40)
        self.btn_just = pygame.Rect(SCREEN_WIDTH - 140, base_y, 110, 40)
        self.btn_clear_p = pygame.Rect(SCREEN_WIDTH - 380, base_y + 50, 150, 40)
        
        self.meld_btns = []
        meld_types = [
            ("吃 (Chi)", const.CHOW), ("碰 (Pon)", const.PONG), 
            ("明槓 (Kan)", const.KONG), ("暗槓 (AnKan)", const.CONCEALED_KONG),
            ("加槓 (Add)", const.ADD_KONG)
        ]
        start_my = 550
        for i, (label, mtype) in enumerate(meld_types):
            rect = pygame.Rect(SCREEN_WIDTH - 380 + (i%2)*190, start_my + (i//2)*45, 180, 35)
            self.meld_btns.append({"rect": rect, "label": label, "type": mtype})
            
        self.btn_clear_staging = pygame.Rect(SCREEN_WIDTH - 380, start_my - 50, 100, 30)
        self.btn_add_discard = pygame.Rect(SCREEN_WIDTH - 200, start_my - 50, 150, 30)

        self.player_tabs = []
        for i in range(4):
            rect = pygame.Rect(SCREEN_WIDTH - self.panel_width + i*100, 0, 100, 40)
            self.player_tabs.append(rect)

        self.btn_edit_dora = pygame.Rect(0,0,0,0) 
        self.wind_btns = [] 
        self.riichi_btns = [] # 立直按鈕
        self.score_rects = []

    # --- 繪圖輔助 ---
    def draw_tile(self, tile34, x, y, is_selected=False, scale=1.0):
        w = int(TILE_WIDTH * scale)
        h = int(TILE_HEIGHT * scale)
        rect = pygame.Rect(x, y, w, h)
        color = (240, 240, 230) if not is_selected else (255, 200, 200)
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
        name = get_tile_name(tile34)
        tile_str = get_tile_str(tile34)
        text_color = (0,0,0)
        if 'm' in tile_str: text_color = (150, 0, 0)
        elif 's' in tile_str: text_color = (0, 100, 0)
        elif 'p' in tile_str: text_color = (0, 0, 150)
        if name in ['中', '白', '發']: 
            if name == '中': text_color = (200, 0, 0)
            elif name == '發': text_color = (0, 150, 0)
        font = self.font if scale > 0.8 else self.small_font
        font_surf = font.render(name, True, text_color)
        font_rect = font_surf.get_rect(center=rect.center)
        self.screen.blit(font_surf, font_rect)
        return rect

    # --- 繪製中央全域資訊區 ---
    def draw_center_info(self):
        rect = self.zone_center
        pygame.draw.rect(self.screen, (40, 45, 40), rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 150, 100), rect, 2)
        
        cx, cy = rect.x + 20, rect.y + 20
        
        # 0. Current Player
        curr_idx = self.game_state.current_editing_player
        cp_text = self.font.render(f"當前玩家 (Current): P{curr_idx}", True, (100, 255, 100))
        self.screen.blit(cp_text, (cx, cy))
        
        # 1. 寶牌 (Dora)
        cy += 30
        title = self.font.render("寶牌 (Dora):", True, (255, 200, 100))
        self.screen.blit(title, (cx, cy))
        dy = cy + 25
        dx = cx
        for i, t in enumerate(self.game_state.dora_indicators):
            tr = self.draw_tile(t, dx, dy)
            if pygame.mouse.get_pressed()[2] and tr.collidepoint(pygame.mouse.get_pos()):
                self.game_state.remove_last_dora()
                pygame.time.delay(150)
            dx += TILE_WIDTH + 5
        
        # 編輯模式按鈕
        self.btn_edit_dora = pygame.Rect(rect.right - 140, cy, 120, 30)
        bg_c = (200, 50, 50) if self.edit_mode == "DORA" else (80, 80, 80)
        txt = "模式: 寶牌" if self.edit_mode == "DORA" else "模式: 暫存"
        self.draw_btn(self.btn_edit_dora, txt, bg_c)

        # 2. 自風
        wy = dy + TILE_HEIGHT + 20
        w_title = self.font.render("自風 (My Wind):", True, (200, 200, 255))
        self.screen.blit(w_title, (cx, wy))
        wx = cx + 150
        winds = ["東", "南", "西", "北"]
        self.wind_btns = []
        for i, w_name in enumerate(winds):
            btn_rect = pygame.Rect(wx + i*60, wy - 5, 50, 30)
            is_active = (self.game_state.my_wind == i)
            c = (0, 120, 255) if is_active else (60, 60, 60)
            self.draw_btn(btn_rect, w_name, c)
            self.wind_btns.append({"rect": btn_rect, "val": i})

        # 3. 立直狀態 (Riichi)
        ry = wy + 40
        r_title = self.font.render("立直 (Riichi):", True, (255, 100, 100))
        self.screen.blit(r_title, (cx, ry))
        rx = cx + 150
        self.riichi_btns = []
        for i in range(4):
            btn_rect = pygame.Rect(rx + i*60, ry - 5, 50, 30)
            is_riichi = self.game_state.riichi_info[i]
            c = (220, 50, 50) if is_riichi else (60, 60, 60)
            label = f"P{i}"
            self.draw_btn(btn_rect, label, c)
            self.riichi_btns.append({"rect": btn_rect, "idx": i})

        # 4. 分數
        sy = ry + 50
        pygame.draw.line(self.screen, (80,100,80), (rect.x+10, sy-10), (rect.right-10, sy-10), 2)
        s_title = self.font.render("分數 (Scores):", True, (255, 200, 100))
        self.screen.blit(s_title, (cx, sy))
        
        self.score_rects = []
        for i in range(4):
            val_str = str(self.game_state.scores[i])
            if self.score_input_focus == i:
                val_str = self.input_buffer + "_"
                color = (255, 50, 50)
            else:
                color = (255, 255, 255)
            
            label = f"P{i}: {val_str}"
            s_surf = self.font.render(label, True, color)
            sx = cx + (i % 2) * 250
            score_y = sy + 30 + (i // 2) * 30
            s_rect = s_surf.get_rect(topleft=(sx, score_y))
            self.screen.blit(s_surf, s_rect)
            
            hit_rect = s_rect.inflate(20, 10)
            self.score_rects.append({"rect": hit_rect, "idx": i})

    def draw_player_zone(self, player_idx):
        p = self.game_state.players[player_idx]
        
        if player_idx == 0: 
            rect = self.zone_p0
            label = "P0 (本家/Self)"
        elif player_idx == 1: 
            rect = self.zone_p1
            label = "P1 (下家/Right)"
        elif player_idx == 2: 
            rect = self.zone_p2
            label = "P2 (對家/Top)"
        else: 
            rect = self.zone_p3
            label = "P3 (上家/Left)"
            
        is_editing = (self.game_state.current_editing_player == player_idx)
        border_color = (255, 200, 0) if is_editing else (80, 80, 80)
        bg_color = (35, 45, 35) if is_editing else (30, 40, 30)
        pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, rect, 2)
        title_surf = self.font.render(label, True, (180, 180, 180))
        self.screen.blit(title_surf, (rect.x + 10, rect.y + 5))

        cx, cy = rect.x + 15, rect.y + 30
        
        # 1. Melds
        for m_idx, m in enumerate(p.meld):
            meld_w = len(m.tiles34) * (TILE_WIDTH * 0.7 + 1)
            meld_rect = pygame.Rect(cx, cy, meld_w, TILE_HEIGHT * 0.7)
            
            if is_editing and pygame.mouse.get_pressed()[2] and meld_rect.collidepoint(pygame.mouse.get_pos()):
                self.game_state.remove_meld(player_idx, m_idx)
                pygame.time.delay(150)
                return 

            for t in m.tiles34:
                self.draw_tile(t, cx, cy, scale=0.7)
                cx += int(TILE_WIDTH * 0.7) + 1
            cx += 8 
        
        # 2. Discards
        is_vertical_zone = (player_idx in [1, 3])
        start_dx = rect.x + 15
        dx, dy = start_dx, cy + int(TILE_HEIGHT*0.7) + 15
        for i, t in enumerate(p.discards):
            tr = self.draw_tile(t, dx, dy, scale=0.8)
            if is_editing and pygame.mouse.get_pressed()[2] and tr.collidepoint(pygame.mouse.get_pos()):
                self.game_state.remove_last_discard(player_idx)
                pygame.time.delay(150)
            dx += int(TILE_WIDTH * 0.8) + 2
            limit = 200 if is_vertical_zone else 800
            if (dx - start_dx) > limit:
                dx = start_dx; dy += int(TILE_HEIGHT * 0.8) + 2

        # 3. Hand (All Players logic, but usually P0)
        # P0 固定顯示在下方，其他玩家如果要顯示也可以，但空間有限
        if player_idx == 0:
            hand_y = rect.bottom - TILE_HEIGHT - 20
            hand_x = rect.x + 50
            lbl = self.font.render(f"手牌 ({len(p.hand)})", True, (255, 255, 200))
            self.screen.blit(lbl, (rect.x + 50, hand_y - 25))
            
            for i, t in enumerate(p.hand):
                tr = self.draw_tile(t, hand_x, hand_y)
                if is_editing and pygame.mouse.get_pressed()[2] and tr.collidepoint(pygame.mouse.get_pos()):
                    self.game_state.remove_tile_from_hand(0, i)
                    pygame.time.delay(150)
                if len(p.hand) == 14 and i == 12: hand_x += 15 
                hand_x += TILE_WIDTH + 2
            
            if len(p.hand) < 14 and is_editing:
                ghost = pygame.Rect(hand_x, hand_y, TILE_WIDTH, TILE_HEIGHT)
                pygame.draw.rect(self.screen, (100, 150, 100), ghost, 2)
                if pygame.mouse.get_pressed()[0] and ghost.collidepoint(pygame.mouse.get_pos()):
                    if self.edit_mode == "STAGING": 
                        t_int = STRING_TO_34.get(self.selected_palette_tile_str, 0)
                        self.game_state.add_tile_to_hand(0, t_int)
                        pygame.time.delay(150)

    def draw_control_panel(self):
        panel_rect = pygame.Rect(SCREEN_WIDTH - self.panel_width, 0, self.panel_width, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect)
        
        for i, rect in enumerate(self.player_tabs):
            c = (0, 120, 215) if self.game_state.current_editing_player == i else (80, 80, 80)
            self.draw_btn(rect, f"P{i}", c)

        start_x, start_y = SCREEN_WIDTH - self.panel_width + 20, 60
        x, y = start_x, start_y
        for i, t_str in enumerate(TILES):
            is_sel = (t_str == self.selected_palette_tile_str)
            t_int = STRING_TO_34[t_str]
            tr = self.draw_tile(t_int, x, y, is_sel)
            
            if pygame.mouse.get_pressed()[0] and tr.collidepoint(pygame.mouse.get_pos()):
                self.selected_palette_tile_str = t_str
                if self.edit_mode == "STAGING":
                    self.staging_area_tiles.append(t_int)
                elif self.edit_mode == "DORA":
                    self.game_state.add_dora(t_int)
                pygame.time.delay(150)
            x += TILE_WIDTH + 5
            if (i+1) % 9 == 0: x = start_x; y += TILE_HEIGHT + 5

        sy = 420
        pygame.draw.line(self.screen, (150,150,150), (SCREEN_WIDTH - self.panel_width, sy), (SCREEN_WIDTH, sy), 2)
        lbl = self.font.render("暫存區 (點選牌庫加入):", True, (255, 200, 100))
        self.screen.blit(lbl, (start_x, sy + 10))
        sx = start_x
        for idx, st_tile in enumerate(self.staging_area_tiles):
            sr = self.draw_tile(st_tile, sx, sy + 40)
            if pygame.mouse.get_pressed()[0] and sr.collidepoint(pygame.mouse.get_pos()):
                self.staging_area_tiles.pop(idx)
                pygame.time.delay(150)
            sx += TILE_WIDTH + 5
            if sx > SCREEN_WIDTH - 20: sx = start_x; sy += 60

        self.draw_btn(self.btn_clear_staging, "清空暫存", (150, 50, 50))
        self.draw_btn(self.btn_add_discard, "加入棄牌", (50, 150, 100))
        for btn in self.meld_btns:
            self.draw_btn(btn["rect"], btn["label"], (100, 100, 150))

        pygame.draw.line(self.screen, (150,150,150), (SCREEN_WIDTH - self.panel_width, 680), (SCREEN_WIDTH, 680), 2)
        self.draw_btn(self.btn_draw, "摸牌決策", (0, 120, 200))
        self.draw_btn(self.btn_resp, "回應打牌", (200, 120, 0))
        self.draw_btn(self.btn_just, "副露切牌", (100, 180, 0))
        self.draw_btn(self.btn_clear_p, "清空玩家", (200, 50, 50))
        msg = self.font.render(self.ai_message, True, (255, 255, 255))
        self.screen.blit(msg, (SCREEN_WIDTH - 380, 800))

    def draw_btn(self, rect, text, color):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        c = (min(color[0]+30, 255), min(color[1]+30, 255), min(color[2]+30, 255)) if hover else color
        pygame.draw.rect(self.screen, c, rect, border_radius=5)
        txt_surf = self.font.render(text, True, (255, 255, 255))
        tr = txt_surf.get_rect(center=rect.center)
        self.screen.blit(txt_surf, tr)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            
            if event.type == pygame.KEYDOWN and self.score_input_focus != -1:
                if event.key == pygame.K_BACKSPACE:
                    self.input_buffer = self.input_buffer[:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    if self.input_buffer.isdigit():
                        self.game_state.update_score(self.score_input_focus, int(self.input_buffer))
                    self.score_input_focus = -1
                    self.input_buffer = ""
                elif event.unicode.isdigit():
                    self.input_buffer += event.unicode
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = event.pos
                    
                    clicked_score = False
                    for item in self.score_rects:
                        if item["rect"].collidepoint(pos):
                            self.score_input_focus = item["idx"]
                            self.input_buffer = str(self.game_state.scores[item["idx"]])
                            clicked_score = True
                    if not clicked_score:
                        if self.score_input_focus != -1:
                            if self.input_buffer.isdigit():
                                self.game_state.update_score(self.score_input_focus, int(self.input_buffer))
                            self.score_input_focus = -1
                    
                    # 點擊盤面直接切換玩家
                    if self.zone_p0.collidepoint(pos): self.game_state.switch_player(0)
                    elif self.zone_p1.collidepoint(pos): self.game_state.switch_player(1)
                    elif self.zone_p2.collidepoint(pos): self.game_state.switch_player(2)
                    elif self.zone_p3.collidepoint(pos): self.game_state.switch_player(3)

                    # Player Tabs (切換玩家)
                    for i, rect in enumerate(self.player_tabs):
                        if rect.collidepoint(pos):
                            self.game_state.switch_player(i) # Explicit switch
                    
                    if self.btn_edit_dora.collidepoint(pos):
                        if self.edit_mode == "STAGING": self.edit_mode = "DORA"
                        else: self.edit_mode = "STAGING"
                    
                    for btn in self.wind_btns:
                        if btn["rect"].collidepoint(pos):
                            self.game_state.set_wind(btn["val"])

                    # Riichi Btns
                    for btn in self.riichi_btns:
                        if btn["rect"].collidepoint(pos):
                            self.game_state.toggle_riichi(btn["idx"])

                    if self.btn_clear_staging.collidepoint(pos):
                        self.staging_area_tiles = []
                    if self.btn_add_discard.collidepoint(pos):
                        curr = self.game_state.current_editing_player
                        for t in self.staging_area_tiles:
                            self.game_state.add_discard(curr, t)
                        self.staging_area_tiles = []
                    
                    curr_p = self.game_state.current_editing_player
                    for btn in self.meld_btns:
                        if btn["rect"].collidepoint(pos):
                            if not self.staging_area_tiles:
                                self.ai_message = "錯誤: 暫存區無牌"
                            else:
                                new_meld = Meld(btn["type"], copy.copy(self.staging_area_tiles))
                                self.game_state.add_meld(curr_p, new_meld)
                                self.staging_area_tiles = []
                                self.ai_message = f"已新增 {btn['label']}"
                    
                    if self.btn_clear_p.collidepoint(pos):
                        p = self.game_state.players[curr_p]
                        p.hand = []; p.discards = []; p.meld = []
                        self.game_state.mark_dirty()
                    
                    if self.btn_draw.collidepoint(pos):
                        self.ai_message = "計算中..."
                        self.draw_all(); pygame.display.flip()
                        self.ai_message = self.ai.process_draw(self.game_state)
                        
                    if self.btn_resp.collidepoint(pos):
                        self.ai_message = "計算中..."
                        self.draw_all(); pygame.display.flip()
                        self.ai_message = self.ai.process_discard(self.game_state)
                        
                    if self.btn_just.collidepoint(pos):
                        self.ai_message = "計算中..."
                        self.draw_all(); pygame.display.flip()
                        self.ai_message = self.ai.just_discard(self.game_state)

    def draw_all(self):
        self.screen.fill(BG_COLOR)
        self.draw_center_info()
        for i in range(4):
            self.draw_player_zone(i)
        self.draw_control_panel()
        
        if self.game_state.is_dirty:
            self.ai.debug_print_state(self.game_state)
            self.game_state.is_dirty = False

    def run(self):
        while True:
            self.handle_events()
            self.draw_all()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    app = MahjongGUI()
    app.run()