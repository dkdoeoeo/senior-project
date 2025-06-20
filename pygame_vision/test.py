from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.shanten import Shanten

shanten = Shanten()
tiles = TilesConverter.string_to_34_array(man='13569', pin='123459', sou='443')
result = shanten.calculate_shanten(tiles)

print(result)