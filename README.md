Discard model:
	
	data process:

		標籤值:1~152(牌局資訊)、153(label:棄牌值)
		
		時機:每次棄牌

		步驟:每次棄牌時根據抽到的玩家視角添加一筆資訊

chow model:

	data process:(新增一個flag記是否有紀錄的資訊)

		標籤值:1~152(牌局資訊)、153(上家棄牌)、154(label:是否吃)

		時機:每次可以吃的時候

		步驟:
			
			1.每次棄牌檢查下一位的手牌能不能吃

			2.如果可以就紀錄一筆資訊(只包含牌局資訊以及打出的牌，未含label值) record_chow_info

			3.等到下一輪如果是吃操作就label=1，否則label=0後寫入datalist record_chow_label 例外:槓、碰直接不計

			4.每次N或抽牌就重置flag record_chow_label
