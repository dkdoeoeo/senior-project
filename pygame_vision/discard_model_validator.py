import torch
import torch.nn.functional as F
import threading
import time
import csv
import time
import torch.nn as nn
from model_define import MyDataset
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
from datetime import datetime
import os
from model_define import DiscardCNN

class Discard_model_validator(threading.Thread):
    def __init__(self, model_path, best_model_path='E:/專題/discard_model/RL/best_model.pth', duration = 3600, device="cuda", log_file='E:/專題/模型訓練過程/RL_validation_log.csv'):
        """
        model_path: 要驗證的模型路徑
        validation_file: 驗證資料集路徑
        best_model_path: 儲存最佳模型的檔案路徑
        duration: 驗證時間 (秒)，預設 1 小時
        """
        super().__init__()
        self.model_path = model_path
        self.best_model_path = best_model_path
        self.duration = duration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_file = log_file
        self.loss_criterion = nn.CrossEntropyLoss()
        self.batch_size = 32
    
    def run(self):
        try:
            print("開始驗證")
            """
            讀取csv
            """
            self.dataset = MyDataset('E:/專題/data/2022/DiscardData.csv', chunk_size=50000)
            self.test_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

            model = torch.load(self.model_path)
            model.to(self.device)
            model.eval()

            correct = 0
            total = 0
            start_time = time.time()
            with torch.no_grad():
                while time.time() - start_time < self.duration:
                    for train_data, value_data in self.test_dataloader:
                        torch.autograd.set_detect_anomaly(True)
                        channel1_tensor = train_data.unsqueeze(1)
                        channel1_tensor = channel1_tensor.to(self.device)
                        value_data = value_data.to(self.device)
                        
                        print("channel1_tensor.shape:",channel1_tensor.shape)
                        output = model(channel1_tensor)
                        output = output.float()
                        value_data = value_data.long()
                        probabilities = F.softmax(output, dim=1)
                        predicted_labels = torch.argmax(probabilities, dim=1)
                        correct_predictions = (predicted_labels == value_data).sum().item()
                        accuracy = (correct_predictions / value_data.size(0))*100
                        
                        correct += correct_predictions
                        total += value_data.size(0)
                        totalAccuracy = (correct/total)*100
                        
                        loss = self.loss_criterion(output, value_data)
                        
                        current_loss = loss
                        
                        clear_output()

                        print(predicted_labels)
                        print(value_data)
                        print(f"Correct Predictions: {correct_predictions}")
                        print(f"Total Predictions: {value_data.size(0)}")
                        print(f"Cur Accuracy: {accuracy:.4f}","%")
                        print(f"Total Accuracy: {totalAccuracy:.4f}","%")
                        print(f"loss: {current_loss}")

                        if time.time() - start_time >= self.duration:
                            break
                    
                    accuracy = correct / total if total > 0 else 0.0
                    print(f"[Validator] 模型 {self.model_path} 1小時驗證準確率: {accuracy:.2%}")

            # 寫入紀錄檔
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.model_path, accuracy])

            # 檢查是否需要更新最佳模型
            best_acc = 43.3
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    for row in csv.reader(f):
                        try:
                            best_acc = max(best_acc, float(row[2]))
                        except:
                            continue

            if accuracy >= best_acc:
                torch.save(model, self.best_model_path)
                print(f"[Validator] 新最佳模型，準確率 {accuracy:.2%}，已更新 {self.best_model_path}")
        
        except Exception as e:
            print(f"驗證執行緒錯誤：{e}")


#測試區
if __name__ == "__main__":
    best_model_path = 'E:/專題/discard_model/RL/best_model.pth'
    discard_model_validator = Discard_model_validator(best_model_path, best_model_path = best_model_path)
    discard_model_validator.start()