"""
所有設定集中在這一個檔案，其他模組都從這裡讀參數，不再散落在各支腳本裡各自寫死。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- 模型 ----
    trash_weights: str = "/mnt/backups/weihong/handoff_20260820_training/weights/v4_last.pt"
    person_vehicle_weights: str = "yolo11n.pt"  # COCO 預訓練，自動下載，不用重練
    device: str = "0"          # GPU index，CPU 環境改成 "cpu"
    imgsz: int = 1280 #待測試
    conf_trash: float = 0.25
    conf_person_vehicle: float = 0.25
    # COCO 類別：0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
    person_vehicle_classes: List[int] = field(default_factory=lambda: [0, 2, 3, 5, 7])#field addr 
    """lamda=makes_classes
       def make_classes():
           return [0, 2, 3, 5, 7]
    """
    

    # ---- 輸入/輸出 ----
    images_root: str = "/home/wangcc/Taoyuan_littering_Detection_repo/images"
    output_csv: str = "suspected_littering_events.csv"

    # ---- 事件判定（簡化版，只有 4 個參數）----
    match_radius_px: float = 450.0     # 垃圾出現的那一刻，找最近人車的最大容許距離
    new_event_window: int = 15         # 往前看幾幀，判斷這個位置是不是「新出現」
    new_event_radius: float = 30.0     # 在這個半徑內，前面幀有出現過就不算新的（背景垃圾）
    min_persist_frames: int = 3        # 候選位置要連續出現幾幀，才算是穩定事件（濾掉 YOLO 閃爍誤判）

"""
(100, 200), (101, 202)
"""