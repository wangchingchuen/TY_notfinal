"""
把「跑一次 YOLO 推論、拿到框」這件事包成物件。
TrashDetector 用第一階段的 v4_last.pt，PersonVehicleDetector 用 yolo11n.pt，
兩者共用同一個 BaseDetector，介面完全一致，之後要換模型只要改 config，不用動邏輯。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO


@dataclass
class Detection:
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class BaseDetector:
    def __init__(self, weights: str, device: str, conf: float, imgsz: int,
                 classes: Optional[List[int]] = None):
        print(f"🧠 載入模型: {weights}")
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes

    def detect(self, image_path: Path) -> List[Detection]:
        results = self.model.predict(
            source=str(image_path), conf=self.conf, imgsz=self.imgsz,
            device=self.device, classes=self.classes, verbose=False,
        )
        boxes = results[0].boxes
        return [
            Detection(
                cls_id=int(box.cls[0]),
                conf=float(box.conf[0]),
                x1=float(box.xyxy[0][0]), y1=float(box.xyxy[0][1]),
                x2=float(box.xyxy[0][2]), y2=float(box.xyxy[0][3]),
            )
            for box in boxes
        ]


class TrashDetector(BaseDetector):
    """第一階段垃圾偵測模型，不限定類別（本身就只有 trash 一類）。"""
    def __init__(self, weights: str, device: str, conf: float, imgsz: int):
        super().__init__(weights, device, conf, imgsz, classes=None)


class PersonVehicleDetector(BaseDetector):
    """yolo11n，只抓 COCO 的 person/car/motorcycle/bus/truck。"""
    def __init__(self, weights: str, device: str, conf: float, imgsz: int,
                 classes: List[int]):
        super().__init__(weights, device, conf, imgsz, classes=classes)
