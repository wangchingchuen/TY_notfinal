"""
簡化版事件判定器，取代原本 final_check_new_correct.py 裡一整套跨幀 ID 追蹤系統。

不做的事（刻意拿掉，換取簡單）：
  - 不建立垃圾的追蹤 ID
  - 不判斷垃圾「有沒有在動」（EPS_MOVE / N_STATIC / N_CANDIDATE 全部拿掉）
  - 不用 PATIENCE_FRAMES 容忍垃圾消失又出現算同一個

只做的事：
  1. 一個垃圾框，如果它出現的位置，在最近 new_event_window 幀內、
     以及之前所有已確認事件的位置附近 new_event_radius 內都沒出現過，
     視為「新位置候選」。
  2. 候選要連續出現 >= min_persist_frames 幀（用簡單的網格座標近似比對，
     不建追蹤 ID），才算是穩定事件，不是 YOLO 單幀誤判。
  3. 事件確認的那一幀，在 match_radius_px 內找距離最近的人/車，記為嫌疑對象。

因此只有 4 個參數（原本 6 個），且每個參數的意義都很直觀，好調、好解釋。
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from config import Config
from detectors import Detection


@dataclass
class LitteringEvent: #要有time
    video_name: str
    frame_num: int
    image_filename: str
    trash_box: Tuple[float, float, float, float]
    person_vehicle_cls: int
    person_vehicle_box: Tuple[float, float, float, float]
    distance: float


class LitteringMatcher:
    def __init__(self, config: Config):
        self.cfg = config
        self._history: List[List[Tuple[float, float]]] = []
        self._pending: Dict[Tuple[int, int], int] = {}
        self._confirmed_positions: List[Tuple[float, float]] = []
        self.events: List[LitteringEvent] = []

    def reset(self):
        """換一支新影片時呼叫，清空所有狀態，事件之間互不影響。"""
        self._history.clear()
        self._pending.clear()
        self._confirmed_positions.clear()

    @staticmethod
    def _grid_key(cx: float, cy: float, cell: float = 20.0) -> Tuple[int, int]:
        return (round(cx / cell), round(cy / cell))

    def _is_new_position(self, cx: float, cy: float) -> bool:
        r = self.cfg.new_event_radius
        for frame_centers in self._history[-self.cfg.new_event_window:]:
            for hx, hy in frame_centers:
                if math.hypot(cx - hx, cy - hy) < r:
                    return False
        for cx0, cy0 in self._confirmed_positions:
            if math.hypot(cx - cx0, cy - cy0) < r:
                return False
        return True

    def process_frame(self, video_name: str, frame_num: int, image_filename: str,
                       trash_boxes: List[Detection], person_vehicle_boxes: List[Detection]):
        current_centers = []

        for box in trash_boxes:
            cx, cy = box.center
            current_centers.append((cx, cy))

            if not self._is_new_position(cx, cy):
                continue  # 背景垃圾，早就在那了，不累積候選

            key = self._grid_key(cx, cy)
            self._pending[key] = self._pending.get(key, 0) + 1

            if self._pending[key] < self.cfg.min_persist_frames:
                continue  # 還沒連續出現夠久，可能是雜訊

            # 連續出現夠久了，確認為事件：找當下最近的人車
            nearest, nearest_dist = None, float("inf")
            for pv in person_vehicle_boxes:
                pvx, pvy = pv.center
                d = math.hypot(cx - pvx, cy - pvy)
                if d < nearest_dist:
                    nearest, nearest_dist = pv, d

            if nearest is not None and nearest_dist <= self.cfg.match_radius_px:
                self.events.append(LitteringEvent(
                    video_name=video_name,
                    frame_num=frame_num,
                    image_filename=image_filename,
                    trash_box=(box.x1, box.y1, box.x2, box.y2),
                    person_vehicle_cls=nearest.cls_id,
                    person_vehicle_box=(nearest.x1, nearest.y1, nearest.x2, nearest.y2),
                    distance=round(nearest_dist, 2),
                ))

            self._confirmed_positions.append((cx, cy))
            del self._pending[key]

        self._history.append(current_centers)
        if len(self._history) > self.cfg.new_event_window:
            self._history.pop(0)
