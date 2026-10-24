"""
主流程物件：組合 TrashDetector + PersonVehicleDetector + LitteringMatcher，
對 images_root 底下每個「影片資料夾」逐一處理，最後彙整成一份 DataFrame。

資料結構要求（跟你現在的 images/ 完全一樣，不用重新整理）：
    images_root/
      <任意子路徑，例如 train/test/validation>/
        <影片名稱資料夾>/
          frame_000001.png
          frame_000002.png
          ...
"""
import re
from pathlib import Path
from typing import List

import pandas as pd

from config import Config
from detectors import TrashDetector, PersonVehicleDetector
from matcher import LitteringMatcher, LitteringEvent


def extract_frame_num(filename: str) -> int:
    m = re.search(r"(\d+)", Path(filename).stem)
    return int(m.group(1)) if m else -1


class LitteringPipeline:
    def __init__(self, config: Config):
        self.cfg = config
        self.trash_detector = TrashDetector(
            weights=config.trash_weights, device=config.device,
            conf=config.conf_trash, imgsz=config.imgsz,
        )
        self.pv_detector = PersonVehicleDetector(
            weights=config.person_vehicle_weights, device=config.device,
            conf=config.conf_person_vehicle, imgsz=config.imgsz,
            classes=config.person_vehicle_classes,
        )#trash and person是parralell嗎
        self.matcher = LitteringMatcher(config)

    def _find_video_dirs(self, images_root: Path) -> List[Path]:
        """
        找出所有「底下直接是圖片」的資料夾，當作一支影片。
        不管圖片是在 images_root/train/影片A/ 還是 images_root/影片A/，都能找到。
        """
        video_dirs = []
        for d in images_root.rglob("*"):
            if not d.is_dir():
                continue
            has_images = any(
                f.suffix.lower() in {".png", ".jpg", ".jpeg"}
                for f in d.iterdir() if f.is_file()
            )
            if has_images:
                video_dirs.append(d)
        return sorted(video_dirs)

    def run_on_video(self, video_dir: Path) -> List[LitteringEvent]:
        video_name = video_dir.name
        self.matcher.reset()

        image_files = sorted(
            (f for f in video_dir.iterdir()
             if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}),
            key=lambda f: extract_frame_num(f.name),
        )

        for img_path in image_files:
            frame_num = extract_frame_num(img_path.name)
            trash_boxes = self.trash_detector.detect(img_path)
            pv_boxes = self.pv_detector.detect(img_path)
            self.matcher.process_frame(video_name, frame_num, img_path.name, trash_boxes, pv_boxes)

        n_events = len(self.matcher.events)
        print(f"▶️ {video_name}：{len(image_files)} 張圖，發現 {n_events} 起疑似事件")
        return list(self.matcher.events)

    def run_on_all(self, images_root: Path) -> pd.DataFrame:
        video_dirs = self._find_video_dirs(images_root)
        print(f"📂 共找到 {len(video_dirs)} 支影片資料夾\n")

        all_events: List[LitteringEvent] = []
        for video_dir in video_dirs:
            all_events.extend(self.run_on_video(video_dir))

        rows = [{
            "影片": e.video_name,
            "圖片檔名": e.image_filename,
            "Frame": e.frame_num,
            "垃圾框_左上X": e.trash_box[0], "垃圾框_左上Y": e.trash_box[1],
            "垃圾框_右下X": e.trash_box[2], "垃圾框_右下Y": e.trash_box[3],
            "人車類別ID": e.person_vehicle_cls,
            "人車框_左上X": e.person_vehicle_box[0], "人車框_左上Y": e.person_vehicle_box[1],
            "人車框_右下X": e.person_vehicle_box[2], "人車框_右下Y": e.person_vehicle_box[3],
            "距離": e.distance,
        } for e in all_events]

        return pd.DataFrame(rows)
