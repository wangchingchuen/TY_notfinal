# 第二階段（物件導向簡化版）

## 資料結構

```
images/                        ← 跟你現在的 images 資料夾結構完全一樣，不用重整理
  train/ 或 test/ 或 validation/（有沒有這層都可以）
    <影片名稱資料夾>/
      frame_000001.png
      frame_000002.png
      ...
```

`pipeline.py` 會自動遞迴找出所有「底下直接放圖片」的資料夾，當作一支影片來處理，
不管中間有沒有 train/test/validation 這層，都不用管。

## 程式結構

```
config.py      設定集中管理（模型路徑、參數），要調參數只改這一個檔案
detectors.py   Detection / TrashDetector / PersonVehicleDetector
                封裝「跑一次 YOLO 推論拿到框」這件事，兩個偵測器介面一致
matcher.py     LitteringEvent / LitteringMatcher
                簡化版事件判定邏輯（4 個參數，取代原本 6 個）
pipeline.py    LitteringPipeline
                串起偵測器 + 判定器，逐支影片、逐幀跑過去，彙整成表格
main.py        入口：讀設定、跑 pipeline、輸出 CSV
```

## 用法

```bash
python3 main.py
```

只需要跑這一支

可以用參數覆寫預設設定：
```bash
python3 main.py --images-root /path/to/images --output-csv my_result.csv --device cpu
```

## 跟原本架構的差異

| | 原本 | 這版 |
|---|---|---|
| 垃圾偵測 | `test_yolo11_find_trash.py` | `TrashDetector` 類別，邏輯不變、模型不變 |
| 人車偵測 | `test_yolo11n_find_car_and_person.py` | `PersonVehicleDetector` 類別，邏輯不變、模型不變 |
| 事件判定參數 | 6 個（THRESHOLD_PIXELS、PATIENCE_FRAMES、MERGE_RADIUS、EPS_MOVE、N_STATIC、N_CANDIDATE） | 4 個（match_radius_px、new_event_window、new_event_radius、min_persist_frames），都在 `config.py` |
| 跨幀追蹤 | 有 ID 系統、狀態機（candidate/active/static） | 沒有，只用「這個位置最近有沒有出現過」判斷是不是新事件 |
| 執行方式 | 手動跑 3～4 支腳本，中間手動改檔名 | 一支 `main.py` 跑到底 |
| 輸出 | `real_trash_results.csv` + `detect_results.csv` + `suspected_littering_with_tracker.csv`，分散在好幾個檔案 | 一份 `suspected_littering_events.csv`，每一列就是一起事件，含垃圾框、人車類別、人車框、距離 |

