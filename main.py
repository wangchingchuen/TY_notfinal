"""
第二階段主程式（物件導向簡化版）。

用法：
    python3 main.py
    python3 main.py --images-root /path/to/images --output-csv result.csv
    python3 main.py --device cpu

只需要跑這一支，不用像以前一樣手動一支一支腳本接力執行。
"""
import argparse #可以用 argparse 來解析命令列參數
from pathlib import Path #把圖片資料夾路徑轉成 Path 物件，方便找檔案、組合路徑。

from config import Config #從專案自己的 config.py 匯入 Config 類別。
from pipeline import LitteringPipeline #從專案自己的 pipeline.py 匯入 LitteringPipeline 類別。


def main():
    parser = argparse.ArgumentParser(description="第二階段：亂丟垃圾事件偵測") #用來保存「命令列參數解析器物件」。ArgumentParser: argparse 裡的一個類別，用來建立命令列參數解析器。description顯示when --help or -h
    parser.add_argument("--images-root", default=None, help="覆寫 config.py 裡的 images_root") #允許使用者指定圖片資料夾
    parser.add_argument("--output-csv", default=None, help="覆寫 config.py 裡的 output_csv")
    parser.add_argument("--device", default=None, help="覆寫 config.py 裡的 device，例如 cpu 或 0")
    parser.add_argument("--trash-weights", default=None, help="覆寫第一階段垃圾模型權重路徑")
    args = parser.parse_args()

    cfg = Config() #建立設定物件，取得模型路徑、圖片路徑、裝置等設定。 為何參數不直接放在()裡
    if args.images_root:
        cfg.images_root = args.images_root
    if args.output_csv:
        cfg.output_csv = args.output_csv
    if args.device:
        cfg.device = args.device
    if args.trash_weights:
        cfg.trash_weights = args.trash_weights

    print("=" * 70)
    print("🚀 啟動第二階段：亂丟垃圾事件偵測")
    print("=" * 70)
    print(f"垃圾模型: {cfg.trash_weights}")
    print(f"人車模型: {cfg.person_vehicle_weights}")
    print(f"圖片根目錄: {cfg.images_root}")
    print(f"裝置: {cfg.device}\n")

    pipeline = LitteringPipeline(cfg)
    df = pipeline.run_on_all(Path(cfg.images_root)) #應該直接從cfg取

    df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成！共 {len(df)} 起疑似違規事件，已存至 {cfg.output_csv}")


if __name__ == "__main__":
    main()
    #是不是同一個垃圾，不要管狀態
