"""
用 Claude API 標註垃圾，記錄每張的 token、費用、秒數，並對 GT 算分。

用法：
    pip install anthropic pillow
    export ANTHROPIC_API_KEY=sk-ant-...          # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
    python3 claude_annotate.py 圖片資料夾
    python3 claude_annotate.py 圖片資料夾 --effort high      # 調整思考量：low / medium（Opus 5.5 預設）/ high / xhigh / max
    python3 claude_annotate.py 圖片資料夾 --bedrock --region us-east-1   # 改走 Amazon Bedrock（用 AWS 憑證）

圖片資料夾放 7 張原圖，檔名用 GT 名稱（例如 八德ATW_菸蒂_在手上.png）才能自動對 GT 算分。
檔名不會送給模型（API 只送圖片內容），所以等同上次的 v3「無檔名」條件。

輸出（在 圖片資料夾/claude_out/）：
    boxes_claude.json   每張的框座標
    usage.csv           每張的 input/output token、費用、秒數
    <圖名>__claude.png   畫上框的圖
指定 --effort 時輸出資料夾改為 claude_out_<effort>，不會蓋掉之前的結果。
"""
import argparse
import base64
import csv
import json
import re
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

PRICE_IN, PRICE_OUT = 4.0, 20.0          # Opus 5.5，美元 / 百萬 token（Batch API 是一半）

PROMPT = """Find all litter in this image (cigarette butts, plastic bottles, cups, paper, bags, any other trash).
The image is {w}x{h} pixels. Give each box in pixel coordinates of this image, as tight as possible around the object.
Return JSON only, no other text: a list of {{"bbox": [x1, y1, x2, y2], "label": "<short name>"}}.
If there is no litter, return []."""

# GT：class 2（垃圾）的 YOLO 正規化框 cx cy w h
GT = {
    "八德ATW_菸蒂_在手上": (0.595062, 0.633104, 0.013875, 0.019903),
    "八德ATW_菸蒂_飛行中": (0.534375, 0.798264, 0.016406, 0.025139),
    "八德ATW_菸蒂_在地上": (0.489297, 0.966875, 0.016250, 0.018194),
    "大溪_20250518_寶特瓶_在飛": (0.455355, 0.558167, 0.025383, 0.062500),
    "大溪_20250518_寶特瓶_在草裡": (0.588508, 0.835660, 0.026391, 0.035069),
    "大溪_ch055_20250717_一般廢棄物_飛行中": (0.516742, 0.344757, 0.013828, 0.022292),
    "大溪20250605_寶特瓶_在飛": (0.428063, 0.420167, 0.013828, 0.019528),
}


def make_client(args):
    if args.bedrock:
        from anthropic import AnthropicBedrock
        return AnthropicBedrock(aws_region=args.region), args.model or "anthropic.claude-opus-5-5"
    from anthropic import Anthropic
    return Anthropic(), args.model or "claude-opus-5-5"


def ask(client, model, path, effort=None):
    img = Image.open(path)
    w, h = img.size
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.standard_b64encode(path.read_bytes()).decode()
    extra = {"output_config": {"effort": effort}} if effort else {}
    t0 = time.time()
    # 用串流接收：max_tokens 很大時，SDK 不允許非串流請求（會直接報錯）
    with client.messages.stream(
        model=model,
        max_tokens=4000 if effort in (None, "low", "medium") else 32000,   # 思考 token 也算在 max_tokens 裡
        extra_body=extra or None,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
            {"type": "text", "text": PROMPT.format(w=w, h=h)},
        ]}],
    ) as stream:
        resp = stream.get_final_message()
    sec = time.time() - t0
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage, sec, (w, h), resp.stop_reason


def parse_boxes(text, w, h):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        b = it.get("bbox") if isinstance(it, dict) else None
        if not (isinstance(b, list) and len(b) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        out.append([x1, y1, x2, y2, str(it.get("label", "trash"))])
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def gt_key(name):
    """檔名裡的空格當成底線，例如「八德ATW 菸蒂_在手上」→「八德ATW_菸蒂_在手上」。"""
    return re.sub(r"\s+", "_", name.strip())


def score(name, boxes, w, h):
    name = gt_key(name)
    if name not in GT:
        return None
    cx, cy, gw, gh = GT[name]
    g = ((cx - gw / 2) * w, (cy - gh / 2) * h, (cx + gw / 2) * w, (cy + gh / 2) * h)
    gx, gy, tol = cx * w, cy * h, max(gw * w, gh * h)
    best = max((iou(b, g) for b in boxes), default=0.0)
    loose = sum(1 for b in boxes if (((b[0] + b[2]) / 2 - gx) ** 2 + ((b[1] + b[3]) / 2 - gy) ** 2) ** 0.5 <= tol)
    return dict(iou=best, strict=best >= 0.5, loose=loose > 0, extra=len(boxes) - loose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--bedrock", action="store_true")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    args = ap.parse_args()

    src = Path(args.folder).expanduser()
    out = src / (f"claude_out_{args.effort}" if args.effort else "claude_out")
    out.mkdir(exist_ok=True)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        sys.exit(f"{src} 裡沒有圖片")
    client, model = make_client(args)
    print(f"模型 {model}，effort {args.effort or '預設'}，共 {len(paths)} 張\n")

    all_boxes, rows, results = {}, [], []
    for p in paths:
        try:
            text, usage, sec, (w, h), stop = ask(client, model, p, args.effort)
        except Exception as e:
            print(f"{p.stem}: 呼叫失敗 {type(e).__name__}: {e}")
            continue
        boxes = parse_boxes(text, w, h)
        if not boxes:
            print(f"    （{p.stem} 沒有框；stop_reason={stop}，回答開頭：{text[:120]!r}）")
        cost = usage.input_tokens / 1e6 * PRICE_IN + usage.output_tokens / 1e6 * PRICE_OUT
        all_boxes[p.stem] = boxes
        rows.append([p.stem, usage.input_tokens, usage.output_tokens, round(cost, 5), round(sec, 1), len(boxes)])
        (out / f"{p.stem}__raw.txt").write_text(text, encoding="utf-8")

        img = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(img)
        for x1, y1, x2, y2, _ in boxes:
            d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        img.save(out / f"{p.stem}__claude.png")

        s = score(p.stem, boxes, w, h)
        tag = "" if s is None else (" 命中" if s["strict"] else (" 寬鬆" if s["loose"] else " 漏")) + f" IoU {s['iou']:.2f}"
        print(f"{p.stem:<36} {len(boxes)} 框  in {usage.input_tokens} / out {usage.output_tokens} tok  "
              f"${cost:.4f}  {sec:.1f}s{tag}")
        if s:
            results.append(s)
        else:
            print(f"    （{p.stem} 找不到對應的 GT，未計分）")

    (out / "boxes_claude.json").write_text(json.dumps(all_boxes, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "usage.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([["圖", "input_tokens", "output_tokens", "費用USD", "秒", "框數"]] + rows)

    n = len(rows)
    if n:
        tot_cost = sum(r[3] for r in rows)
        print(f"\n平均每張：input {sum(r[1] for r in rows)/n:.0f} tok，output {sum(r[2] for r in rows)/n:.0f} tok，"
              f"${tot_cost/n:.4f}，{sum(r[4] for r in rows)/n:.1f} 秒")
        print(f"推算每 1,000 張：${tot_cost/n*1000:.1f}（Batch API 約 ${tot_cost/n*500:.1f}）")
    if results:
        k = len(results)
        print(f"\n對 GT：寬鬆 {sum(r['loose'] for r in results)}/{k}，嚴格 {sum(r['strict'] for r in results)}/{k}，"
              f"總框數 {sum(len(all_boxes[x]) for x in all_boxes)}，多出的框 {sum(r['extra'] for r in results)}")
    print(f"\n結果在 {out}")


if __name__ == "__main__":
    main()
