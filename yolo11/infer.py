#!/usr/bin/env python3
"""Thermal-infrared person-state detection (YOLO11s).

Input : 80x62 thermal-infrared pseudo-color PNG (IR only, no RGB).
Output: one bounding box per person plus one of four states
        (lie / sit / other / off_bed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


CLASS_NAMES = ["lie", "sit", "other", "off_bed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/best.pt", help="YOLO11s checkpoint")
    parser.add_argument("--source", required=True, help="IR image or directory")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        args.source, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
    )

    for result in results:
        print(f"{Path(result.path).name}:")
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print("  (no person detected)")
            continue
        for box in boxes:
            cls = int(box.cls.item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            print(f"  {CLASS_NAMES[cls]:8s} conf={float(box.conf.item()):.3f} bbox={xyxy}")


if __name__ == "__main__":
    main()
