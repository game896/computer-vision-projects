import sys
import os
import cv2
import numpy as np
import pandas as pd

# *** internet ***
data_folder = sys.argv[1]

csv_path = os.path.join(data_folder, "count.csv")
df = pd.read_csv(csv_path)
df["video"] = df["video"].astype(str) + ".mp4"
df["count"] = df["count"].astype(int)

value_map = dict(zip(df["video"], df["count"]))

video_files = [
    f for f in os.listdir(data_folder)
    if f.lower().endswith(".mp4")
]
# ****************
MAE = 0

for video_file in video_files:

    video_path = os.path.join(data_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    success, prev_frame = cap.read()
    if not success:
        continue #bez ovoga ce biti cudnih problema

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    h, w = prev_gray.shape
    x1_upper, x2_upper = int(w * 0.43), int(w * 0.57)
    y1_upper, y2_upper = int(h * 0.45), int(h * 0.72)

    x1_lower, x2_lower = int(w * 0.35), int(w * 0.65)
    y1_lower, y2_lower = int(h * 0.55), int(h * 0.93)

    count_upper = 0
    count_lower = 0
    last_detected_upper = False
    last_detected_lower = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray, prev_gray)
        diff = cv2.GaussianBlur(diff, (7, 7), 0)
        _, diff = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

        box_upper = diff[y1_upper:y2_upper, x1_upper:x2_upper]
        box_lower = diff[y1_lower:y2_lower, x1_lower:x2_lower]

        circles_upper = cv2.HoughCircles(box_upper, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200, param1=100, param2=12, minRadius=46, maxRadius=50)

        detected_upper = circles_upper is not None
        # *** crtanje, sa interneta ***
        # if detected_upper:
        #     dbg = frame.copy()

        #     for c in circles_upper[0]:
        #         x, y, r = c
        #         cv2.circle(
        #             dbg,
        #             (int(x + x1_upper), int(y + y1_upper)),
        #             int(r),
        #             (0, 255, 0),
        #             2
        #         )

        #     cv2.rectangle(
        #         dbg,
        #         (x1_upper, y1_upper),
        #         (x2_upper, y2_upper),
        #         (255, 0, 0),
        #         2
        #     )

        #     cv2.imshow("DEBUG", dbg)
        #     cv2.waitKey(1)

        if detected_upper and not last_detected_upper:
            count_upper += 1

        last_detected_upper = detected_upper

        circles_lower = cv2.HoughCircles(box_lower, cv2.HOUGH_GRADIENT, dp=1.21, minDist=200, param1=110, param2=12, minRadius=62, maxRadius=65)

        detected_lower = circles_lower is not None

        # *** crtanje, sa interneta ***
        # if detected_lower:
        #     dbg = frame.copy()

        #     for c in circles_lower[0]:
        #         x, y, r = c
        #         cv2.circle(
        #             dbg,
        #             (int(x + x1_lower), int(y + y1_lower)),
        #             int(r),
        #             (0, 255, 0),
        #             2
        #         )

        #     cv2.rectangle(
        #         dbg,
        #         (x1_lower, y1_lower),
        #         (x2_lower, y2_lower),
        #         (255, 0, 0),
        #         2
        #     )

        #     cv2.imshow("DEBUG", dbg)
        #     cv2.waitKey(1)

        if detected_lower and not last_detected_lower:
            count_lower += 1

        last_detected_lower = detected_lower

        prev_gray = gray

    cap.release()

    gt_value = value_map.get(video_file, 0)
    MAE += abs(gt_value - count_upper - count_lower)

print(f"{MAE / len(video_files):.2f}")
