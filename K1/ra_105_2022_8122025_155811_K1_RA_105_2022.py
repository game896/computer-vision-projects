import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_folder = sys.argv[1]

#internet ***
csv_path = os.path.join(data_folder, "coin_value_count_g1.csv")
df = pd.read_csv(csv_path, skiprows=1, header=None)
df.columns = ["filename", "value"]
df["value"] = df["value"].astype(int)

value_map = dict(zip(df["filename"], df["value"]))

image_files = [
    f for f in os.listdir(data_folder)
    if f.lower().endswith((".jpg"))
]
# ***

MAE = 0

for image_file in image_files:

    img_path = os.path.join(data_folder, image_file)
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, beta=20) #30 daje bolji mae, ali mi se ne svidja kako radi
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=140,
        param2=30,
        minRadius=20,
        maxRadius=30
    )

    big_circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=50,
        param1=150,
        param2=30,
        minRadius=40,
        maxRadius=51
    )

    filtered_circles = []

    small_c = 0
    big_c = 0

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for (x, y, r) in circles:
            if (x > 280 and y > 200) or (x < 1550 and y > 150):
                filtered_circles.append((x, y, r))
                small_c += 1

    if big_circles is not None:
        big_circles = np.uint16(np.around(big_circles[0]))
        for (x, y, r) in big_circles:
            if (x > 280 and y > 200) or (x < 1550 and y > 150):
                filtered_circles.append((x, y, r))
                big_c += 5

    MAE += abs(value_map[image_file] - (small_c + big_c))

    #output = img.copy()

    #for (x, y, r) in filtered_circles:
    #    cv2.circle(output, (x, y), r, (0, 255, 0), 2)

    #plt.title(image_file)
    #plt.imshow(output)
    #plt.show()


print(f"{MAE/len(image_files):.2f}")
