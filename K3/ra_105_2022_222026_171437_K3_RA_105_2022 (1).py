import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
import pandas as pd
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input

from tensorflow.keras.optimizers import SGD

np.random.seed(42) 
tf.random.set_seed(42)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

# ***** internet *****

def select_roi(image_orig, image_bin):
    contours, _ = cv2.findContours(
        image_bin.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[0])

    merged_boxes = []

    for box in boxes:
        if not merged_boxes:
            merged_boxes.append(box)
            continue

        last = merged_boxes[-1]

        if boxes_overlap(last, box):
            merged_boxes[-1] = merge_boxes(last, box)
        else:
            merged_boxes.append(box)

    regions = []
    for x, y, w, h in merged_boxes:
        region = image_bin[y:y+h, x:x+w]
        regions.append(resize_region(region))
        cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0,255,0), 2)

    return image_orig, regions

def boxes_overlap(a, b, min_area=20):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    if bw * bh < min_area or aw * ah < min_area:
        return False

    return not (
        ax + aw < bx or
        bx + bw < ax or
        ay + ah < by or
        by + bh < ay
    )

def merge_boxes(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y

    return (x, y, w, h)

# ***********

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.axis('off')
    plt.show()

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def dilate(image):
    kernel = np.ones((3, 3)) 
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((2, 2))
    return cv2.erode(image, kernel, iterations=1)

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

# ***** internet *****

def convert_output(labels, alphabet):
    nn_outputs = []
    for char in labels:
        output = np.zeros(len(alphabet), dtype=np.float32)
        output[alphabet.index(char)] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Input(shape=(784, )))
    ann.add(Dense(128, activation='relu'))
    ann.add(Dense(len(alphabet), activation='softmax')) 

    return ann

# ***********

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
    
    sgd = SGD(learning_rate=0.02, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0, shuffle=False)
    return ann

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[np.argmax(output)])
    return result

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

# ***** internet *****

def hamming_distance(true_text, predicted_text):
    min_len = min(len(true_text), len(predicted_text))
    
    diff = sum(1 for i in range(min_len) if true_text[i] != predicted_text[i])
    diff += abs(len(true_text) - len(predicted_text))
    
    return diff

def make_alphabet(labels_df):
    seen = set()
    alphabet = []

    for text in labels_df["text"]:
        for char in text:
            if char not in seen:
                seen.add(char)
                alphabet.append(char)

    return alphabet


# ***** * *****

data_dir = sys.argv[1]

labels_df = pd.read_csv(os.path.join(data_dir, "texts.csv"), sep="|")

letters = []  
letters_labels = [] 

alphabet = make_alphabet(labels_df)

# slike izabrane za trening tako da pokriju sve moguce karaktere

train_files = [
    "picture_1.png",
    "picture_3.png",
    "picture_4.png",
    "picture_5.png",
    "picture_9.png",
    "picture_10.png"
]

for idx, row in labels_df.iterrows():
    filename = os.path.join(data_dir, row["image"])
    if row["image"] not in train_files:
        continue
    true_text = row["text"]

    img = load_image(filename)
    img_gray = image_gray(img)

    edges = cv2.Canny(img_gray, 30, 100, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, maxLineGap=5)

    angles = []
    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        angle = np.median(angles)
    else:
        angle = 0

    img_rot = ndimage.rotate(img, angle)

    img_bin = image_bin(image_gray(img_rot))
    img_bin = erode(img_bin)

    imagegs, regions = select_roi(img_rot.copy(), img_bin)
    #display_image(imagegs)

    for region, char in zip(regions, true_text):
        letters.append(region)         
        letters_labels.append(char) #internet

inputs = prepare_for_ann(letters)
outputs = convert_output(letters_labels, alphabet)
ann = create_ann(output_size=len(alphabet))
ann = train_ann(ann, inputs, outputs, epochs=1000)

distance = 0

for idx, row in labels_df.iterrows():
    filename = os.path.join(data_dir, row["image"])
    true_text = row["text"]

    img = load_image(filename)
    img_gray = image_gray(img)

    edges = cv2.Canny(img_gray, 30, 100, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, maxLineGap=5)

    angles = []
    if lines is not None:
        for [[x1, y1, x2, y2]] in lines:
            angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        angle = np.median(angles)
    else:
        angle = 0

    img_rot = ndimage.rotate(img, angle)

    img_bin = image_bin(image_gray(img_rot))
    img_bin = erode(img_bin)

    imagegs, regions = select_roi(img_rot.copy(), img_bin)

    test_inputs = prepare_for_ann(regions)
    result = ann.predict(np.array(test_inputs, np.float32), verbose=0)
    #print(display_result(result, alphabet))
    
    predicted_letters = display_result(result, alphabet)
    predicted_text = "".join(predicted_letters)

    distance += hamming_distance(true_text, predicted_text)

print(f"{distance:.1f}")