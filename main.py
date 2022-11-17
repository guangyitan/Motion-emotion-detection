# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
from deepface import DeepFace
import numpy as np
import tensorflow as tf
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(len(tf.config.list_physical_devices('GPU'))>0)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def analyse_face():
    imagepath = "happy_face_woman.png"
    image = cv2.imread(imagepath)
    face_analysis = DeepFace.analyze(image)
    print(face_analysis)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    analyse_face()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
