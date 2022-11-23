# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Setting GPU-enabled Tensorflow
import tensorflow as tf
import cv2
import numpy as np
from deepface import DeepFace
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
      # tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
      tf.config.experimental.set_memory_growth(gpu, True)
      print("sucessful")
  except RuntimeError as e:
    print(e)

# pass in a dictionary of confidence score for each emotion
def get_state_of_mind(emotions_dict):
    # declaring variables    
    confusion_list = ['neutral', 'surprise'] # duplicated combinatorial emotions for 'sad' with 'Disappointment/dissatisfaction', so 'sad' is removed from 'Confusion'
    satisfaction_delighterd_list = ['happy', 'neutral']
    disappointment_disstisfaction_list = ['neutral', 'sad']
    frustrated_list = ['sad', 'angry', 'neutral']
    # dictionary of ecombinatorial emotions with its respective state of mind
    state_of_mind_dict = {'Confusion': confusion_list, 
    'Satisfaction/Delighted': satisfaction_delighterd_list, 
    'Disappointment/Dissatisfaction': disappointment_disstisfaction_list,
    'Frustrated': frustrated_list}

    # dictionary used to store and determin the state of mind based on the combinatorial emotions
    state_of_mind_count = {'Confusion': [0, 0], 
    'Satisfaction/Delighted': [0, 0], 
    'Disappointment/Dissatisfaction': [0, 0],
    'Frustrated': [0, 0, 0]}
    state_of_mind_list = ['Confusion', 'Satisfaction/Delighted', 'Disappointment/Dissatisfaction', 'Frustrated']
    count = 0
    state_switched = False

    # sort emotion from highest confidence score to lowest
    # sorted_emotions = emotions_dict # TODO: use for testing
    sorted_emotions = sorted(emotions_dict, key=emotions_dict.get, reverse=True)[:]
    for emotion in sorted_emotions:
        for state in state_of_mind_list:
            # if the emotion belongs to a certain state of mind
            if emotion in state_of_mind_dict[state]: 
                if count == 0:
                    state_of_mind_list.remove(state)
                    state_of_mind_list.insert(0, state)
                    state_switched = True
                # get the index of the emotion in the dictionary
                idx = state_of_mind_dict[state].index(emotion)
                # set the count to 1
                state_of_mind_count[state][idx] = 1
            
            # check if all emotions of a certain state of mind were matched
            if sum(state_of_mind_count[state]) == len(state_of_mind_count[state]):
                return state
        
        if state_switched:
            count = 1



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # set the name of the .mp4 file here
    filename = 'video_1'
    # filename = 'video_2'
    # use the line below if you would like to use your webcam to record your own face
    # filename = 'webcam'

    # declare the facce detection model to detect human faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # read the video using VideoCapture funtion
    video = cv2.VideoCapture(os.getcwd() + "/data/" + filename + ".mp4")
    # use the line below if you would like to use your webcam to record your own face
    # video = cv2.VideoCapture(0)

    # get width and height
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # declare a VideoWriter instance
    result_mp4 = cv2.VideoWriter(os.getcwd() + "/output/" + filename + '_result.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (frame_width, frame_height))

    try:
        while video.isOpened():
            # read the video into frame
            ret, frame = video.read()
            ret, frame1 = video.read() 

            # A. face detection, emotion classification, state of mind
            # A1. convert the frame to grayscale for CascadeClassifier
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # A2. find coordinates of faces, returns (x,y,w,h)
            faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)

            # B. motion detection
            # B1. find the absolute difference between current frame and previous frame
            diff = cv2.absdiff(frame, frame1)
            # B2. convert the frame difference to grayscale 
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            # B3. apply gaussian smoothing onto the grayscale frame to reduce noise 
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            # B4. apply thresholding 
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            # B5. dilate the frame to remove small unwanted detection
            dilated = cv2.dilate(thresh, None, iterations=3)
            # B6. get the contours of detected movement in frame
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # legends in video
            cv2.putText(frame, "Blue: Emotion", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Red: State of Mind", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Green: Motion", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop through all detected faces
            for (x,y,w,h) in faces:
                color = (255,0,0) # blue color
                # A3. draw a blue rectangle around the face
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                try:
                    # A4. perform emotion classification on the face detected
                    analyze = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
                    # A5. get the dominant emotion as the label 
                    emotion_label = analyze['dominant_emotion']
                    # A6. perform state of mind calculation
                    state_of_mind_label = get_state_of_mind(analyze['emotion'])
                    # A7. write the emotion on the frame in blue text
                    frame=cv2.putText(frame, emotion_label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    # A8. write the state of mind on the frame in red text
                    frame=cv2.putText(frame, state_of_mind_label, (x, y + h + 12 ),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                except Exception as e:
                    print(e)

            # loop through all contours
            for contour in contours:
                # B7. generate x, y coordinate, width, height based on the contour
                (x, y, w, h) = cv2.boundingRect(contour)
                # B8. draw a green rectangle if the contour area is greater than 600
                if cv2.contourArea(contour) > 600:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            result_mp4.write(frame)

            cv2.imshow("video", frame) 
            frame = frame1
            ret, frame1 = video.read()
            #   cv2_imshow(frame) # for google colab only
            # cv2.waitKey(5000)
            if cv2.waitKey(40) == 27:
                break

        cv2.destroyAllWindows()
        video.release
        result_mp4.release()


    except Exception as e:
        print(e)
    cv2.destroyAllWindows()
    video.release
    result_mp4.release()
