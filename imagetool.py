from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import time

# ------------------------------------------PIL,Numpy,OS,Matplotlib,Scipy-------------------------------------------------
# Func 1: Balance gray image by CDF algorithm
def cdf_image (folder_path,new_name):
    # Check gray mode of new average image
    image = Image.open(os.path.join(folder_path,new_name))
    if image.mode != "L":
        image = image.convert("L")
    # CDF algorithm
    image_array = np.array(image)
    histogram,bins = np.histogram(image_array,bins=256,range=(0,256),density=True)
    cdf = histogram.cumsum()
    cdf = 255*cdf/cdf[-1]
    image_equal = np.interp(image_array,bins[:-1],cdf)
    image_equalized = Image.fromarray(image_equal.astype("uint8"))
    # Return CDF image
    return image_equalized

# Func 2: Input 10 points
def ginput_image(folder_path,new_name):
    image = Image.open(os.path.join(folder_path,new_name))
    plt.imshow(image)
    plt.title("Input 10 points")
    points = plt.ginput(10,timeout=0)
    plt.show()
    # List x,y
    plt.figure()
    plt.imshow(image)
    plt.title("10 points")
    x=y=[]
    for point in points:
        x,y=point
        plt.plot(x,y,"bo")
    plt.show()
    return x,y

# ------------------------------------------openCV-------------------------------------------------
# Func 1: Open video
def play_video(video_path):
    # Open video link
    video = cv2.VideoCapture(video_path)
    # Read video
    while video.isOpened():
        ret,frame = video.read()
        # Check frame during playing
        if ret != True: 
            print("Video error")
            break
        # Get basic video info and show it
        frame_count = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_length = round(video.get(cv2.CAP_PROP_POS_MSEC)/1000,1)
        cv2.putText(frame,f"FPS:{frame_count}",(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        cv2.putText(frame,f"Resolution:{video_width}x{video_height}",
                    (50,70),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        cv2.putText(frame,f"Time:{video_length}",(50,90),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        cv2.imshow("Video playback",frame)
        if cv2.waitKey(10)==ord("q"):
            break
    video.release()
    cv2.destroyWindow("Video playback")
    return frame

# Func 2: Open webcam
def play_webcam():
    # Open webcam
    cam = cv2.VideoCapture(0)
    # Read video
    while cam.isOpened():
        # Check frame during playing
        ret,frame = cam.read()
        if ret != True:
            print("Webcam error")
            break
        # Display time on video
        cam_time = round(time.time(),1)
        cv2.putText(frame,f"Time:{cam_time}s",(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
        cv2.imshow("Cam playback",frame)
        if cv2.waitKey(10)==ord("q"):
            break
    cam.release()
    cv2.destroyWindow("Cam playback")
    return frame
# Func 3: Face detection by DNN Caffe