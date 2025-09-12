from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import time

# ------------------------------------------PIL,Numpy,OS,Matplotlib,Scipy-------------------------------------------------
# Func 1: Average image
def average_image(list):
      # Create new array
    image = Image.open(list[0])
    total_array = np.array(image,"f")
    count = 1
    try:  
    # Average image    
        for i in list[1:]:
            total_array += np.array(Image.open(i),"f")
            count += 1
        average_array = total_array/count # Average image
        # Make image from average image
        image_array = Image.fromarray(average_array.astype("uint8"))
    except Exception as e:
        print(f"Error: {e}")
    return image_array

# Func 2: Save average image
def save_average_image(folder_path,list):
    # Check name for average image 
    image = Image.open(list[0])
    new_name = "average_image"
    extension = image.format.lower()
    if not new_name.lower().endswith(f".{extension}"):
        new_name += f".{extension}"
    # Call average_image func and save as a new one
    image_array = average_image(list)
    new_average_image = image_array.save(os.path.join(folder_path,new_name),format=image.format,quality=100)
    return new_average_image,new_name

# Func 3: Balance gray image by CDF algorithm
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

# Func 4: Revert grayscale image (255-image)
def revert_image(image_equalized_array):
    reverted_image = 255-image_equalized_array
    return reverted_image

# Func 5: Contour
def contour_image(image_equalized_array):
    image_contour = plt.contour(image_equalized_array,origin="image")
    return image_contour

# Func 6: Input 10 points
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

# Func 3: Cascade Face detection
def cascade_face_video(video_path):
    # Call cascade func
    face_cascade = cv2.CascadeClassifier("F:/09.ComputerVision/raver_library/Haar_Cascade(ML)/" \
                                        "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("F:/09.ComputerVision/raver_library/Haar_Cascade(ML)/" \
                                        "haarcascade_eye_tree_eyeglasses.xml")
    # Convert frame to grayscale
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret,frame = video.read()
        frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not ret:
            print("Video issue")
            break
        # Face,eye detection by Cascade
        face = face_cascade.detectMultiScale(frame_grayscale,scaleFactor=1.1,minNeighbors=5,minSize=(20,20))
        eye = eye_cascade.detectMultiScale(frame_grayscale,scaleFactor=1.1,minNeighbors=1,minSize=(2,2))
        # Draw a rectangle around face,eye
        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
        for (x,y,w,h) in eye:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
        cv2.imshow("Video Playback",frame)
        if cv2.waitKey(10)==ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()
    return face,eye