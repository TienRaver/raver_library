from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

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
        if ret != True: # Check frame during playing
            print("Video error")
            break
        # Get basic video info and show it
        frame_count = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_length = round(video.get(cv2.CAP_PROP_POS_MSEC)/1000,1)
        cv2.putText(frame,f"FPS:{frame_count}",(50,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f"Resolution:{video_width}x{video_height}",
                    (50,70),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        cv2.putText(frame,f"Time:{video_length}",(50,90),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        cv2.imshow("Video play",frame)
        # Stop video any time
        if cv2.waitKey(10) == ord("q"):
            break  
    # Terminate video  
    video.release()
    cv2.destroyAllWindows()