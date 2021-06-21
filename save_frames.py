import cv2 as cv2
import time
import os
import numpy as np

path  = 'path_to_store_images/'

cap = cv2.VideoCapture("path_to_video_link")


frmnumber = 0
while cap.isOpened() :
    
    ret, img = cap.read()
    #img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    frmnumber = frmnumber+1
    img = cv2.resize(img,(1280,720))
   
    rows = img.shape[1]
    cols = img.shape[0]
    if frmnumber%2==0:
        cv2.imshow('img', img)
        cv2.imwrite(os.path.join(path,f'{str(frmnumber)}.jpg'),img)
        cv2.resizeWindow('Frame',800,600)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
cap.release()
cv2.destroyAllWindows()