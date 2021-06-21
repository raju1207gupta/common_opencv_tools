import cv2 as cv2
import time
import os
import numpy as np
import imutils

path  = 'E:/deeplearning_stuff/new_usecase/'

cap = cv2.VideoCapture(0)

firstFrame=None

fps = int(cap.get(cv2.CAP_PROP_FPS))

frmnumber = 0
while cap.isOpened() :
    
    ret, img = cap.read()
    #img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    frmnumber = frmnumber+1
    img = cv2.resize(img,(1280,720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    # else:
    #     firstFrame = cv2.GaussianBlur(firstFrame_gray, (21, 21), 0)
        

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 1:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    rows = img.shape[1]
    cols = img.shape[0]
    # if frmnumber%2==0:
    #     cv2.imshow('img', img)
        #cv2.imwrite(os.path.join(path,f'{str(frmnumber)}.jpg'),img)
        
    cv2.imshow("Security Feed", img)
    #cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    if frmnumber%((fps*1)*1)==0:
        firstFrame = None
    #cv2.resizeWindow('img',800,600)
    key = cv2.waitKey(25) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
cap.release()
cv2.destroyAllWindows()