import cv2 as cv2


import numpy as np

from pathlib import Path

from PIL import Image

path  = 'D:/deeplearning learn/Social distancing/face-mask-detector/final_dataset_images/'
PATH_TO_IMAGE = "D:/deeplearning learn/Social distancing/face-mask-detector/dataset_HUL_annotate1_person/"

frmnumber = 828
dirs = os.listdir(PATH_TO_IMAGE)
for frmno,item in enumerate(dirs):
    if os.path.isfile(f"{PATH_TO_IMAGE}{(frmno+1)*60}.jpg"):
        img = cv2.imread(f"{PATH_TO_IMAGE}{(frmno+1)*60}.jpg")
        
        frmnumber = frmnumber+1
        
        cv2.imshow('img', img)
        cv2.imwrite(os.path.join(path,f'{str(frmnumber)}.jpg'),img)
        

        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break

# Clean
#cap.release()
cv2.destroyAllWindows()