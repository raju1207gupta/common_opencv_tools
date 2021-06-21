import cv2
from cv2 import dnn_superres
import sys
import os

def upsample(image,scale_factor):
    
    #path_to_file = config_reader.getPPE_PathToCkpt()
    path_to_folder = "path_to_superresolution_autoencoder"
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    # Read image
    #image = cv2.imread('./input.png')

    # Read the desired model
    #path = "EDSR_x4.pb"
    if scale_factor==4:
        #path = f"{path_to_folder}/FSRCNN-small_x4.pb"
        path = f"{path_to_folder}/FSRCNN_x4.pb"
        sr.readModel(path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("fsrcnn", 4)
        #sr.setModel("edsr", 4)

    # if scale_factor==4:
    #     path = f"{path_to_folder}/FSRCNN_x4.pb"
    #     sr.readModel(path)

    #     # Set the desired model and scale to get correct pre- and post-processing
    #     sr.setModel("fsrcnn", 4)
    #     #sr.setModel("edsr", 4)

    if scale_factor==3:
        path = f"{path_to_folder}/FSRCNN_x3.pb"
        sr.readModel(path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("fsrcnn", 3)
        #sr.setModel("edsr", 4)
    
    if scale_factor==2:
        path = f"{path_to_folder}/FSRCNN_x2.pb"
        sr.readModel(path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("fsrcnn", 2)
        #sr.setModel("edsr", 4)

    # Upscale the image
    result = sr.upsample(image)
    

    return result

path  = "path_to_stored_images/"

PATH_TO_IMAGE = "path_to_be_changed_images/"
mask_model_inp_dim = [299,299]
dirs = os.listdir(PATH_TO_IMAGE)
for item in dirs:
    if os.path.isfile(PATH_TO_IMAGE+item):
        
        try:
            if not os.path.isfile(path+item):
                face = cv2.imread(PATH_TO_IMAGE+item)
                face = upsample(face,4)
                (face_h, face_w) = face.shape[:2]
                face = cv2.resize(face, (mask_model_inp_dim[0], mask_model_inp_dim[1]),fx=int(mask_model_inp_dim[0]/face_w),fy=int(mask_model_inp_dim[1]/face_h), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('img', face)
                cv2.imwrite(os.path.join(path,f'{str(item)}'),face)
        except:
            pass
        
        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break

cv2.destroyAllWindows()