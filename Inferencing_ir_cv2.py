import subprocess
subprocess.call([r'C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin\setupvars.bat'])


import cv2 as cv
print(cv.__file__)

# Load the model
net = cv.dnn.readNet('C:/Users/Raju/Documents/Intel/OpenVINO/openvino_models/ir/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml', 'C:/Users/Raju/Documents/Intel/OpenVINO/openvino_models/ir/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.bin')

# Specify target device
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
     
# Read an image
frame = cv.imread('D:/deeplearning learn/Automatic_License_plate_recognition/ANPR/Licence_plate_detection/test.jpg')
     
# Prepare input blob and perform an inference
blob = cv.dnn.blobFromImage(frame, size=(300, 300))
net.setInput(blob)
out = net.forward()
     
# Draw detected faces on the frame
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])

    if confidence > 0.5:
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

# Save the frame to an image file
cv.imshow('out.png', frame) 
cv.waitKey(0)
