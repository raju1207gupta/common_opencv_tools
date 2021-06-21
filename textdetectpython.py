import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

img = cv2.imread("imagetotextTest.jpg")

text = pytesseract.image_to_string(img)
f = open("outputfile.txt", "w")
f.write(str(text))
f.close()
print(text)