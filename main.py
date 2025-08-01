import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Load the image.
IMAGE_PATH = 'images/front/single_book_cover_good_contrast_1.jpeg'
image = cv2.imread(IMAGE_PATH)

# Start with english, spanish, and french.
reader = easyocr.Reader(['en','es','fr'], gpu=False)

# -------------- Pre-processing -------------------

# convert image to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------------- Perform OCR ----------------------

# Use OCR to detect text in image.
result = reader.readtext(gray)

# -------------- Output Visual --------------------

# Get the image.
img = gray.copy()

# Loop through each detection result and mark it down.
for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    text = detection[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 5)
    img = cv2.putText(img, text, top_left, font, 1, (255,255,255), 2, cv2.LINE_AA)
    print(f"[{text}], Confidence: {detection[2] * 100:.2f}%")

# Display the plot.
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()