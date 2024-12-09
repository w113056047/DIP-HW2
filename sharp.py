import cv2
import imutils
import sys
import numpy as np

imageName = "DSC00779.jpeg"
image = cv2.imread(imageName)

if image is None:
    print("Could not read input image")
    sys.exit()

# show the original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# laplacian
kernal = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(image, -1, kernal).astype(np.uint8)

cv2.imshow("Sharped", sharpened_image)
cv2.waitKey(0)

cv2.imwrite("sharped.jpeg", sharpened_image)

cv2.destroyAllWindows()