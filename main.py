import cv2
import imutils
import sys
import numpy as np

imageName = "input.jpeg"
image = cv2.imread(imageName)

if image is None:
    print("Could not read input image")
    sys.exit()

# show the original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)


# laplacian
l_kernal = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
laplacian = cv2.filter2D(image, -1, l_kernal)

cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)

l_sharped = cv2.subtract(image, laplacian)
l_sharped = np.where(l_sharped < 0, 0, l_sharped).astype(np.uint8)
cv2.imshow("sharp", l_sharped)
cv2.waitKey(0)

cv2.imwrite("l_sharped.jpeg", l_sharped)

g_x_kernal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
g_y_kernal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
g_x = cv2.filter2D(image, -1, g_x_kernal)
g_y = cv2.filter2D(image, -1, g_y_kernal)

# cv2.imshow("Gx", g_x)
# cv2.waitKey(0)

# cv2.imshow("Gy", g_y)
# cv2.waitKey(0)

gradient = (g_x.__pow__(2).__add__(g_y.__pow__(2))).__pow__(0.5)

a_filter = np.array(
    [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
)

average_gradient = cv2.filter2D(gradient, -1, a_filter)

cv2.imshow("Average Gradient", average_gradient)
cv2.waitKey(0)

mask = np.zeros_like(average_gradient)
cv2.normalize(average_gradient, mask, 0.0, 1.0, cv2.NORM_MINMAX)
mask = np.where(mask < 0.5, 0, 1).astype(np.uint8)

enhanced_detail = cv2.multiply(mask, laplacian).astype(np.uint8)

cv2.imshow("Enhanced Detail", enhanced_detail)
cv2.waitKey(0)

result = cv2.subtract(image, enhanced_detail)

outputFile = "output.jpeg"
cv2.imwrite(outputFile, result)

cv2.imshow("output", result)
cv2.waitKey(0)

cv2.destroyAllWindows()
