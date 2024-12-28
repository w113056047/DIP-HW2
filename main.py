import cv2
import sys
import numpy as np


def conv(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0]
    l = kernel.shape[1]

    new_image = np.zeros_like(src)
    src = cv2.copyMakeBorder(
        src, 1, 1, 1, 1, cv2.BORDER_REPLICATE,
    )

    for i in range(src.shape[0] - 2):
        for j in range(0, src.shape[1] - 2):
            product = np.sum(
                np.multiply(
                    src[i : i + k, j : j + l],
                    kernel,
                )
            )

            if product < 0:
                product = 0
            elif product > 255:
                product = 255

            new_image[i, j] = product

    return new_image

scale = 0.2

image_name = "intput.jpeg"
image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Could not read input image")
    sys.exit()

# show the original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)


# laplacian
l_kernal = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
laplacian = conv(image, l_kernal)

cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)

l_sharped = cv2.addWeighted(image, 1, laplacian, -scale, 0)
l_sharped = np.where(l_sharped < 0, 0, l_sharped).astype(np.uint8)
cv2.imshow("sharp", l_sharped)
cv2.waitKey(0)

cv2.imwrite("l_sharped.jpeg", l_sharped)

g_x_kernal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
g_y_kernal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
g_x = conv(image, g_x_kernal)
g_y = conv(image, g_y_kernal)

# cv2.imshow("Gx", g_x)
# cv2.waitKey(0)

# cv2.imshow("Gy", g_y)
# cv2.waitKey(0)

gradient = (g_x ** 2 + g_y ** 2) ** 0.5

cv2.imshow("Gradient", gradient.astype(np.uint8))
cv2.waitKey(0)

average_blur_filter = np.array(
    [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
)

average_gradient = conv(gradient, average_blur_filter)

# cv2.imshow("Average Gradient", average_gradient)
# cv2.waitKey(0)

mask = np.zeros_like(average_gradient)
cv2.normalize(average_gradient, mask, 0.0, 1.0, cv2.NORM_MINMAX)
mask = np.where(mask < 0.5, 0, 1).astype(np.uint8)

enhanced_detail = cv2.multiply(mask, laplacian).astype(np.uint8)

cv2.imshow("Enhanced Detail", enhanced_detail)
cv2.waitKey(0)

result = cv2.addWeighted(image, 1, enhanced_detail, -scale, 0)

outputFile = "output.jpeg"
cv2.imwrite(outputFile, result)

cv2.imshow("output", result)
cv2.waitKey(0)

cv2.destroyAllWindows()
