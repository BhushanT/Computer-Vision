import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def translate(image, translation_x, translation_y):
    rows, cols, dim = image.shape
    translate_matrix = np.float32([[1, 0, translation_x],
                               [0, 1, translation_y],
                               [0, 0, 1]])
    translated_image = cv2.warpPerspective(image, translate_matrix, (int(cols),int(rows)))
    return translated_image

def shear(image, shear_coefficient):  # x + ay, y
    rows, cols, dim = image.shape
    shear_matrix = np.float32([[1, shear_coefficient,0],
                               [0, 1,               0],
                               [0, 0,               1]])
    sheared_image = cv2.warpPerspective(image, shear_matrix, (3*int(cols),3*int(rows)))
    sheared_image = translate(sheared_image,0,0.5*cols)
    return sheared_image
        
def rotate(image, angle):
    rows, cols, dim = image.shape
    rotation_matrix = np.float32([[np.cos(angle)   , -(np.sin(angle)),0],
                                  [np.single(angle), np.cos(angle)   ,0],
                                  [0               ,          0      ,1]])
    rotated_image = cv2.warpPerspective(image, rotation_matrix, (2*int(cols),2*int(rows)))
    return rotated_image

def scale(image, scale_factor):   # this function is uniform scaling only
    rows, cols, dim = image.shape
    scale_matrix = np.float32([[scale_factor, 0,0],
                               [0, scale_factor,0],
                               [0,0,1]])
    scaled_image = cv2.warpPerspective(image, scale_matrix, (2*int(cols),2*int(rows)))
    return scaled_image




img = cv2.imread('test_image_black.png')
input = img
shear_coefficient = 2
angle = np.radians(45)   #function takes input in radians, either input radians directly or convert it from degrees first
scale_factor = math.sqrt(2)  
translation_x = 4
translation_y = 0
sheared_image = shear(input, shear_coefficient)
rotated_image = rotate(input, angle)
scaled_image = scale(input, scale_factor)
translated_image = translate(input, translation_x, translation_y)
combined_image = translate(scale(rotate(sheared_image,angle),scale_factor), translation_x, translation_y)

plt.figure(figsize=(10,10))
plt.subplot(231), plt.imshow(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
plt.title('Input'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Shear'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Rotate'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Scale'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Translate'), plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Combine'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()








