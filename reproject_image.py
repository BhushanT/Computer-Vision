import numpy as np
from scipy import linalg
import cv2
from matplotlib import pyplot as plt

P = np.array([[5  ,400,500,20 ],
     [100,300,490,20 ],
     [1  ,  1,  1,5  ]])


U,D,V = np.linalg.svd(P)

camera_center = np.divide(V[3],V[3][3])
print("The camera center in the real world is: \n X: " + str(camera_center[0]) + "\n Y: " + str(camera_center[1]) + "\n Z: " + str(camera_center[2]))

#Use RQ decomposition to find calibration matrix K. Then use that to find focal lengths x and y
M = P[:,:3]
K,R = linalg.rq(M)
K = np.divide(K,K[2][2])
K[:, 1] = -K[:, 1]  #Make focal length positive and correspondingly adjust the rotation matrix
R[1, :] = -R[1, :]
print("=========")
print("K: " + str(K))
print("R: " + str(R))

# Z = 2f as definied by the problem statement. Takes the form [X, Y, 2f, 1]transpose x =PX

# Load the image
img = cv2.imread('bhushan_image.png')
height, width, channel = img.shape

# Initialize output image
reprojected_image = np.zeros((height, width, channel), dtype=np.uint8)

# Define camera parameters
f = 120

for i in range(height):
    for j in range(width):
        X_cam = np.array([width/2 - j, height/2 - i, 2*f, 1])
        x_img = np.dot(K, np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1).dot(X_cam))
        current_j = int(np.clip(round(x_img[0] / x_img[2]), 0, width - 1))
        current_i = int(np.clip(round(x_img[1] / x_img[2]), 0, height - 1))
        reprojected_image[current_i, current_j, :] = img[i, j, :]

reprojected_image = np.flipud(reprojected_image)
reprojected_image = np.fliplr(reprojected_image)

plt.figure(figsize=(10,10))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Old Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(cv2.cvtColor(reprojected_image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('New Image'), plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()