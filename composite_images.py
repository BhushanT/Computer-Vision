import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
def process_input(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    [magnitude, angle ] = cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1])
    return dft_shift,magnitude,angle

def reconstruct_magnitude_only(image):
    dft_shift,magnitude,angle = process_input(image)

    magnitude_only_angle = np.float32(np.zeros(angle.shape))
    [dft_shift[:,:,0], dft_shift[:,:,1]] = cv2.polarToCart(magnitude, magnitude_only_angle)
    inverse_shift = np.fft.ifftshift(dft_shift)
    reconstructed_image = cv2.idft(inverse_shift)
    reconstructed_image = cv2.normalize(cv2.magnitude(reconstructed_image[:,:,0], reconstructed_image[:,:,1]), None, 0, 30, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return reconstructed_image

def reconstruct_phase_only(image):
    dft_shift,magnitude,angle = process_input(image)
    phase_only_magnitude = np.float32(np.ones(magnitude.shape))
    [dft_shift[:,:,0], dft_shift[:,:,1]] = cv2.polarToCart(phase_only_magnitude, angle)
    inverse_shift = np.fft.ifftshift(dft_shift)
    reconstructed_image = cv2.idft(inverse_shift)
    reconstructed_image = cv2.normalize(cv2.magnitude(reconstructed_image[:,:,0], reconstructed_image[:,:,1]), None, 0, 30, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return reconstructed_image

def composite_image(image_A, image_B):     # Using the magnitude of A and the phase of B
    dft_shift_A,magnitude_A,angle_A = process_input(image_A)
    dft_shift_B,magnitude_B,angle_B = process_input(image_B)
    [dft_shift_A[:,:,0], dft_shift_A[:,:,1]] = cv2.polarToCart(magnitude_A, angle_B)
    inverse_shift = np.fft.ifftshift(dft_shift_A)
    reconstructed_image = cv2.idft(inverse_shift)
    reconstructed_image = cv2.normalize(cv2.magnitude(reconstructed_image[:,:,0], reconstructed_image[:,:,1]), None, 0,1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return reconstructed_image

img_A = cv2.imread('image_1.jpg')
img_B = cv2.imread('image_2.jpg')

magnitude_only_A = reconstruct_magnitude_only(img_A)
phase_only_A = reconstruct_phase_only(img_A)
magnitude_only_B = reconstruct_magnitude_only(img_B)
phase_only_B = reconstruct_phase_only(img_B)

composite_A_B = composite_image(img_A, img_B)
composite_B_A = composite_image(img_B, img_A)

input_rgb_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)


#CV library processes input in bgr, not rgb. Matplot lib uses rgb.
reconstruct_rgb_A = cv2.cvtColor(phase_only_A, cv2.COLOR_BGR2RGB)
reconstruct_2_rgb_A = cv2.cvtColor(magnitude_only_A, cv2.COLOR_BGR2RGB)
input_rgb_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
reconstruct_rgb_B = cv2.cvtColor(phase_only_B, cv2.COLOR_BGR2RGB)
reconstruct_2_rgb_B = cv2.cvtColor(magnitude_only_B, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.subplot(231), plt.imshow(input_rgb_A)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(reconstruct_rgb_A, cmap='gray')
plt.title('Phase Only'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(reconstruct_2_rgb_A, cmap='gray')
plt.title('Magnitude Only'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(input_rgb_B)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(reconstruct_rgb_B, cmap='gray')
plt.title('Phase Only'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(reconstruct_2_rgb_B, cmap='gray')
plt.title('Magnitude Only'), plt.xticks([]), plt.yticks([])


plt.show()

plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(input_rgb_A)
plt.title('Input A'), plt.xticks([]), plt.yticks([])

composite_rgb_A = cv2.cvtColor(composite_A_B, cv2.COLOR_BGR2RGB)
plt.subplot(222), plt.imshow(composite_rgb_A)
plt.title('Composite B on A'), plt.xticks([]), plt.yticks([])

input_rgb_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
plt.subplot(223), plt.imshow(input_rgb_B)
plt.title('Input B'), plt.xticks([]), plt.yticks([])

composite_rgb_B = cv2.cvtColor(composite_B_A, cv2.COLOR_BGR2RGB)
plt.subplot(224), plt.imshow(composite_rgb_B)
plt.title('Composite A on B'), plt.xticks([]), plt.yticks([])

plt.show()


