#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import cv2
from matplotlib import pyplot as plt

# Creating Gaussian low-pass filter 
def low_pass_filter(rows, cols, cutoff_freq):
    img_filter = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            euc_dist = np.sqrt((r-rows/2)**2 + (c-cols/2)**2) #Using Euclidean Distance to help compute filter shape, i.e., low frequencies in center and high frequencies away from center
            img_filter[r,c] = np.exp( (-euc_dist**2) / (2*(cutoff_freq**2))) #Using cutoff frequency to obtain a Gaussian distribution
    return img_filter

# Transform image to frequency domain and create low-Pass filtered image in spatial domain
def low_pass_fft(x):
    fm = np.fft.fft2(x) # Transform 2D image into frequency domain using Fourier Transform
    fmshift = np.fft.fftshift(fm) # Shift low frequencies to center of image and high Frequencies to corners
    product_freq = fmshift * low_pass_filter # Computing the product of the shifted fourier transform image with the low pass filter in frequency domain
    lowpassed = np.fft.ifftshift(product_freq) # Shift frequencies back to original placement (high frequencies at center, low frequencies away from center)
    lowpassed = np.abs(np.fft.ifft2(lowpassed).real) # Perform inverse Fourier transformation to get image back to spatial domain
    lowpassed = cv2.normalize(lowpassed,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U) # Before merging, normalize to convert array from float64 to uint8
    return lowpassed

# Transform image to frequency domain and compute high-pass filtered Image in spatial domain
def high_pass_fft(x):
    fm = np.fft.fft2(x)
    fmshift = np.fft.fftshift(fm)
    mult_high = fmshift * high_pass_filter #Multiply image in frequency domain (shifted Fourier transform) with high pass filter
    highpassed = np.fft.ifftshift(mult_high)
    highpassed = np.abs(np.fft.ifft2(highpassed).real)
    highpassed = cv2.normalize(highpassed,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)
    return highpassed

    
# Read in Trump image, resize image, split b,g,r channels
img = cv2.imread("trump.png")
print(img.shape)
img = cv2.resize(img, (250, 300))
b,g,r = cv2.split(img)

# Create Gaussian Low Pass/High Filters
rows, cols = img.shape[:2]
low_pass_filter = low_pass_filter(rows, cols, 10)
high_pass_filter = 1-low_pass_filter # compute high-pass filter by subtracting 1 from low-pass filter values, where the values lie between 0 (high freq) and 1 (low freq)

# Plot high-pass Trump image
plt.figure()
b,g,r = map(high_pass_fft, (b,g,r)) # call map to perform high_pass_fft on each color channel
r = r + 50 #adding equal values to increase brightness of image
g = g + 50
b = b + 50 
high_pass_img = cv2.merge((b,g,r)) # merge all 3 color channels
high_pass_img = cv2.cvtColor(high_pass_img, cv2.COLOR_BGR2RGB) # convert image to RGB
plt.imshow(high_pass_img) # display image


# Read in cat image, resize image, split b,g,r channels
img1 = cv2.imread("cat.png")
img1 = cv2.resize(img1, (250, 300))
b,g,r = cv2.split(img1)

plt.figure()
b,g,r = map(low_pass_fft, (b,g,r)) # call map to perform high_pass_fft on each color channel
low_pass_img = cv2.merge((b,g,r)) # merge all 3 color channels
low_pass_img = cv2.cvtColor(low_pass_img, cv2.COLOR_BGR2RGB) # convert image to RGB
plt.imshow(low_pass_img1) # display image


#Display Hybrid Image
hybrid = cv2.addWeighted(low_pass_img1, 0.4, high_pass_img, 0.6, 0) # adjust weights to make low-pass image less opaque

#Displaying Hybrid Images
plt.figure(figsize=(8, 8), constrained_layout=False)
plt.subplot(121), plt.imshow(np.abs(hybrid)), plt.title("Hybrid Low-Pass Cat & High-Pass Trump Image")
plt.axis('off')
plt.figure(figsize=(2, 2), constrained_layout=False) #Display smaller image to get the effect of peering image from far away
plt.subplot(122), plt.imshow(np.abs(hybrid)), plt.title("Far Away Appearance of Hybrid Low-Pass Cat & High-Pass Trump Image")
plt.axis('off')

# save the image
plt.imsave('hybrid.png', hybrid)


# 

# In[ ]:




