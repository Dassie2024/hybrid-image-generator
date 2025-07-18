# Hybrid Image Generator

This project blends two images into a single optical illusion by combining the low-frequency components of one image with the high-frequency details of another using Fourier transforms.

### The result? 
A trippy hybrid where you might see Trump’s face at a glance, but a cat’s face when you squint or step back. 
It’s a mind-bending exploration of human perception and frequency filtering in image processing!

## Demo
![Hybrid Example](images/hybrid.png)

## Techniques Used
- Fourier Transform
- Frequency Filtering
- Gaussian Filters
- Perceptual Image Blending

## How to Run
1. `python hybrid.py`
2. Input: Two aligned images (same dimensions)
3. Output: Hybrid image

## Requirements
- numpy
- matplotlib
- OpenCV

## Source Images

| Trump Image | Cat Image |
|-------------|-----------|
| ![](images/trump.png) | ![](images/cat.png) |

## Hybrid Result

![Hybrid Image](images/hybrid.png)
