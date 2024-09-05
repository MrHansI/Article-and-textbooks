import cv2
import numpy as np
import torch 


class Preprocessing:
    def sobel(image):
        sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        img_src1 = cv2.filter2D(image, -1, sx)
        img_src2 = cv2.filter2D(image, -1, sy)
        return img_src1 + img_src2

    def line_detector(image, coef=1, line=-256):
        l1 = np.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]])
        l2 = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
        return cv2.filter2D(image, -1, l1) + cv2.filter2D(image, -1, l2)

    def edge_detector(image):
        l1 = np.array([[-4, 5, 5], [-4, 5, 5], [-4, -4, -4]])
        l2 = np.array([[5, 5, 5], [-4, 5, -4], [-4, -4, -4]])
        return cv2.filter2D(image, -1, l1) + cv2.filter2D(image, -1, l2)

    def mean(image, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
        return cv2.filter2D(image, ddepth=-1, kernel=kernel)

    def gaussian(image, kernel_size=3, std=1):
        kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=std)
        return cv2.filter2D(image, ddepth=-1, kernel=kernel)

    def gabor(image, ksize=3, theta=np.pi / 4, sigma=3.0, lmbd=10.0, gamma=0.5, psi=0):
        filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lmbd, gamma, psi, ktype=cv2.CV_32F)
        return cv2.filter2D(image, -1, filter)
