from skimage import io, filters
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('globeshot.png')

##cv2.imshow('image', image)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

#Pic1
#displays the greyscale version of the image adjacent to an Edge version
#with titles using matplotlib

img = cv2.imread('globeshot.png', 0)
edges = cv2.Canny(img, 100, 200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


##Pic2
##displays the greyscale version along with four other filters on the image using
##different types of thresholds

img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,210,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 210)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


#Pic3
#displays the reverse of the original image beside the orange areas isolated and
#reversed

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_orange = np.array([3,100,100])
upper_orange = np.array([23,255,255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_orange, upper_orange)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(image,image, mask= mask)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(res,cmap = 'gray')
plt.title('Orange Image'), plt.xticks([]), plt.yticks([])

plt.show()


