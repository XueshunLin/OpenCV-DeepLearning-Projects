# import the necessary packages
import cv2
import matplotlib.pyplot as plt
import numpy as np

def imrect(im1):
# Perform Image rectification on an 3D array im.
# Parameters: im1: numpy.ndarray, an array with H*W*C representing image.(H,W is the image size and C is the channel)
# Returns: out: numpy.ndarray, rectified image。
#   out =im1
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    
    # Convert the image to 8-bit unsigned integers
    img_gray = (img_gray * 255).astype(np.uint8)
    
    # Applying a average blur to the image 
    img_blur = cv2.blur(img_gray, (10,10))
    
    # used adaptive thresholding to create a binary image
    # used the mean thresholding method
    # in case to deal with different lighting conditions
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    # Detecting edges in the imagel, using the Canny edge detector
    img_edges = cv2.Canny(img_thresh, 50, 200)
    # Applying a morphological operation to close gaps in between object edges
    # used a elliptical shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    img_closed = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel, iterations=10)

    # Finding the contours in the image using the cv2.findContours method
    contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sorting the contours by area, and keeping only the largest one
    maxcontour = max(contours, key=cv2.contourArea)

    # Approximating the polygonal curves of the contour, then finding the corners of the polygon
    epsilon = 0.01 * cv2.arcLength(maxcontour, True)
    corners = cv2.approxPolyDP(maxcontour, epsilon, True)
    
    # sort the points based on their x-coordinates
    horizontally_sorted = corners[np.argsort(corners[:, 0, 0])].squeeze()
    # grab the left side points and right side points from the sorted x-coordinates
    left = horizontally_sorted[:2]
    right = horizontally_sorted[2:]

    # sort the points in the left side so that the top-left point is the one with higher y-coordinate
    # and the bottom-left point is the one with lower y-coordinate
    top_left, bottom_left = left[np.argsort(left[:, 1])]

    # compute the Euclidean distance between the top-left and the two right side points
    # the point with the maximum distance will be the bottom-right point
    distance = np.linalg.norm(top_left - right, axis=1)
    top_right, bottom_right = right[np.argsort(distance)]

    # re-arrange the points in the order
    corners_in_order = np.float32([top_left, top_right, bottom_right, bottom_left])

    # Define the width and height of the output image, set to the average of the distances between the corners
    width = np.mean([np.linalg.norm(top_left - top_right), np.linalg.norm(bottom_left - bottom_right)]) 
    height = np.mean([np.linalg.norm(top_left - bottom_left), np.linalg.norm(top_right - bottom_right)])

    # Define the four corners of the output image
    targets = np.float32([[50, 50], [width - 50, 50], [width - 50, height - 50], [50, height - 50]])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(corners_in_order, targets)

    # Apply the perspective transform to the input image
    out = cv2.warpPerspective(im1, M, (int(width), int(height)))

    return (out)
 

if __name__ == "__main__":

    # This is the code for generating the rectified output
    # If you have any question about code, please send email to e0444157@u.nus.edu
    # fell free to modify any part of this code for convenience.
    img_names = ['./data/test1.jpg','./data/test2.jpg']
    for name in img_names:
        image = np.array(cv2.imread(name, -1), dtype=np.float32)/255.
        rectificated = imrect(image)
        cv2.imwrite('./data/Result_'+name[7:],np.uint8(np.clip(np.around(rectificated*255,decimals=0),0,255)))
