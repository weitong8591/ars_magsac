import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_matches(img1, kp1, img2, kp2, matches, inliers, color=None):
    """
    https://gist.github.com/woolpeeker/d7e1821e1b5c556b32aafe10b7a1b7e8

    Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.

    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.

    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 3
    thickness = 3
    if color:
        c = color
    for i, m in enumerate(matches):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        #print(c)
        if i%5==0:
            # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
            # wants locs as a tuple of ints.
            if inliers[i]>=1:

            	end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
            	end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
            	cv2.line(new_img, end1, end2, c.tolist(), thickness)
            	cv2.circle(new_img, end1, r, c.tolist(), thickness)
            	cv2.circle(new_img, end2, r, c.tolist(), thickness)

    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()
    return new_img
