import cv2 as cv
import numpy as np

# Load the haystack image (larger image) in reduced color mode for faster processing
haystack_img = cv.imread('img.png', cv.IMREAD_UNCHANGED)

# Load the needle image (template image to be found) in reduced color mode
needle_img = cv.imread('img_1.png', cv.IMREAD_UNCHANGED)

# template matching
result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

# Find the minimum and maximum values and their locations in the result matrix
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

# Print the top-left position of the best match and the confidence value of the match
print('Best match top left position %s' % str(max_loc))
print('Best match confidence: %s' % max_val)

# Set a threshold to determine if a match is considered good
# Might have to go over this number again
threshold = 0.8
if max_val >= threshold:  # If the best match confidence is greater than or equal to the threshold
    print('Found needle')  # Indicate that the needle image was found

    # Get the width and height of the needle image
    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    # Calculate the bottom-right corner of the rectangle to be drawn
    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    # Draw a rectangle around the detected match
    cv.rectangle(haystack_img, top_left, bottom_right,
                 color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)

    # Save the result image with the drawn rectangle
    cv.imwrite('result.jpg', haystack_img)

else:
    print('Needle not found')  # Indicate that the needle image was not found