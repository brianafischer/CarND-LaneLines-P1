# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import imageio
import os
import math

# %matplotlib inline - not sure why this was in the iPython notebook...
# Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML


# reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    plt.imshow(mask)
    plt.show()

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below

    Note: Since the origin of the image is the top left corner, the following
    slopes determine a left or right lane:
        Left Lane  = Negative Slope
        Right Lane = Positive Slope
    """

    left_lane_slopes = []
    right_lane_slopes = []

    left_lane_intercepts = []
    right_lane_intercepts = []

    y_min = img.shape[0]
    y_max = y_min

    for line in lines:
        for x1, y1, x2, y2 in line:
            line_segment_slope = (y2 - y1) / (x2 - x1)

            # Collect Left Lane Segments
            if -math.inf < line_segment_slope < 0.0:
                left_lane_slopes.append(line_segment_slope)
                left_lane_intercepts.append(y1 - line_segment_slope * x1)

            # Collect Right Lane Segments
            if 0.0 < line_segment_slope < math.inf:
                right_lane_slopes.append(line_segment_slope)
                right_lane_intercepts.append(y1 - line_segment_slope * x1)

            # Find the segment that is furthest away
            y_min = min(y_min, y1, y2)

    y_min += 15  # add small threshold

    if len(right_lane_slopes) > 0:
        ave_positive_slope = sum(right_lane_slopes) / len(right_lane_slopes)
        intercept = sum(right_lane_intercepts) / len(right_lane_intercepts)
        x_min = int((y_min - intercept) / ave_positive_slope)
        x_max = int((y_max - intercept) / ave_positive_slope)
        cv2.line(img, (x_min, y_min), (x_max, y_max), [255, 0, 128], 10)

    if len(left_lane_slopes) > 0:
        ave_negative_slope = sum(left_lane_slopes) / len(left_lane_slopes)
        intercept = sum(left_lane_intercepts) / len(left_lane_intercepts)
        x_min = int((y_min - intercept) / ave_negative_slope)
        x_max = int((y_max - intercept) / ave_negative_slope)
        cv2.line(img, (x_min, y_min), (x_max, y_max), [255, 0, 128], 10)

    plt.imshow(img)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Select algorithm parameters here
GAUSSIAN_KERNEL = 3
CANNY_LOW = 75
CANNY_HIGH = 150

BOTTOM = 75  # bottom region of interest
TOP = 40  # top region of interest

RHO = 2  # distance resolution (pixels) of the Hough transform
THETA = np.pi / 180  # angular resolution (radians) of the Hough transform
MIN_VOTES = 100  # min number of votes (intersections in Hough grid cell)
MIN_LINE_LEN = 20  # min number of pixels making up a line
MAX_LINE_GAP = 30  # max gap in pixels between line segments


def pipeline(image):
    # 1. Convert the image to grayscale
    grayscale_image = grayscale(image)
    plt.imshow(grayscale_image, cmap='gray')
    plt.show()

    # 2. apply a gaussian blur to remove noise
    smooth_image = gaussian_blur(grayscale_image, GAUSSIAN_KERNEL)
    plt.imshow(smooth_image, cmap='gray')

    # 3. apply canny edge detection
    edges = canny(smooth_image, CANNY_LOW, CANNY_HIGH)
    plt.imshow(edges, cmap='gray')

    # 4. apply the region of interest
    image_height = image.shape[0]  # should be 540 pixels
    image_width = image.shape[1]  # should be 960 pixels

    vertices = np.array([[(BOTTOM, image_height),
                          (image_width / 2 - TOP, image_height / 2 + TOP),
                          (image_width / 2 + TOP, image_height / 2 + TOP),
                          (image_width - BOTTOM, image_height)]], dtype=np.int32)

    regions = region_of_interest(edges, vertices)
    # todo: draw the region of interest
    plt.imshow(regions, cmap='gray')

    # 5. now apply a hough transform on the detected edges to find the lines
    hough_image = hough_lines(regions, RHO, THETA, MIN_VOTES, MIN_LINE_LEN, MAX_LINE_GAP)
    plt.imshow(hough_image)
    plt.show()

    return weighted_img(hough_image, image)


#
# TEST ON PICTURES SECTION...
#
for test_image in os.listdir("test_images/"):
    image = mpimg.imread("test_images/" + test_image)
    output = pipeline(image)
    # mpimg.imsave("out_images/" + test_image, output)
    plt.imshow(output)
    plt.show()
