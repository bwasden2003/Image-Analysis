import os
import numpy as np
import skimage as sk
from skimage import io, color, filters
from skimage.measure import label, regionprops
from skimage import morphology
import matplotlib.pyplot as plt
import matplotlib.patches as patches

global image_obj
global min_maxr, min_maxc

def detect_lateral_flow_tests(image, lst, line_length = 10):
    # Increase gammma of the image
    image = sk.exposure.adjust_gamma(image, 0.4)

    # Create a binary mask based on color thresholding
    threshold = filters.threshold_otsu(image)
    threshold_image = image > threshold

    edges = sk.feature.canny(threshold_image)
    
    # Perform Hough line detection
    h, theta, d = sk.transform.hough_line(edges)

    # Find the most prominent lines
    _, angles, dists = sk.transform.hough_line_peaks(h, theta, d)

    # Create eroded and dilated versions
    eroded = morphology.binary_erosion(threshold_image, morphology.rectangle(line_length, 1))
    dilated = morphology.binary_dilation(eroded, morphology.rectangle(line_length, 1))

    # io.imshow(dilated)
    # plt.show()
	
    # Label connected components in the binary mask
    labeled = label(dilated)
    background_color = np.mean(labeled[0:1, 0:1], axis=(0, 1))
    # Analyze each labeled region
    copy_image = np.copy(image)
    for region in regionprops(labeled):
        # Get the bounding box coordinates of the region
        minr, minc, maxr, maxc = region.bbox
        maxr, maxc = max(maxr, minr + min_maxr), max(maxc, minc + min_maxc)
        # Calculate the mean color within the bounding box
        mean_color = np.mean(image[minr:maxr, minc:maxc], axis=(0, 1))

        # Check if the mean color is lighter than the background color and the region is big enough to qualify as a test strip
        if mean_color > background_color and region.area >= 100:
            # Draw a rectangle around the detected test
            if check_duplicate(minr, minc, maxr, maxc, lst):
                straighten_region(copy_image, region, angles)
                lst.append(region)    
    image = copy_image

def straighten_region(copy_image, region, angles):
    minr, minc, maxr, maxc = region.bbox
    maxr, maxc = max(maxr, minr + min_maxr), max(maxc, minc + min_maxc)

    # Calculate the center of the region
    center_row = (maxr + minr) // 2
    center_col = (maxc + minc) // 2

    # Calculate the average angle for the region
    region_angles = np.rad2deg(angles)
    region_angle = np.mean(region_angles)

    # Rotate the region to straighten it
    rotated_region = sk.transform.rotate(copy_image[minr:maxr, minc:maxc], region_angle, center=(center_row - minr, center_col - minc))

    # Replace the rotated region in the copy of the original image
    copy_image[minr:maxr, minc:maxc] = rotated_region

def check_duplicate(minr, minc, maxr, maxc, lst) -> bool:
    # Iterate through tests that have already been identified
    for region in lst:
        cur_minr, cur_minc, cur_maxr, cur_maxc = region.bbox
        cur_maxr, cur_maxc = max(cur_maxr, cur_minr + min_maxr), max(cur_maxc, cur_minc + min_maxc)
        # Check whether the current region's dimensions overlap
        if (minr <= cur_maxr and maxr >= cur_minr and minc <= cur_maxc and maxc >= cur_minc):
            return False
    return True

def draw_rectangle(minr, minc, maxr, maxc):
    # Create a rectangle patch
    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=2, edgecolor='r', facecolor='none')
    # Add the rectangle to the current plot
    plt.gca().add_patch(rect)


folder_path = 'images'
# Iterate through images in folder
for image in os.listdir(folder_path):
    # Only consider jpg and jpeg for now
    if image.endswith(('.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, image)
        image_obj = io.imread(image_path, as_gray=True)

        # Find width (in pixels) of the current image
        width = len(image_obj[0])

        # Use width to calculate the min size (in pixels) a LFA could be (this heavily depends on how the images are taken and will probably have to be reworked)
        min_maxc = (int)(width / 40)
        min_maxr = (int)(min_maxc * 20)

        # Make a list for the test strip regions within the image
        test_strips = []
        detect_lateral_flow_tests(image_obj, test_strips)

        # Go though test strips and make a new list of images of just the test strips
        test_strip_images = []
        for test in test_strips:
            minr, minc, maxr, maxc = test.bbox
            # Adjust indecies so that they are always in the bounds of the image
            maxr, maxc = min(max(maxr, minr + min_maxr), len(image_obj)), min(max(maxc, minc + min_maxc), len(image_obj[0]))
            test_strip_images.append(image_obj[minr:maxr, minc:maxc])
        # for test in test_strips:
        #     minr, minc, maxr, maxc = test.bbox
        #     draw_rectangle(minr, minc, maxr, maxc)
        for test in test_strip_images:
            lines = []
            detect_lateral_flow_tests(test, lines)
            for line in lines:
                minr, minc, maxr, maxc = line.bbox
                draw_rectangle(minr, minc, maxr, maxc)
            plt.imshow(test)
            plt.show()

		# Display the image with detected tests
        # plt.imshow(image_obj)
        # plt.axis('off')
        # plt.show()
    else:
        continue