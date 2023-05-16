import numpy as np
import skimage as sk
from skimage import io, color, filters
from skimage.measure import label, regionprops
from skimage import morphology
import matplotlib.pyplot as plt
import matplotlib.patches as patches

global image
global test_strips
global min_maxr, min_maxc

def detect_lateral_flow_tests(image, line_length = 10):
    
    # Increase gammma of the image
    image = sk.exposure.adjust_gamma(image, 0.5)

    # Create a binary mask based on color thresholding
    threshold = filters.threshold_otsu(image)
    threshold_image = image > threshold

    # Create eroded and dilated versions
    eroded = morphology.binary_erosion(threshold_image, morphology.rectangle(line_length, 1))
    dilated = morphology.binary_dilation(eroded, morphology.rectangle(line_length, 1))

    edges = sk.feature.canny(image)

    # Perform Hough line detection
    h, theta, d = sk.transform.hough_line(edges)

    # Find the most prominent lines
    _, angles, dists = sk.transform.hough_line_peaks(h, theta, d)

    plt.imshow(dilated)
    # Label connected components in the binary mask
    labeled = label(dilated)
    background_color = np.mean(labeled[0:1, 0:1], axis=(0, 1))
    # Analyze each labeled region
    for region in regionprops(labeled):
        # Get the bounding box coordinates of the region
        minr, minc, maxr, maxc = region.bbox
        maxr, maxc = max(maxr, minr + min_maxr), max(maxc, minc + min_maxc)
        # Calculate the mean color within the bounding box
        mean_color = np.mean(image[minr:maxr, minc:maxc], axis=(0, 1))

        # Check if the mean color is lighter than the background color and the region is big enough to qualify as a test strip
        if mean_color > background_color and region.area >= 100:
            # Draw a rectangle around the detected test
            if check_duplicate(minr, minc, maxr, maxc):
                straighten_region(region, angles)
                test_strips.append(region)    
    plt.show()

def straighten_region(region, angles):
    minr, minc, maxr, maxc = region.bbox
    maxr, maxc = max(maxr, minr + min_maxr), max(maxc, minc + min_maxc)

    region_angles = np.rad2deg(angles)
    region_angle = np.mean(region_angles)

    # Rotate the region to straighten it
    region_image = image[minr:maxr, minc:maxc]
    rotated_region = sk.transform.rotate(region_image, region_angle)

    # Replace the rotated region in the original image
    image[minr:maxr, minc:maxc] = rotated_region    

def check_duplicate(minr, minc, maxr, maxc) -> bool:
    # Iterate through tests that have already been identified
    for test in test_strips:
        cur_minr, cur_minc, cur_maxr, cur_maxc = test.bbox
        cur_maxr, cur_maxc = max(cur_maxr, cur_minr + min_maxr), max(cur_maxc, cur_minc + min_maxc)
        # Check whether the current region's dimensions overlap
        if (minr <= cur_maxr and maxr >= cur_minr and minc <= cur_maxc and maxc >= cur_minc):
            return False
    return True

def draw_rectangle(minr, minc, maxr, maxc):
    # Create a rectangle patch
    maxr, maxc = max(maxr, minr + min_maxr), max(maxc, minc + min_maxc)
    # coordinates = [(minr, minc), (minr, maxc), (maxr, minc), (maxr, maxc)]
    # x_coords, y_coords = zip(*coordinates)
    # plt.plot(x_coords, y_coords, 'b-')  # Plot the points and connect them with lines
    # plt.gca().set_aspect('equal')  # Set aspect ratio to equal (if needed)
    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=2, edgecolor='r', facecolor='none')
    # Add the rectangle to the current plot
    plt.gca().add_patch(rect)


image_path = 'images/image2.jpeg'
image = io.imread(image_path, as_gray=True)
width = len(image[0])
min_maxc = (int)(width / 40)
min_maxr = (int)(min_maxc * 20)
test_strips = []
detect_lateral_flow_tests(image)
for test in test_strips:
    minr, minc, maxr, maxc = test.bbox
    draw_rectangle(minr, minc, maxr, maxc)


# Display the image with detected tests
plt.imshow(image)
plt.axis('off')
plt.show()