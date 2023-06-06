import os
import numpy as np
import skimage as sk
from skimage import io, color, filters, restoration
from skimage.measure import label, regionprops
from skimage import morphology
import matplotlib.pyplot as plt
import matplotlib.patches as patches

global image_obj
global min_maxr, min_maxc
global angle_threshold
global region_num

def detect_lateral_flow_tests(image, lst, line_length = 10):
    # Increase gammma of the image
    image_copy = np.copy(image)
    image_copy = sk.exposure.adjust_gamma(image_copy, 0.4)

    # Create a binary mask based on color thresholding
    threshold = filters.threshold_otsu(image_copy)
    threshold_image = image_copy > threshold
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
    copy_image = np.copy(image_copy)
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
    image_copy = copy_image

def straighten_region(copy_image, region, angles):
    minr, minc, maxr, maxc = region.bbox
    maxr, maxc = max(maxr, minr + min_maxr), max(maxc, minc + min_maxc)

    # Calculate the center of the region
    center_row = (maxr + minr) // 2
    center_col = (maxc + minc) // 2

    # Calculate the average angle for the region
    region_angles = np.rad2deg(angles)
    region_angle = np.mean(region_angles)

    # Check if the angle is greater than a threshold	
    if region_angle > angle_threshold:	
        # Rotate the region by the opposite angle	
        rotated_region = sk.transform.rotate(copy_image[minr:maxr, minc:maxc], -region_angle, center=(center_row - minr, center_col - minc))	
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
    
def remove_outliers(lst):
    min_width = float('inf')
    new_lst = []
    for region in lst:
        cur_minr, cur_minc, cur_maxr, cur_maxc = region.bbox
        cur_maxc = max(cur_maxc, cur_minc + min_maxc)
        width = cur_maxc - cur_minc
        if width < min_width:
            min_width = width
    for region in lst:
        cur_minr, cur_minc, cur_maxr, cur_maxc = region.bbox
        cur_maxc = max(cur_maxc, cur_minc + min_maxc)
        width = cur_maxc - cur_minc
        if width <= (min_width * 2):
            new_lst.append(region)
    return new_lst

def detect_corners(image):
    coords = sk.feature.corner_peaks(sk.feature.corner_harris(image), min_distance=5, threshold_rel=0.02)
    coords_subpix = sk.feature.corner_subpix(image, coords, window_size=13)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
            linestyle='None', markersize=6)
    ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    ax.axis((0, 310, 200, 0))
    plt.show()

def detect_lines(image):

    # found that the strips vary from about 2% of the length of the image to like 4%
    min_area = len(image[0]) * (len(image) * 0.0065)
    max_area = len(image[0]) * (len(image) * 0.04)

    fig = plt.figure(figsize=(10,7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    image_copy = np.copy(image)
    image_copy = sk.filters.gaussian(image_copy)
    thresholded_image = sk.filters.threshold_local(image_copy, block_size=3, method='median', mode='reflect')
    
    image_copy = image_copy * (thresholded_image > 0)

    # Adjust gamma
    image_copy = sk.exposure.adjust_gamma(image_copy, 0.3)

    enhanced_image = sk.exposure.equalize_adapthist(image_copy, clip_limit=0.05)

    # Set percentiles for range
    p80, p97 = np.percentile(enhanced_image, (80, 97))
    enhanced_image = sk.exposure.rescale_intensity(enhanced_image, in_range=(p80, p97), out_range=(0, 1))

    edges = filters.sobel(enhanced_image)

    # Step 4: Analyze and classify the shapes
    # Find contours of potential shapes
    contours = sk.measure.find_contours(edges, 0.1)

    # Label connected regions in the binary image
    labeled_image = sk.measure.label(edges)
    
    strips = []
    for region in regionprops(labeled_image):
        # Get the bounding box coordinates of the region
        minr, minc, maxr, maxc = region.bbox
        # Calculate the area of the bounding box
        area = (maxc - minc) * (maxr - minr)
        # If bounding box area is greater than min_area we expand the bounding box coordinates to the whole width of the image
        if area > min_area and (maxc - minc) > (maxr - minr):
            minc = 0
            maxc = len(image[0])
        # Check if the region is big enough to qualify as a test strip
        # Draw a rectangle around the detected test
        # area = (maxc - minc) * (maxr - minr)
        # if ((maxc - minc) > (maxr - minr)) and (area < max_area and area > min_area):
        #     strips.append(region)
        draw_rectangle(minr, minc, maxr, maxc)
        # elif (maxc - minc) > (maxr - minr):
        #     print(str(area) + " --- max: " + str(max_area) + " --- min: " + str(min_area))
    fig.add_subplot(1, 2, 2)
    plt.imshow(edges)
    plt.show()

def ai_attempt(image):
    # Step 2: Preprocessing techniques (e.g., noise reduction or contrast enhancement)
    # Apply a Gaussian blur to reduce noise
    image_blurred = filters.gaussian(image, sigma=1.0)

    # Step 3: Detect potential rectangular shapes
    # Apply an edge detection algorithm to find edges in the image
    edges = filters.sobel(image_blurred)

    # Step 4: Analyze and classify the shapes
    # Find contours of potential shapes
    contours = sk.measure.find_contours(edges, 0.1)

    # Define parameters for rectangle detection
    min_rectangle_area = len(image[0]) * (len(image) * 0.0065)  # Minimum area to consider a shape as a rectangle
    min_rectangle_aspect_ratio = 0.5  # Minimum aspect ratio to consider a shape as a rectangle

    # Iterate over the identified contours
    rectangles = []
    for contour in contours:
        # Approximate the contour with a polygon
        polygon = sk.measure.approximate_polygon(contour, tolerance=5)

        # Check if the polygon has four vertices (rectangle)
        if len(polygon) == 4:
            # Calculate the area and aspect ratio of the rectangle
            x, y = zip(*polygon)
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            aspect_ratio = (max(x) - min(x)) / (max(y) - min(y))

            # Check if the area and aspect ratio meet the criteria for a rectangle
            if area > min_rectangle_area and aspect_ratio > min_rectangle_aspect_ratio:
                rectangles.append(polygon)

    # Step 5: Label the rectangles in the image
    labeled_image = np.zeros_like(image)
    for i, rectangle in enumerate(rectangles):
        # Draw the rectangle on the labeled image
        labeled_image[sk.measure.grid_points_in_poly(image.shape, rectangle)] = i + 1

    # Step 6: Display the original grayscale image with labeled rectangles
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(labeled_image, cmap='rainbow', alpha=0.5)
    axes[1].set_title('Labeled Rectangles')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

angle_threshold = 10
folder_path = 'images'
region_num = 0
# Iterate through images in folder
for image in os.listdir(folder_path):
    # Only consider jpg and jpeg for now
    if image.endswith(('.jpg', '.jpeg')) and region_num == 2:
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
        test_strips = remove_outliers(test_strips)

        # Go though test strips and make a new list of images of just the test strips
        test_strip_images = []
        print(image)
        for test in test_strips:
            minr, minc, maxr, maxc = test.bbox
            # Adjust indecies so that they are always in the bounds of the image
            maxr, maxc = min(max(maxr, minr + min_maxr), len(image_obj)), min(max(maxc, minc + min_maxc), len(image_obj[0]))
            test_strip_images.append(image_obj[minr:maxr, minc:maxc])
        
        for test_image in test_strip_images:
            detect_lines(test_image)
        # Display the image with detected tests
        # plt.imshow(image_obj)
        # plt.axis('off')
        # plt.show()
    region_num += 1