import os
import numpy as np

import skimage as sk
from skimage import io, color, filters, restoration
from skimage.measure import label, regionprops
from skimage import morphology

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

image_obj = None
min_maxr, min_maxc = 0, 0
angle_threshold = 0
file_path=""


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

def detect_lines(image) -> list:

    # found that the strips vary from about 2% of the length of the image to like 4%
    min_area = len(image[0]) * (len(image) * 0.0065)
    max_area = len(image[0]) * (len(image) * 0.04)

    # fig = plt.figure(figsize=(10,7))
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(image)
    image_copy = np.copy(image)
    image_copy = sk.filters.gaussian(image_copy)
    thresholded_image = sk.filters.threshold_local(image_copy, block_size=3, method='gaussian', mode='reflect')
    
    image_copy = image_copy * (thresholded_image > 0)

    # Adjust gamma
    image_copy = sk.exposure.adjust_gamma(image_copy, 0.5)

    enhanced_image = sk.exposure.equalize_adapthist(image_copy, clip_limit=0.05)

    # Set percentiles for range
    p80, p97 = np.percentile(enhanced_image, (80, 99))
    enhanced_image = sk.exposure.rescale_intensity(enhanced_image, in_range=(p80, p97), out_range=(0, 1))

    # Label connected regions in the binary image
    labeled_image = sk.measure.label(enhanced_image)
    
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
        if ((maxc - minc) > (maxr - minr)) and (area < max_area and area > min_area):
            strips.append([minr, maxr, minc, maxc])
            # draw_rectangle(minr, minc, maxr, maxc)
        # elif (maxc - minc) > (maxr - minr):
        #     print(str(area) + " --- max: " + str(max_area) + " --- min: " + str(min_area))
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(edges)
    # plt.show()
    return (image, strips)


# Uses imageJ to analyze the control/test strip area
def image_analysis(lines):
    image_plus = np.array(image_obj)

    plot_profiles = []
    region_percentile = .75
    line = 0
    for region in lines:
        line += 1
        min_r, max_r, min_c, max_c = region[0], region[1], region[2], region[3]
        cropped_image = image_plus[int(min_r * (2 - region_percentile)):int(max_r * region_percentile), min_c:max_c]
        profile = np.mean(cropped_image, axis=1)
        plot_profiles.append([cropped_image, profile])
    
        # Calculate the differences between adjacent elements to find slopes
    plot_results = []
    for pair in plot_profiles:
        profile = pair[1]
        start_index, end_index, modified_array, area = analyze_peaks(profile)

        second_peak_data = profile[end_index + 1:]
        if len(second_peak_data) > 1:
            second_start_index, second_end_index, second_modified_array, second_area = analyze_peaks(second_peak_data)
            second_start_index += end_index + 1
            second_end_index += end_index + 1

            plot_results.append([pair[0], pair[1], [(start_index, end_index, area), (second_start_index, second_end_index, second_area)]])
        else:
            plot_results.append([pair[0], pair[1], [(start_index, end_index, area)]]) 
    
    return plot_results
    # for result in plot_results:
    #     if len(result) > 1:
    #         start_index, end_index, area = result[0]
    #         second_start_index, second_end_index, second_area = result[1]
    #         print(f"Peak {1} - Start index: {start_index}, End index: {end_index}, Area: {area}")
    #         print(f"Peak {2} - Start index: {second_start_index}, End index: {second_end_index}, Area: {second_area}")
    #     else:
    #         start_index, end_index, area = result[0]
    #         print(f"Peak {1} - Start index: {start_index}, End index: {end_index}, Area: {area}")
            

def analyze_peaks(profile):
    peak_index = np.argmax(profile)

    # Search backward to find the start index
    start_index = peak_index
    while start_index > 0 and profile[start_index] >= profile[start_index - 1]:
        start_index -= 1

    # Search forward to find the end index
    end_index = peak_index
    while end_index < len(profile) - 1 and profile[end_index] >= profile[end_index + 1]:
        end_index += 1

    # Create a line segment connecting start_index and end_index
    modified_array = np.copy(profile)
    modified_array[start_index:end_index + 1] = np.linspace(profile[start_index], profile[end_index],
                                                            end_index - start_index + 1)

    area = np.trapz(profile[start_index:end_index + 1]) - np.trapz(modified_array[start_index:end_index + 1])
    
    return start_index, end_index, modified_array, area

class MyGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Image Results")

        self.button = tk.Button(self.window, text="Open Image", command=self.open_image)
        self.button.pack()

        self.results = []
        self.current_index = -1

        # Create GUI components
        self.canvas = tk.Canvas(self.window, width=400, height=400)
        self.canvas.pack()

        self.prev_button = tk.Button(self.window, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.window, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.LEFT)


    def open_image(self):
        global file_path
        # Image to analyze
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        self.process_image(file_path)

    def process_image(self, file_path):
        global min_maxc
        global min_maxr
        global image_obj
        global angle_threshold

        angle_threshold = 10
        # Only consider jpg and jpeg for now
        image_obj = io.imread(file_path, as_gray=True)

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
        for test in test_strips:
            minr, minc, maxr, maxc = test.bbox
            # Adjust indecies so that they are always in the bounds of the image
            maxr, maxc = min(max(maxr, minr + min_maxr), len(image_obj)), min(max(maxc, minc + min_maxc), len(image_obj[0]))
            test_strip_images.append([minr, maxr, minc, maxc])

        # Run image_analysis funciton
        result = image_analysis(test_strip_images)

        self.current_index = 0
        self.results = result

        # Display results
        self.show_result()
    
    def show_result(self):
        result_image = self.results[self.current_index][0]
        result_array = self.results[self.current_index][1]
        result_params = self.results[self.current_index][2]

        result_image = Image.fromarray(result_image)
        result_image = result_image.resize((400, 400))
        self.result_photo = ImageTk.PhotoImage(result_image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.result_photo)
        
        if len(result_params) == 2:
            # Get parameters from result_params
            start_index = result_params[0][0]
            end_index = result_params[0][1]
            area = result_params[0][2]

            second_start_index = result_params[1][0]
            second_end_index = result_params[1][1]
            second_area = result_params[1][2]

            fig, (ax1, ax2) = plt.figure()
            ax1.imshow(result_image)
            ax1.axis("off")

            ax2.plot(result_array, label='Original Array')
            ax2.scatter(start_index, result_array[start_index], color='red', label='Start Index (First Peak)')
            ax2.scatter(end_index, result_array[end_index], color='green', label='End Index (First Peak)')
            ax2.plot([start_index, end_index], [result_array[start_index], result_array[end_index]], 'k--',
                    label='Diagonal Line (First Peak)')

            ax2.scatter(second_start_index, result_array[second_start_index], color='purple',
                        label='Start Index (Second Peak)')
            ax2.scatter(second_end_index, result_array[second_end_index], color='orange',
                        label='End Index (Second Peak)')
            ax2.plot([second_start_index, second_end_index],
                    [result_array[second_start_index], result_array[second_end_index]], 'k--',
                    label='Diagonal Line (Second Peak)')
            ax2.annotate(f'Area (Peak {1}): {area:.2f}', xy=(start_index, result_array[start_index]),
                        xytext=(start_index, result_array[start_index] + 1), arrowprops=dict(arrowstyle='->'))
            ax2.annotate(f'Area (Peak {2}): {second_area:.2f}', xy=(second_start_index, result_array[second_start_index]),
                        xytext=(second_start_index, result_array[second_start_index] + 1), arrowprops=dict(arrowstyle='->'))
            
            ax2.xlabel('Index')
            ax2.ylabel('Value')
            ax2.title('Array Analysis')
            ax2.legend()

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT)
    
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_result()
    
    def show_next(self):
        if self.current_index < len(self.results) - 1:
            self.current_index += 1
            self.show_result()

    def run(self):
        # Start the window's main loop
        self.window.mainloop()

gui = MyGUI()
gui.run()