import os

import numpy as np

import cv2

import csv

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

# https://stackoverflow.com/questions/55636313/selecting-an-area-of-an-image-with-a-mouse-and-recording-the-dimensions-of-the-s

def detect_lateral_flow_tests(image, line_length = 10):
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
    
    # Label connected components in the binary mask
    labeled = label(dilated)
    background_color = np.mean(labeled[0:1, 0:1], axis=(0, 1))
    # Analyze each labeled region
    copy_image = np.copy(image_copy)
    lst = []
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
                lst.append([minr, maxr, minc, maxc])
    image_copy = copy_image
    lst = remove_outliers(lst)
    lfa_images = []
    for region in lst:
        minr, maxr, minc, maxc = region[0], region[1], region[2], region[3]
        lfa_image = image_copy[minr:maxr, minc:maxc]
        lfa_images.append(lfa_image)
    return lfa_images

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

    fig = plt.figure(figsize=(10,7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
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
            draw_rectangle(minr, minc, maxr, maxc)
        # elif (maxc - minc) > (maxr - minr):
        #     print(str(area) + " --- max: " + str(max_area) + " --- min: " + str(min_area))
    return strips


# Function needs to be reworked so that it takes in one image and returns one output plot
def image_analysis(test_strips):
    image_plus = np.array(image_obj, dtype='int64')

    plot_profiles = []
    region_percentile = 1
    line = 0
    for region in test_strips:
        line += 1
        min_r, max_r, min_c, max_c = region[0], region[1], region[2], region[3]
        height = max_r - min_r
        width = max_c - min_c
        cropped_image = image_plus[min_r + int(height * .2):min_r + int(height * .6), min_c + int(width * .4):min_c + int(width * .6)]
        # cropped_image = image_plus[int(line_start - part_height):int((max_r - min_r) // 2 + part_height), min_c:max_c]
        plt.imshow(cropped_image)
        plt.show()
        profile = np.nanmean(cropped_image, axis=1)
        plot_profiles.append([cropped_image, profile])
    
    plot_results = []
    for pair in plot_profiles:
        profile = pair[1]
        start_index, end_index, modified_array, area = analyze_peaks(profile)

        second_peak_data = profile[end_index + 1:]
        if len(second_peak_data) > 1:
            second_start_index, second_end_index, second_modified_array, second_area = analyze_peaks(second_peak_data)
            second_start_index += end_index + 1
            second_end_index += end_index + 1

            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)

            ax.plot(profile, label='Original Array')
            ax.scatter(start_index, profile[start_index], color='red', label='Start Index (First Peak)')
            ax.scatter(end_index, profile[end_index], color='green', label='End Index (First Peak)')
            ax.plot([start_index, end_index], [profile[start_index], profile[end_index]], 'k--',
                    label='Diagonal Line (First Peak)')

            ax.scatter(second_start_index, profile[second_start_index], color='purple',
                        label='Start Index (Second Peak)')
            ax.scatter(second_end_index, profile[second_end_index], color='orange',
                        label='End Index (Second Peak)')
            ax.plot([second_start_index, second_end_index],
                    [profile[second_start_index], profile[second_end_index]], 'k--',
                    label='Diagonal Line (Second Peak)')
            ax.annotate(f'Area (Peak {1}): {area:.2f}', xy=(start_index, profile[start_index]),
                        xytext=(start_index, profile[start_index] + 1), arrowprops=dict(arrowstyle='->'))
            ax.annotate(f'Area (Peak {2}): {second_area:.2f}', xy=(second_start_index, profile[second_start_index]),
                        xytext=(second_start_index, profile[second_start_index] + 1), arrowprops=dict(arrowstyle='->'))
            
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title('Array Analysis')
            ax.legend()
            plt.show()

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
    def __init__(self, root):
        self.root = root
        self.root.attributes('-fullscreen', True)
        
        self.image_path = None
        self.original_image = None
        self.cropped_images = []
        self.current_region = 0
        self.selected_region = None
        self.analyzed_data = None
        
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.button_open = tk.Button(self.root, text="Open Image", command=self.open_image)
        self.button_open.pack(side=tk.TOP, padx=10, pady=10)
        
        self.button_previous = tk.Button(self.root, text="Previous", command=self.show_previous_region, state=tk.DISABLED)
        self.button_previous.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.button_next = tk.Button(self.root, text="Next", command=self.show_next_region, state=tk.DISABLED)
        self.button_next.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.button_choose_region = tk.Button(self.root, text="Choose Region", command=self.choose_region, state=tk.DISABLED)
        self.button_choose_region.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.button_analyze_plot = tk.Button(self.root, text="Analyze Plot", command=self.analyze_plot, state=tk.DISABLED)
        self.button_analyze_plot.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.button_download = tk.Button(self.root, text="Download Data", command=self.download_data, state=tk.DISABLED)
        self.button_download.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.cropped_images = detect_lateral_flow_tests(self.original_image)
            self.current_region = 0
            self.selected_region = None
            
            self.update_image()
            
            # Enable buttons
            self.button_previous.config(state=tk.NORMAL)
            self.button_next.config(state=tk.NORMAL)
            self.button_choose_region.config(state=tk.NORMAL)
            self.button_analyze_plot.config(state=tk.DISABLED)
            self.button_download.config(state=tk.DISABLED)


    def update_image(self):
        if self.original_image is not None:
            region = self.cropped_images[self.current_region] if self.cropped_images else self.original_image
            image = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            image = self.resize_image(image)
            self.display_image(image)

    def resize_image(self, image):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        image.thumbnail((screen_width, screen_height), Image.ANTIALIAS)
        return image
    
    def display_image(self, image):
        self.canvas.delete("all")
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo

    # def process_image(self, file_path):
    #     global min_maxc
    #     global min_maxr
    #     global image_obj
    #     global angle_threshold

    #     angle_threshold = 10
    #     # Only consider jpg and jpeg for now
    #     image_obj = io.imread(file_path, as_gray=True)

    #     # Find width (in pixels) of the current image
    #     width = len(image_obj[0])

    #     # Use width to calculate the min size (in pixels) a LFA could be (this heavily depends on how the images are taken and will probably have to be reworked)
    #     min_maxc = (int)(width / 40)
    #     min_maxr = (int)(min_maxc * 20)

    #     # Make a list for the test strip regions within the image
    #     test_strips = []
    #     detect_lateral_flow_tests(image_obj, test_strips)
    #     test_strips = remove_outliers(test_strips)

    #     # Go though test strips and make a new list of images of just the test strips
    #     test_strip_images = []
    #     for test in test_strips:
    #         minr, minc, maxr, maxc = test.bbox
    #         # Adjust indecies so that they are always in the bounds of the image
    #         maxr, maxc = min(max(maxr, minr + min_maxr), len(image_obj)), min(max(maxc, minc + min_maxc), len(image_obj[0]))
    #         test_strip_images.append([minr, maxr, minc, maxc])

    #     # Run image_analysis funciton

    #     self.current_index = 0

    #     # Display results
    #     self.show_images(test_strip_images)
    
    def analyze_plot(self):
        if self.original_image is not None:
            if self.selected_region is not None:
                # Perform analysis on the selected region
                x, y, w, h = self.selected_region
                region = self.cropped_images[self.current_region][y:y + h, x:x + w]
            else:
                region = self.cropped_images[self.current_region]
            # Perform analysis function on the region
            self.analyzed_data = image_analysis(region)  # Replace with your own analysis function
            
            # Show analyzed data
            self.display_analyzed_data()
            
            # Enable buttons
            self.button_download.config(state=tk.NORMAL)



    # def show_result(self):
    #     result_image = self.results[self.current_index][0]
    #     result_array = self.results[self.current_index][1]
    #     result_params = self.results[self.current_index][2]

    #     image_display_width = self.window_width // 4
    #     image_display_height = self.window_height - 200

    #     image_display_ratio = min(image_display_width / result_image.shape[1], image_display_height / result_image.shape[0])
    #     display_width = int(result_image.shape[1] * image_display_ratio)
    #     display_height = int(result_image.shape[0] * image_display_ratio)

    #     result_image = np.uint8(result_image)
    #     result_image = ImageTk.PhotoImage(image=Image.fromarray(result_image).resize((display_width, display_height)))

    #     self.image_label.configure(image=result_image)
    #     self.image_label.image = result_image

    #     self.plot_label.pack_forget()
    #     self.plot_label = tk.Label(self.window)
    #     self.plot_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    #     if len(result_params) == 2:
    #         # Get parameters from result_params

    #     canvas = FigureCanvasTkAgg(fig, master=self.plot_label)
    #     canvas.draw()
    #     canvas_widget = canvas.get_tk_widget()
    #     canvas_widget.pack(fill=tk.BOTH, expand=True)
    #     plt.close(fig)

    #     self.window.update_idletasks()
    
    def show_previous_region(self):
        self.current_region = (self.current_region - 1) % len(self.cropped_images)
        self.update_image()
    
    def show_next_region(self):
        self.current_region = (self.current_region + 1) % len(self.cropped_images)
        self.update_image()

    def choose_region(self):
        if self.original_image is not None:
            if self.cropped_images is not None:
                region = cv2.selectROI(self.cropped_images[self.current_region])
            else:
                region = cv2.selectROI(self.original_image)
            self.selected_region = region
            x, y, w, h = region
            cv2.rectangle(self.cropped_images[self.current_region], (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.update_image()
            
            # Disable/Enable buttons
            self.button_choose_region.config(state=tk.DISABLED)
            self.button_analyze_plot.config(state=tk.NORMAL)
            self.button_download.config(state=tk.DISABLED)
    
    def download_data(self):
        if self.analyzed_data is not None:
            file_path = filedialog.asksaveasfilename(defaultextension='.csv')
            if file_path:
                with open(file_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Data'])
                    for data_row in self.analyzed_data:
                        writer.writerow([data_row])
    
    def close(self):
        self.root.destroy()

def test_function():
    global min_maxr, min_maxc
    global angle_threshold
    global file_path
    global image_obj
    file_path = "images/image1.jpg"
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

    result = image_analysis(test_strip_images)

root = tk.Tk()
gui = MyGUI(root)
root.mainloop()

# start_index = result_params[0][0]
#             end_index = result_params[0][1]
#             area = result_params[0][2]

#             second_start_index = result_params[1][0]
#             second_end_index = result_params[1][1]
#             second_area = result_params[1][2]

#             fig = plt.figure(figsize=(6, 4))
#             ax = fig.add_subplot(111)

#             ax.plot(result_array, label='Original Array')
#             ax.scatter(start_index, result_array[start_index], color='red', label='Start Index (First Peak)')
#             ax.scatter(end_index, result_array[end_index], color='green', label='End Index (First Peak)')
#             ax.plot([start_index, end_index], [result_array[start_index], result_array[end_index]], 'k--',
#                     label='Diagonal Line (First Peak)')

#             ax.scatter(second_start_index, result_array[second_start_index], color='purple',
#                         label='Start Index (Second Peak)')
#             ax.scatter(second_end_index, result_array[second_end_index], color='orange',
#                         label='End Index (Second Peak)')
#             ax.plot([second_start_index, second_end_index],
#                     [result_array[second_start_index], result_array[second_end_index]], 'k--',
#                     label='Diagonal Line (Second Peak)')
#             ax.annotate(f'Area (Peak {1}): {area:.2f}', xy=(start_index, result_array[start_index]),
#                         xytext=(start_index, result_array[start_index] + 1), arrowprops=dict(arrowstyle='->'))
#             ax.annotate(f'Area (Peak {2}): {second_area:.2f}', xy=(second_start_index, result_array[second_start_index]),
#                         xytext=(second_start_index, result_array[second_start_index] + 1), arrowprops=dict(arrowstyle='->'))
            
#             ax.set_xlabel('Index')
#             ax.set_ylabel('Value')
#             ax.set_title('Array Analysis')
#             ax.legend()