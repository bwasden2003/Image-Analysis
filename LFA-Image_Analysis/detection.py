import os

import numpy as np
import pandas as pd

import cv2
import tifffile as tf

import platform

import skimage as sk
from skimage import io, color, filters, restoration
from skimage.measure import label, regionprops
from skimage import morphology

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import xlsxwriter

image_obj = None
min_maxr, min_maxc = 0, 0
angle_threshold = 0
file_path = ""


def detect_lateral_flow_tests(image, line_length=10):
    if image.ndim == 2:
        # Increase gamma of the 2D grayscale image
        image_gray = image
        image_copy = sk.exposure.adjust_gamma(image_gray, 0.4)

        # Create a binary mask based on color thresholding
        threshold = filters.threshold_otsu(image_copy)
        threshold_image = image_copy > threshold
    elif image.ndim == 3:
        # Increase gamma of each channel in the 3D color image
        image_gray = sk.color.rgb2gray(image)
        image_copy = sk.exposure.adjust_gamma(image_gray, 0.4)

        # Create a binary mask based on color thresholding
        threshold = filters.threshold_otsu(image_copy)
        threshold_image = image_copy > threshold
    else:
        raise ValueError("Invalid image dimensions. Only 2D and 3D images are supported.")

    edges = sk.feature.canny(threshold_image)

    # Perform Hough line detection
    h, theta, d = sk.transform.hough_line(edges)

    # Find the most prominent lines
    _, angles, dists = sk.transform.hough_line_peaks(h, theta, d)

    # Create eroded and dilated versions
    eroded = morphology.binary_erosion(
        threshold_image, morphology.rectangle(line_length, 1))
    dilated = morphology.binary_dilation(
        eroded, morphology.rectangle(line_length, 1))

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
        if image.ndim == 2:
            mean_color = np.mean(image[minr:maxr, minc:maxc])
        else:
            mean_color = np.mean(image[minr:maxr, minc:maxc, :], axis=(0, 1))
        # Check if the mean color is lighter than the background color and the region is big enough to qualify as a test strip
        if np.mean(mean_color) > np.mean(background_color) and region.area >= 100:
            # Draw a rectangle around the detected test
            if check_duplicate(minr, minc, maxr, maxc, lst):
                straighten_region(copy_image, region, angles)
                lst.append([minr, maxr, minc, maxc])
    image_copy = copy_image
    lst = remove_outliers(lst)
    lfa_images = []
    sorted_regions = sorted(lst, key=lambda region: (region[2], region[3]))
    for region in sorted_regions:
        minr, maxr, minc, maxc = region[0], region[1], region[2], region[3]
        print(f"x1: {minc}, x2: {maxc}")
        lfa_image = image[minr:maxr, minc:maxc]
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
        rotated_region = sk.transform.rotate(
            copy_image[minr:maxr, minc:maxc], -region_angle, center=(center_row - minr, center_col - minc))
        # Replace the rotated region in the copy of the original image
        copy_image[minr:maxr, minc:maxc] = rotated_region


def check_duplicate(minr, minc, maxr, maxc, lst) -> bool:
    # Iterate through tests that have already been identified
    for region in lst:
        cur_minr, cur_maxr, cur_minc, cur_maxc = region[0], region[1], region[2], region[3]
        cur_maxr, cur_maxc = max(
            cur_maxr, cur_minr + min_maxr), max(cur_maxc, cur_minc + min_maxc)
        # Check whether the current region's dimensions overlap
        if (minr <= cur_maxr and maxr >= cur_minr and minc <= cur_maxc and maxc >= cur_minc):
            return False
    return True


def draw_rectangle(minr, minc, maxr, maxc):
    # Create a rectangle patch
    rect = patches.Rectangle((minc, minr), maxc - minc, maxr -
                             minr, linewidth=2, edgecolor='r', facecolor='none')
    # Add the rectangle to the current plot
    plt.gca().add_patch(rect)


def remove_outliers(lst):
    min_width = float('inf')
    new_lst = []
    for region in lst:
        cur_minr, cur_maxr, cur_minc, cur_maxc = region[0], region[1], region[2], region[3]
        cur_maxc = max(cur_maxc, cur_minc + min_maxc)
        width = cur_maxc - cur_minc
        if width < min_width:
            min_width = width
    for region in lst:
        cur_minr, cur_maxr, cur_minc, cur_maxc = region[0], region[1], region[2], region[3]
        cur_maxc = max(cur_maxc, cur_minc + min_maxc)
        width = cur_maxc - cur_minc
        if width <= (min_width * 2):
            new_lst.append(region)
    return new_lst


def detect_lines(image) -> list:

    # found that the strips vary from about 2% of the length of the image to like 4%
    min_area = len(image[0]) * (len(image) * 0.0065)
    max_area = len(image[0]) * (len(image) * 0.04)

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    image_copy = np.copy(image)
    image_copy = sk.filters.gaussian(image_copy)
    thresholded_image = sk.filters.threshold_local(
        image_copy, block_size=3, method='gaussian', mode='reflect')

    image_copy = image_copy * (thresholded_image > 0)

    # Adjust gamma
    image_copy = sk.exposure.adjust_gamma(image_copy, 0.5)

    enhanced_image = sk.exposure.equalize_adapthist(
        image_copy, clip_limit=0.05)

    # Set percentiles for range
    p80, p97 = np.percentile(enhanced_image, (80, 99))
    enhanced_image = sk.exposure.rescale_intensity(
        enhanced_image, in_range=(p80, p97), out_range=(0, 1))

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
    return strips



def image_analysis(region, selected, range = 15):

    image_copy = np.copy(region)
    if len(image_copy.shape) == 3:
        height, width, channels = image_copy.shape
        dim = (1, 2)
    else:
        height, width = image_copy.shape
        dim = 1

    if not selected:
        cropped_image = image_copy[int(height * .25):int(height * .6), int(width * .3):int(width * .7)]
    else:
        cropped_image = image_copy
    profile = np.nanmean(cropped_image, axis=dim)

    start_index, end_index, area = analyze_peaks(profile)
    second_peak_data = None
    if start_index > 1 and end_index < len(profile) - 2:
        prev_peak_data = profile[:start_index - range] if start_index - range > 0 else profile[:start_index - 1]
        next_peak_data = profile[end_index + range:] if end_index + range < len(profile) else profile[end_index + 1:]
        if len(prev_peak_data) > 1 and len(next_peak_data) > 1:
            next_peak_argmax = np.argmax(next_peak_data)
            prev_peak_argmax = np.argmax(prev_peak_data)
            if next_peak_data[next_peak_argmax] < prev_peak_data[prev_peak_argmax]:
                second_peak_data = prev_peak_data
                prev = True
            else:
                second_peak_data = next_peak_data
                prev = False
            if len(second_peak_data) > 1:
                second_start_index, second_end_index, second_area = analyze_peaks(second_peak_data)
                if not prev:
                    second_start_index += end_index + range
                    second_end_index += end_index + range
                if (second_start_index <= start_index and second_end_index > start_index) or (second_start_index <= end_index and second_end_index > end_index) or (second_start_index >= start_index and second_end_index <= end_index):
                    return [profile, [(start_index, end_index, area)]]
                else:
                    return [profile, [(start_index, end_index, area), (second_start_index, second_end_index, second_area)]]

    return [profile, [(start_index, end_index, area)]]

def inverse_image_analysis(region, selected, range = 1):
    image_copy = np.copy(region)
    height, width, channels= image_copy.shape

    if not selected:
        cropped_image = image_copy[int(height * .25):int(height * .6), int(width * .3):int(width * .7), :]
    else:
        cropped_image = image_copy
    profile = np.nanmean(cropped_image, axis=(1,2))

    start_index, end_index, area = analyze_peaks(profile, inverse=True)
    second_peak_data = None
    if start_index > 1 and end_index < len(profile) - 2:
        prev_peak_data = profile[:start_index - range]
        next_peak_data = profile[end_index + range:]
        if len(prev_peak_data) > 1 and len(next_peak_data) > 1:
            next_peak_argmin = np.argmin(next_peak_data)
            prev_peak_argmin = np.argmin(prev_peak_data)
            if next_peak_data[next_peak_argmin] > prev_peak_data[prev_peak_argmin]:
                second_peak_data = prev_peak_data
                prev = True
            else:
                second_peak_data = next_peak_data
                prev = False
            if len(second_peak_data) > 1:
                second_start_index, second_end_index, second_area = analyze_peaks(second_peak_data, inverse=True)
                if not prev:
                    second_start_index += end_index + 1
                    second_end_index += end_index + 1
                if (second_start_index <= start_index and second_end_index > start_index) or (second_start_index <= end_index and second_end_index > end_index) or (second_start_index >= start_index and second_end_index <= end_index):
                    return [profile, [(start_index, end_index, area)]]
                else:
                    return [profile, [(start_index, end_index, area), (second_start_index, second_end_index, second_area)]]

    return [profile, [(start_index, end_index, area)]]

def analyze_peaks(profile, inverse=False, window_size=25):
    peak_index = np.argmax(profile) if not inverse else np.argmin(profile)
    window_start = max(0, peak_index - window_size)
    window_end = min(len(profile), peak_index + window_size + 1)

    intensity_threshold = np.mean(profile[window_start:window_end])
    # Search backward to find the start index
    start_index = peak_index
    if inverse:
        while start_index > 0 and (np.all(profile[start_index] <= profile[start_index - 1]) or profile[start_index] < intensity_threshold):
            start_index -= 1
        # Search forward to find the end index
        end_index = peak_index
        while end_index < len(profile) - 1 and (np.all(profile[end_index] <= profile[end_index + 1]) or profile[end_index] < intensity_threshold):
            end_index += 1
    else: 
        while start_index > 0 and (np.all(profile[start_index] >= profile[start_index - 1]) or profile[start_index] > intensity_threshold):
            start_index -= 1
        # Search forward to find the end index
        end_index = peak_index
        while end_index < len(profile) - 1 and (np.all(profile[end_index] >= profile[end_index + 1]) or profile[end_index] > intensity_threshold):
            end_index += 1
    # Create a line segment connecting start_index and end_index
    modified_array = np.copy(profile)
    modified_array[start_index:end_index + 1] = np.linspace(profile[start_index], profile[end_index],
                                                            end_index - start_index + 1)
    if inverse:
        area = np.trapz(modified_array[start_index:end_index + 1]) - \
            np.trapz(profile[start_index:end_index + 1])
    else:
        area = np.trapz(profile[start_index:end_index + 1]) - \
            np.trapz(modified_array[start_index:end_index + 1])

    return start_index, end_index, area


class MyGUI:
    colors = ['red', 'green', 'blue']
    def __init__(self, root):
        self.root = root
        self.root.geometry(
            f"{self.get_screen_width()}x{self.get_screen_height()}")

        self.image_path = None
        self.original_image = None
        self.cropped_images = []
        self.current_region = 0
        self.selected_region = None
        self.analyzed_data = None
        self.select_window_open = False
        self.selected_profile = None
        self.selected_plot_img = None
        self.output_plot_img = None
        self.output_results = None
        self.num_tests = 0
        self.selected_color = None
        self.gamma_level = 1
        self.low_exposure = None
        self.high_exposure = None
        self.invert = tk.BooleanVar()

        open_frame = tk.Frame(self.root)
        open_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.button_open = tk.Button(
            open_frame, text="Open Image", command=self.open_image)
        self.button_open.pack(anchor=tk.CENTER)

        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.button_previous = tk.Button(
            button_frame, text="Previous", command=self.show_previous_region)
        self.button_previous.pack(side=tk.LEFT, padx=5)

        self.button_next = tk.Button(
            button_frame, text="Next", command=self.show_next_region)
        self.button_next.pack(side=tk.LEFT, padx=5)

        self.button_choose_gamma = tk.Button(button_frame, text="Adjust Gamma", command=self.adjust_gamma)
        self.button_choose_gamma.pack(side=tk.LEFT, padx=5)
        
        self.button_adjust_contrast = tk.Button(button_frame, text="Adjust Contrast", command=self.adjust_contrast)
        self.button_adjust_contrast.pack(side=tk.LEFT, padx=5)

        self.button_choose_region = tk.Button(
            button_frame, text="Choose Region", command=self.choose_region)
        self.button_choose_region.pack(side=tk.LEFT, padx=5)

        self.button_delete_region = tk.Button(
            button_frame, text="Delete Region", command=self.delete_region)
        self.button_delete_region.pack(side=tk.LEFT, padx=5)
        self.button_delete_region.config(state=tk.DISABLED)

        self.button_analyze_plot = tk.Button(
            button_frame, text="Analyze Plot", command=self.analyze_plot)
        self.button_analyze_plot.pack(side=tk.LEFT, padx=5)
        self.button_analyze_plot.config(state=tk.DISABLED)

        self.button_download = tk.Button(
            button_frame, text="Download Data", command=self.download_data)
        self.button_download.pack(side=tk.LEFT, padx=5)
        self.button_download.config(state=tk.DISABLED)

        info_frame = tk.Frame(self.root)
        info_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.num_regions_label = tk.Label(info_frame, text=("Number of regions: " + str(self.num_tests)), width = 20)
        self.num_regions_label.pack(side=tk.TOP)
    
        self.cur_region_label = tk.Label(info_frame, text=("Current region: " + str(self.current_region)), width = 20)
        self.cur_region_label.pack(side=tk.TOP)

        self.checkbox_invert = tk.Checkbutton(info_frame, text="Invert Image", variable=self.invert, onvalue=True, offvalue=False, command=self.update_image)
        self.checkbox_invert.pack()

        image_frame = tk.Frame(info_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(image_frame)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        self.plot_canvas = tk.Canvas(self.root)
        self.plot_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.output_results = tk.LabelFrame(self.plot_canvas)
        self.output_results.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def get_screen_width(self):
        if platform.system() == "Windows":
            try:
                from ctypes import windll
                user32 = windll.user32
                return user32.GetSystemMetrics(0)
            except ImportError:
                return 800  # Default width if unable to retrieve actual screen width
        elif platform.system() == "Darwin":
            try:
                from AppKit import NSScreen
                return int(NSScreen.mainScreen().frame().size.width)
            except ImportError:
                return 800  # Default width if unable to retrieve actual screen width
        else:
            return 800  # Default width for other platforms

    def get_screen_height(self):
        if platform.system() == "Windows":
            try:
                from ctypes import windll
                user32 = windll.user32
                return user32.GetSystemMetrics(1)
            except ImportError:
                return 600  # Default height if unable to retrieve actual screen height
        elif platform.system() == "Darwin":
            try:
                from AppKit import NSScreen
                return int(NSScreen.mainScreen().frame().size.height)
            except ImportError:
                return 600  # Default height if unable to retrieve actual screen height
        else:
            return 600  # Default height for other platforms

    def open_image(self):
        global min_maxc, min_maxr
        global angle_threshold
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if self.image_path:
            extension = os.path.splitext(self.image_path)[1]
            image = tf.imread(self.image_path) if extension in ['.tif', '.tiff'] else cv2.imread(self.image_path)

            self.original_image = image
            width = len(self.original_image[0])
            min_maxc = (int)(width / 40)
            min_maxr = (int)(min_maxc * 20)
            angle_threshold = 10

            self.cropped_images = detect_lateral_flow_tests(
                self.original_image)
            self.num_tests = len(self.cropped_images)
            self.num_regions_label.configure(text=("Number of regions: " + str(self.num_tests)))
            self.current_region = 0
            self.cur_region_label.configure(text=("Current region: " + str(self.current_region + 1)))
            self.selected_region = None

            self.update_image()

            # Enable buttons
            self.button_previous.config(state=tk.NORMAL)
            self.button_next.config(state=tk.NORMAL)
            self.button_choose_region.config(state=tk.NORMAL)
            self.button_delete_region.config(state=tk.NORMAL)
            self.button_analyze_plot.config(state=tk.NORMAL)
            self.button_download.config(state=tk.DISABLED)

    def update_image(self):
        if self.original_image is not None:
            region = self.cropped_images[self.current_region] if self.cropped_images else self.original_image
            if region.dtype != 'uint8':
                # Scale the values to 0-255 if they're not already
                region = (region - region.min()) / (region.max() - region.min()) * 255.0
                region = region.astype('uint8')
            image = np.copy(region)
            if self.selected_region is not None:
                x, y, w, h = self.selected_region
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if self.invert.get() == True:
                image = sk.util.invert(image)
            image = sk.exposure.adjust_gamma(image, self.gamma_level)
            if self.low_exposure is not None or self.high_exposure is not None:
                image = sk.exposure.rescale_intensity(image, in_range=(self.low_exposure, self.high_exposure))
            image = Image.fromarray(image)
            image = self.resize_image(image)
            self.display_image(image)

    def resize_image(self, image):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        image.thumbnail((screen_width, screen_height))
        return image

    def display_image(self, image):
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        img_width, img_height = image.size

        scale = min(canvas_width / img_width, canvas_height / img_height)
        display_width = int(img_width * scale)
        display_height = int(img_height * scale)

        resized_image = image.resize(
            (display_width, display_height))
        img_tk = ImageTk.PhotoImage(resized_image)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=img_tk)
        self.image_canvas.image = img_tk

    def analyze_plot(self):
        if self.original_image is not None:
            selected = False
            if self.selected_region is not None:
                # Perform analysis on the selected region
                x, y, w, h = self.selected_region
                region = self.cropped_images[self.current_region][y:y + h, x:x + w]
                selected = True
            else:
                region = self.cropped_images[self.current_region]
            
            region = sk.exposure.adjust_gamma(region, self.gamma_level)
            if self.low_exposure is not None or self.high_exposure is not None:
                region = sk.exposure.rescale_intensity(region, in_range=(self.low_exposure, self.high_exposure))
            if self.invert.get() == True:
                region = sk.util.invert(region)
                # Perform analysis function on the region
                self.analyzed_data = inverse_image_analysis(region, selected)
            else:
                self.analyzed_data = image_analysis(region, selected)
            # Returns [profile, [(start, end, area), ...]]
            # Show analyzed data
            self.display_analyzed_data()

            # Enable buttons
            self.button_download.config(state=tk.NORMAL)

    def display_analyzed_data(self):
        if self.analyzed_data is not None:
            self.selected_profile = self.analyzed_data[0]
            params = self.analyzed_data[1]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(self.selected_profile, label='Original Array')
            area = 0
            second_area = 0
            if len(params) > 1:
                first_params = params[0]
                start_index = first_params[0]
                end_index = first_params[1]
                area = first_params[2]

                second_params = params[1]
                second_start_index = second_params[0]
                second_end_index = second_params[1]
                second_area = second_params[2]

                ax.scatter(
                    start_index, self.selected_profile[start_index], color='red', label='Start Index (First Peak)')
                ax.scatter(
                    end_index, self.selected_profile[end_index], color='green', label='End Index (First Peak)')
                ax.plot([start_index, end_index], [self.selected_profile[start_index], self.selected_profile[end_index]], 'k--',
                        label='Diagonal Line (First Peak)')

                ax.scatter(second_start_index, self.selected_profile[second_start_index], color='purple',
                           label='Start Index (Second Peak)')
                ax.scatter(second_end_index, self.selected_profile[second_end_index], color='orange',
                           label='End Index (Second Peak)')
                ax.plot([second_start_index, second_end_index],
                        [self.selected_profile[second_start_index],
                            self.selected_profile[second_end_index]], 'k--',
                        label='Diagonal Line (Second Peak)')
                ax.annotate(f'Peak {1}', xy=(start_index, self.selected_profile[start_index]),
                            xytext=(start_index, self.selected_profile[start_index] + 1), arrowprops=dict(arrowstyle='->'))
                ax.annotate(f'Peak {2}', xy=(second_start_index, self.selected_profile[second_start_index]),
                            xytext=(second_start_index, self.selected_profile[second_start_index] + 1), arrowprops=dict(arrowstyle='->'))
            else:
                first_params = params[0]
                start_index = first_params[0]
                end_index = first_params[1]
                area = first_params[2]

                ax.scatter(
                    start_index, self.selected_profile[start_index], color='red', label='Start Index (First Peak)')
                ax.scatter(
                    end_index, self.selected_profile[end_index], color='green', label='End Index (First Peak)')
                ax.plot([start_index, end_index], [self.selected_profile[start_index], self.selected_profile[end_index]], 'k--',
                        label='Diagonal Line (First Peak)')
                ax.annotate(f'Peak {1}', xy=(start_index, self.selected_profile[start_index]),
                            xytext=(start_index, self.selected_profile[start_index] + 1), arrowprops=dict(arrowstyle='->'))

            ax.set_xlabel('Distance (Top to Bottom)')
            ax.set_ylabel('Intensity')
            ax.set_title('LFA Plot Analysis')
            # Convert the plot to an image
            self.selected_plot_img = self.plot_to_image(fig)

            canvas_width = self.plot_canvas.winfo_width()
            canvas_height = self.plot_canvas.winfo_height()

            self.plot_canvas.delete("all")
            self.plot_canvas.create_image(
                canvas_width // 2, canvas_height // 2, image=self.selected_plot_img)
            self.plot_canvas.image = self.selected_plot_img
            for label in self.output_results.winfo_children():
                label.destroy()
            label1 = tk.Label(self.output_results,
                              text="Area Peak 1: " + str(round(area, 2)))
            label1.pack()
            if second_area != 0:
                label2 = tk.Label(
                    self.output_results, text="Area Peak 2: " + str(round(second_area, 2)))
                label2.pack()
            self.output_results.pack(side=tk.BOTTOM, pady=50)

    def plot_to_image(self, plot):
        # Save the plot as a temporary image file
        temp_file = "temp_plot.png"
        plot.savefig(temp_file)

        # Load the saved image file as a Tkinter-compatible image
        self.output_plot_img = io.imread(temp_file)
        plot_img = ImageTk.PhotoImage(Image.open(temp_file))

        # Clean up the temporary file
        if platform.system() == "Windows":
            # On Windows, the file might not be immediately available for deletion
            self.root.after(100, lambda: self.delete_temp_file(temp_file))
        else:
            self.delete_temp_file(temp_file)

        return plot_img

    def delete_temp_file(self, filename):
        try:
            os.remove(filename)
        except:
            pass

    def show_previous_region(self):
        self.current_region = (self.current_region -
                               1) % len(self.cropped_images)
        self.cur_region_label.configure(text=("Current region: " + str(self.current_region + 1)))
        self.selected_region = None
        self.gamma_level = 1
        self.low_exposure, self.high_exposure = None, None
        self.update_image()

    def show_next_region(self):
        self.current_region = (self.current_region +
                               1) % len(self.cropped_images)
        self.cur_region_label.configure(text=("Current region: " + str(self.current_region + 1)))
        self.selected_region = None
        self.gamma_level = 1
        self.low_exposure, self.high_exposure = None, None
        self.update_image()

    def adjust_gamma(self):
        def save_result():
            self.gamma_level = slider.get() / 100
            self.update_image()
            window.destroy()
        window = tk.Toplevel()
        window.title("Adjust Gamma")
        slider = tk.Scale(window, variable=tk.DoubleVar(), from_=0, to=200, orient=tk.HORIZONTAL, resolution=1)
        slider.set(self.gamma_level * 100)
        button = tk.Button(window, text="Save", command=save_result)
        slider.pack(padx=20, pady=10)
        button.pack(pady=10)
        window.mainloop()

    def adjust_contrast(self):
        def save_result():
            self.low_exposure = scale_low.get()
            self.high_exposure = scale_high.get()
            self.update_image()
            window.destroy()
        window = tk.Toplevel()
        window.title("Adjust Exposure")
        scale_low = tk.Scale(window, variable=tk.DoubleVar(), from_=0, to=100, orient=tk.HORIZONTAL, label="Low Percentile", resolution=1)
        scale_low.set(self.low_exposure) if self.low_exposure is not None else 0
        scale_high = tk.Scale(window, variable=tk.DoubleVar(), from_=0, to=100, orient=tk.HORIZONTAL, label="High Percentile", resolution=1)
        scale_high.set(self.high_exposure) if self.high_exposure is not None else 100
        button = tk.Button(window, text="Save", command=save_result)
        scale_low.pack(padx=20, pady=10)
        scale_high.pack(padx=20, pady=10)
        button.pack(pady=10)
        window.mainloop()

    def choose_region(self):
        self.button_choose_region.config(state=tk.DISABLED)
        if self.original_image is not None:
            cv2.namedWindow("Select Region", cv2.WINDOW_NORMAL)
            
            if self.cropped_images is not None:
                image = self.cropped_images[self.current_region]
            else:
                image = self.original_image
            if image.dtype != np.uint8:
                image = (image - image.min()) / (image.max() - image.min()) * 255.0
                image = image.astype(np.uint8)
            region = cv2.selectROI("Select Region", image)
            self.selected_region = region if region != (0, 0, 0, 0) else None
            cv2.destroyWindow("Select Region")
            self.update_image()

            # Disable/Enable buttons
            self.button_choose_region.config(state=tk.NORMAL)
            self.button_analyze_plot.config(state=tk.NORMAL)
            self.button_download.config(state=tk.DISABLED)

    def delete_region(self):
        if len(self.cropped_images) > 0:
            del self.cropped_images[self.current_region]
            self.num_tests -= 1
            self.num_regions_label.configure(text=("Number of regions: " + str(self.num_tests)))
            if self.current_region >= len(self.cropped_images):
                self.current_region = len(self.cropped_images) - 1
                self.cur_region_label.configure(text=("Current region: " + str(self.current_region + 1)))
            self.update_image()

    def download_data(self):
        if self.analyzed_data is not None:
            file_path = filedialog.asksaveasfilename(filetypes=[('excel file', '*.xlsx')])
            if file_path:
                if not file_path.lower().endswith('.xlsx'):
                    file_path += '.xlsx'
                data = pd.DataFrame(self.selected_profile,
                                    columns=["Intensity"])
                writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
                data.to_excel(writer, sheet_name='Sheet1')

                workbook = writer.book
                worksheet = writer.sheets['Sheet1']

                cv2.imwrite(r'temp1.png', self.original_image)

                cv2.imwrite(
                    r'temp2.png', self.cropped_images[self.current_region])

                cv2.imwrite(r'temp3.png', self.output_plot_img)

                worksheet.insert_image('D3', 'temp1.png')
                worksheet.insert_image('M3', 'temp2.png')
                worksheet.insert_image('O3', 'temp3.png')
                writer.book.close()

                if platform.system() == "Windows":
                    # On Windows, the file might not be immediately available for deletion
                    self.root.after(
                        100, lambda: self.delete_temp_file('temp1.png'))
                else:
                    self.delete_temp_file('temp1.png')

                if platform.system() == "Windows":
                    # On Windows, the file might not be immediately available for deletion
                    self.root.after(
                        100, lambda: self.delete_temp_file('temp2.png'))
                else:
                    self.delete_temp_file('temp2.png')

                if platform.system() == "Windows":
                    # On Windows, the file might not be immediately available for deletion
                    self.root.after(
                        100, lambda: self.delete_temp_file('temp3.png'))
                else:
                    self.delete_temp_file('temp3.png')


    def close(self):
        self.root.destroy()


root = tk.Tk()
gui = MyGUI(root)
root.mainloop()