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

# https://stackoverflow.com/questions/55636313/selecting-an-area-of-an-image-with-a-mouse-and-recording-the-dimensions-of-the-s

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


# Uses imageJ to analyze the control/test strip area
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

    SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='red',
                          outline='')
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Image Results")

        self.button = tk.Button(self.window, text="Open Image", command=self.open_image)
        self.button.pack()

        self.results = []
        self.current_index = -1

        # Create GUI components
        self.window_width = self.window.winfo_screenwidth()
        self.window_height = self.window.winfo_screenheight()
        self.window.geometry("%dx%d" % (self.window_width, self.window_height))

        self.image_label = tk.Label(self.window)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot_label = tk.Frame(self.window)
        self.plot_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(side=tk.TOP, pady=10)

        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(self.button_frame, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=10)


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
        self.analyze_test_strips(test_strip_images)

        self.current_index = 0

        # Display results
        # self.show_result()

    def analyze_test_strips(self, test_strip_images):
        for test_strip in test_strip_images:
            minr, maxr, minc, maxc = test_strip[0], test_strip[1], test_strip[2], test_strip[3]
            image = image_obj[minr:maxr, minc:maxc]
            image_display_width = self.window_width // 4
            image_display_height = self.window_height - 200

            image_display_ratio = min(image_display_width / image.shape[1], image_display_height / image.shape[0])
            display_width = int(image.shape[1] * image_display_ratio)
            display_height = int(image.shape[0] * image_display_ratio)

            result_image = np.uint8(image)
            result_image = ImageTk.PhotoImage(image=Image.fromarray(result_image).resize((display_width, display_height)))
            
            canvas = tk.Canvas(self.window, width = display_width, height = display_height, borderwidth=0, highlightthickness=0)
            canvas.pack(expand=True)
            canvas.create_image(0, 0, image=result_image, anchor=tk.NW)
            self.selection_obj = SelectionObject(canvas, self.SELECT_OPTS)

            def on_drag(start, end, **kwarg):
                self.selection_obj.update(start, end)

            self.posn_tracker = MousePositionTracker(canvas)
            self.posn_tracker.autodraw(command=on_drag)

    def show_result(self):
        result_image = self.results[self.current_index][0]
        result_array = self.results[self.current_index][1]
        result_params = self.results[self.current_index][2]

        image_display_width = self.window_width // 4
        image_display_height = self.window_height - 200

        image_display_ratio = min(image_display_width / result_image.shape[1], image_display_height / result_image.shape[0])
        display_width = int(result_image.shape[1] * image_display_ratio)
        display_height = int(result_image.shape[0] * image_display_ratio)

        result_image = np.uint8(result_image)
        result_image = ImageTk.PhotoImage(image=Image.fromarray(result_image).resize((display_width, display_height)))

        self.image_label.configure(image=result_image)
        self.image_label.image = result_image

        self.plot_label.pack_forget()
        self.plot_label = tk.Label(self.window)
        self.plot_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        if len(result_params) == 2:
            # Get parameters from result_params
            start_index = result_params[0][0]
            end_index = result_params[0][1]
            area = result_params[0][2]

            second_start_index = result_params[1][0]
            second_end_index = result_params[1][1]
            second_area = result_params[1][2]

            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)

            ax.plot(result_array, label='Original Array')
            ax.scatter(start_index, result_array[start_index], color='red', label='Start Index (First Peak)')
            ax.scatter(end_index, result_array[end_index], color='green', label='End Index (First Peak)')
            ax.plot([start_index, end_index], [result_array[start_index], result_array[end_index]], 'k--',
                    label='Diagonal Line (First Peak)')

            ax.scatter(second_start_index, result_array[second_start_index], color='purple',
                        label='Start Index (Second Peak)')
            ax.scatter(second_end_index, result_array[second_end_index], color='orange',
                        label='End Index (Second Peak)')
            ax.plot([second_start_index, second_end_index],
                    [result_array[second_start_index], result_array[second_end_index]], 'k--',
                    label='Diagonal Line (Second Peak)')
            ax.annotate(f'Area (Peak {1}): {area:.2f}', xy=(start_index, result_array[start_index]),
                        xytext=(start_index, result_array[start_index] + 1), arrowprops=dict(arrowstyle='->'))
            ax.annotate(f'Area (Peak {2}): {second_area:.2f}', xy=(second_start_index, result_array[second_start_index]),
                        xytext=(second_start_index, result_array[second_start_index] + 1), arrowprops=dict(arrowstyle='->'))
            
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title('Array Analysis')
            ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_label)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

        self.window.update_idletasks()
    
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

class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas):
        self.canvas = canvas
        self.canv_width = self.canvas.cget('width')
        self.canv_height = self.canvas.cget('height')
        self.reset()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event)
        self._command(self.start, (event.x, event.y))  # User callback.

    def _update(self, event):
        # Update cross-hair lines.
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
        self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
        self.show()

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self, command=lambda *args: None):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self._command = command
        self.canvas.bind("<Button-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.update)
        self.canvas.bind("<ButtonRelease-1>", self.quit)

    def quit(self, event):
        self.hide()  # Hide cross-hairs.
        self.reset()


class SelectionObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, select_opts):
        # Create attributes needed to display selection.
        self.canvas = canvas
        self.select_opts1 = select_opts
        self.width = self.canvas.cget('width')
        self.height = self.canvas.cget('height')

        # Options for areas outside rectanglar selection.
        select_opts1 = self.select_opts1.copy()  # Avoid modifying passed argument.
        select_opts1.update(state=tk.HIDDEN)  # Hide initially.
        # Separate options for area inside rectanglar selection.
        select_opts2 = dict(dash=(2, 2), fill='', outline='white', state=tk.HIDDEN)

        # Initial extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = 0, 0,  1, 1
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.canvas.create_rectangle(omin_x, omin_y,  omax_x, imin_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imin_y,  imin_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(imax_x, imin_y,  omax_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imax_y,  omax_x, omax_y, **select_opts1),
            # Inner rectangle.
            self.canvas.create_rectangle(imin_x, imin_y,  imax_x, imax_y, **select_opts2)
        )

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], omin_x, omin_y,  omax_x, imin_y),
        self.canvas.coords(self.rects[1], omin_x, imin_y,  imin_x, imax_y),
        self.canvas.coords(self.rects[2], imax_x, imin_y,  omax_x, imax_y),
        self.canvas.coords(self.rects[3], omin_x, imax_y,  omax_x, omax_y),
        self.canvas.coords(self.rects[4], imin_x, imin_y,  imax_x, imax_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)

    def _get_coords(self, start, end):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1])))

    def hide(self):
        for rect in self.rects:
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)

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

gui = MyGUI()
gui.run()