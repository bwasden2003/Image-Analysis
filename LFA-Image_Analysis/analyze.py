import imagej
import numpy as np
import matplotlib.pyplot as mp
import matplotlib.patches as patches
import skimage as sk

def __init__():
	global image_file, image_obj
	image_file = "images/image1.jpg"
	image_obj = sk.io.imread(image_file)
	histogram()

def histogram():
	global image_obj
	def plot_img_and_hist(image, axes, bins=256):
		ax_img, ax_hist = axes
		ax_cdf = ax_hist.twinx()

		# Display image
		ax_img.imshow(image, cmap=mp.cm.gray)
		ax_img.set_axis_off()

		# Display histogram
		ax_hist.hist(image.ravel(), bins=bins)
		ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
		ax_hist.set_xlabel('Pixel intensity')

		xmin, xmax = sk.util.dtype.dtype_range[image.dtype.type]
		ax_hist.set_xlim(xmin, xmax)

		# Display cumulative distribution
		img_cdf, bins = sk.exposure.cumulative_distribution(image, bins)
		ax_cdf.plot(bins, img_cdf, 'r')

		return ax_img, ax_hist, ax_cdf

	# prevent overlap of y-axis labels
	strip_detection()


def corner_detection():
	global image_obj
	coords = sk.feature.corner_peaks(sk.feature.corner_harris(image_obj), min_distance=5, threshold_rel=0.02)
	coords_subpix = sk.feature.corner_subpix(image_obj, coords, window_size=13)

	fig, ax = mp.subplots()
	ax.imshow(image_obj, cmap=mp.cm.gray)
	ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
			linestyle='None', markersize=6)
	ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
	ax.axis((0, 310, 200, 0))
	mp.show()

def edge_detection():
	global image_obj

	edges1 = sk.filters.sobel(image_obj)
	edges2 = sk.feature.canny(image_obj, sigma=3)

	# display results
	fig, ax = mp.subplots(nrows=1, ncols=3, figsize=(8, 3))

	ax[0].imshow(image_obj, cmap='gray')
	ax[0].set_title('noisy image', fontsize=20)

	ax[1].imshow(edges1, cmap='gray')
	ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

	ax[2].imshow(edges2, cmap='gray')
	ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

	for a in ax:
		a.axis('off')

	fig.tight_layout()
	mp.show()

def strip_detection():
	global image_file, image_obj
	strips = []

	threshold = sk.filters.threshold_otsu(image_obj)

	color_corrected_image = sk.morphology.closing(image_obj > threshold / 1.5, sk.morphology.square(3))
	cleared = sk.segmentation.clear_border(color_corrected_image)
	
	label_image = sk.measure.label(cleared)

	# to make the background transparent, pass the value of `bg_label`,
	# and leave `bg_color` as `None` and `kind` as `overlay`
	image_label_overlay = sk.color.label2rgb(label_image, image=image_obj, bg_label=0)

	fig, ax = mp.subplots(figsize=(10, 6))
	ax.imshow(image_label_overlay)

	for region in sk.measure.regionprops(label_image):
		# take regions with large enough areas
		if region.area >= 8:
			# draw rectangle around segmented coins
			minr, minc, maxr, maxc = region.bbox
			rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
									fill=False, edgecolor='red', linewidth=2)
			strips.append(rect)
	for strip in strips:
		ax.add_patch(strip)
	
	ax.set_axis_off()
	mp.tight_layout()
	mp.show()
	
	# run through image and grab all strips (have to figure out strip detection)
		# probably have to rotate strips to make sure they are uniform rectangles
	# grab just the control and test strip area
	# for image in images:
	# use imagej api to select area and the perform analysis

__init__()