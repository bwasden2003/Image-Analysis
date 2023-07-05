# Image-Analysis
Image Analysis of LFAs Project
This project uses skimage, tkinter, numpy, and pandas in order to easily perform image analysis of lateral flow assays.
Running the code opens up a gui made with tkinter that allows the user to choose an image for analysis (the image is expected to contain multiple LFAs).
The code then detects the LFAs within the image and separates them into separate images for ease of use. The LFA images can be swapped between using the next and previous buttons and deleted if the code detects anything it shouldn't.
Each LFA can be analyzed using the analyze plot button which uses numpy to turn the image in to a numpy array to analyze for peaks by searching for the two largest peaks (representing positive signals).
If the user would like to analyze a smaller region of the LFA, the select region button can be used to do so.
The plots of the LFAs can be downloaded (in csv format) with the download button. The output excel sheet contains the numpy array values as well as the output plot and the original image.
