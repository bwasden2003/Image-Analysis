# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 10:41:05 2018

@author: Binh
"""
import matplotlib 
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage import util 
from skimage.transform import rotate

from skimage.color import rgb2gray
from skimage.measure import profile_line
from skimage.exposure import rescale_intensity,equalize_hist
import tkinter as tk    

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import font
from tkinter import filedialog
from scipy.signal import chirp, find_peaks, peak_widths
from scipy import sparse
from scipy.sparse.linalg import spsolve


# =============================================================================
# 
#"Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens 
#<There are two parameters: p for asymmetry and λ for smoothness.
# Both have to be tuned to the data at hand. 
#We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with
# positive peaks) and 10^2 ≤ λ ≤ 10^9 , but exceptions may occur.
# In any case one should vary λ on a grid that is approximately linear for log λ>>
# =============================================================================
def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z


   
def strip_detection():
    global p98,count,strip,r,image,wid,fig1,ax1_2,threshfactor ,minArea
    global dimension,imageArt ,boxArt, stripCount,label_image,img_rescale

    settingChange()
    colorModeChange()

        
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = rescale_intensity(image,in_range=(p2, p98))
#    image=equalize_hist(image, nbins=256, mask=None)
    ax1_2 = fig1.add_subplot(2,1,2)
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh*(threshfactor/100), square(3))


    # remove artifacts connected to image border
    cleared = clear_border(bw)
    
    # label image regions
    label_image = label(cleared)

    strip=np.zeros((50,4))
    
    vcount=0
    hcount=0
    count=0
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= minArea:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
#            store boxes location to strip array :topr,topc,width,length
            strip[count,:]= [ minr, minc, maxc - minc,  maxr - minr]
            if (maxc - minc)>(maxr - minr):
                hcount+=1
            else:
                vcount+=1

            count+=1
       
    plot_boxes()
    if count>0:
        getProfileBtn.config(state=tk.NORMAL)
        reanalyzeBtn.configure(state=tk.NORMAL)
        analyzeBtn.configure(state=tk.DISABLED)
        
        lengthenBtn.configure(state=tk.NORMAL)
        shortenBtn.configure(state=tk.NORMAL)
    
        matchLengthBtn.configure(state=tk.NORMAL)
        widthFracScale.configure(state=tk.NORMAL)
        lambdaScale.configure(state=tk.NORMAL)
        pScale.configure(state=tk.NORMAL)
        stripSelSpn.configure(state=tk.NORMAL)
        upBtn.configure(state=tk.NORMAL)
        downBtn.configure(state=tk.NORMAL)
        leftBtn.configure(state=tk.NORMAL)
        rightBtn.configure(state=tk.NORMAL)
        alignTopBtn.configure(state=tk.NORMAL)
        alignBottomBtn.configure(state=tk.NORMAL)
        shrinkBtn.configure(state=tk.NORMAL)
        expandBtn.configure(state=tk.NORMAL)
        matchWidthBtn.configure(state=tk.NORMAL)
        prominenceScale.configure(state=tk.NORMAL)
        relHeightScale.configure(state=tk.NORMAL)
        distanceScale.config(state=tk.NORMAL)
        addBoxBtn.configure(state=tk.NORMAL)
        
    
        

def plot_boxes():
    global count,strip,r,img_rescale,wid,fig1,ax1_2,threshfactor ,minArea,imageArt, boxArt, stripCount,label_image
    ax1_2.imshow(img_rescale)   
#    print(img_rescale)
    strip=strip[strip[:count,1].argsort()]

    for i in range (0,count):
            rect = mpatches.Rectangle((strip[i,1], strip[i,0]), strip[i,2], strip[i,3],
                                      fill=False, edgecolor='red', linewidth=2)
            
            rects=ax1_2.add_patch(rect)
            
# =============================================================================
            dr = DraggableRectangle(rects)
            dr.connect()
            drs.append(dr)
 
# =============================================================================
#    ax1_2.set_axis_off()
    fig1.canvas.draw()


    fig1.show()
    stripCount.set(count)

    return strip, count

def get_Profiles():
    global strip, count,image, fig2,ax2,layout, intensityMode,widthFraction,backgroundOpt,maxlength
    global profiles,peakHeight,peakLocation,peakArea,profile_size,prominenceVal,distanceVal,p98,SampleIDs
    fig2,ax2=plt.subplots(nrows=count+1,figsize=(20,10))
#    fig2.canvas.manager.window.wm_geometry("+%d+%d" % (700, 0))
    settingChange()
    intensityModeChange()
    intensityMode=bool(intensityOpt.get())

    plt.ion()
    
    if (intensityMode):
        image=util.invert(image)
    
    maxlength=np.nanmax(strip[:count,3:],axis=0)
    averagedim=(np.average(strip[:count,2:],axis=0))
    profile_size=int(maxlength*1.1)
    profiles=np.zeros((count,profile_size))
    peakLocation=np.zeros((count,profile_size))
    peakHeight=np.zeros((count,profile_size))
    peakArea=np.zeros((count,profile_size))
    adjmaxloc=np.zeros(count)
    maxIntensity=0
    
    SampleIDs=str(SampleIDsEntry.get()).split(',')
    listLength=len(SampleIDs)
    if listLength<count:
        for ind in range(listLength,count):
            SampleIDs.append("Strip "+str(ind+1))
    for i in range(0,count): 

        ax2[i] = plt.subplot(((count-1)//4+1),4,i+1)
        ystart=int(strip[i,0])
        xstart=int(strip[i,1]+strip[i,2]/2)
        yend=strip[i,0]+strip[i,3]
        xend=int(strip[i,1]+strip[i,2]/2)
        striplength=int(yend-ystart)
    
        profiledata=(profile_line(image, (ystart,xstart),
               (yend,xend),linewidth=int(strip[i,2]*widthFraction)))
        profiles[i,:np.shape(profiledata)[0]] =  profiledata  

# =============================================================================
        currentMaxI=np.amax(profiles)
        if (currentMaxI>maxIntensity):
            maxIntensity=currentMaxI
#        profile=profiledata
        reg_of_interest=profiles[i,:]
        
        base=baseline_als(reg_of_interest[:striplength],lambdaPara, pPara,)
#            subplotpos=111
#            ax = fig2.add_subplot()
        ax2[i].plot(reg_of_interest)
        ax2[i].plot(base)
#        if (backgroundOpt):
#            reg_of_interest=np.subtract(reg_of_interest,base)
#            base=np.subtract(base,base)
        peaks, _ = find_peaks(reg_of_interest,prominence= prominenceVal, distance=distanceVal)
    #    results_half = peak_widths(profile[i], peaks, rel_height=0.5)
    #    results_half[0]  # widths
        peakLocation[i,:np.shape(peaks)[0]]=peaks
        peakHeight[i,:np.shape(peaks)[0]]= reg_of_interest[peaks]
        
        results_full = peak_widths(reg_of_interest, peaks, rel_height=relHeight)
        results_full[0]  # widths

        #Plot signal, peaks and contour lines at which the widths where calculated
        for j in range(0,np.shape(peaks)[0]):
            area = np.trapz(reg_of_interest[int(peaks[j]-
                (results_full[0][j])/2):int(peaks[j]+(results_full[0][j])/2)])    
#            print("total area: ",area )
            area_under_base=np.trapz(base[int(peaks[j]-(results_full[0][j])/2)
                :int(peaks[j]+(results_full[0][j])/2)])
#            print("baseline area: ",area_under_base )

            area=area-area_under_base
            peakArea[i,j]= area
#            print("new  area: ",area )
        #    area_under_first_peak
    #        print("area_under peak", area_under_first_peak, "; area_under_base " , area_under_base )
            ax2[i].annotate(int(area), xy=(peaks[j], reg_of_interest[peaks[j]]), 
                 xytext=(peaks[j]+maxlength*0.050, reg_of_interest[peaks[j]]*1.075),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=1),
                 )

            ax2[i].plot(peaks, reg_of_interest[peaks], "x")
            ax2[i].hlines(*results_full[1:], color="C3")
    backgroundOpt=False
    for k in range(0,count): 
        ax2[k].set_ylim(0,maxIntensity*1.15)
        ax2[k].set_title(SampleIDs[k])
    exportBtn.configure(state=tk.NORMAL)
    fig2.tight_layout()


def open_Image():
    filename =  filedialog.askopenfilename(initialdir = "/",
        title = "Select file",
        filetypes = (("image files","*.jpg *.gif *.png *.tif"),
        ("all files","*.*")))
    if (len(filename)> 1):
        print(filename, " file is opened")
        minA.configure(state=tk.NORMAL)
        THfactor.configure(state=tk.NORMAL)
#        indexSpinner.configure(state=tk.NORMAL)
        resetSetting.configure(state=tk.NORMAL)
        analyzeBtn.configure(state=tk.NORMAL)
        intensityOpt.set(1)
        initialize(filename)

def initialize(filename):
    global fig1, ax1,strip,count,r, canvas,artist1,colorOpt,dimension,colorRdb,cmapls
    r = plt.imread(filename) 
#    print ("dimension ",len(r.shape)," datat type: ",r.dtype.name)
#    print(r)
    dimension=len(r.shape)
    if dimension>2:
            MODES = [
                ("Red      ", "0"),
                ("Green    ", "1"),
                ("Blue     ", "2"),
                ("Grayscale", "3"),
            ]
            colorOpt = tk.StringVar(window)
            colorOpt.set("3") # initializes
            startRow=0
            for text, mode in MODES:   
                colorRdb = tk.Radiobutton(window, text=text, command=colorModeChange,
                                variable=colorOpt, value=mode, indicatoron=0,font=appHighlightFont)
                colorRdb.grid(row=startRow+int(mode),column=3)
                

        
    count = 0
    strip=np.zeros((50,4))
    fig1 = plt.figure(figsize=(7,9))
#    fig1.canvas.manager.window.wm_geometry("+%d+%d" % (500, 0))
#    canvas = FigureCanvasTkAgg(fig1, master=window)
    
    plt.ion()
    ax1 = fig1.add_subplot(2,1,1)
    artist1=ax1.imshow(r)  
    fig1.show()
    addBoxBtn.configure(state=tk.NORMAL)

def re_Analyze():
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    strip_detection()
        
        
def settingChange(var=None):
    global threshfactor,minArea,colorIndex,intensityMode,widthFraction,relHeight
    global lambdaPara,pPara,selectedStrip,count,prominenceVal,distanceVal
    threshfactor=float(THfactor.get())
    minArea=int(minA.get()) 
    widthFraction=int(widthFracScale.get())/100
    prominenceVal=float(prominenceScale.get())
    distanceVal=int(distanceScale.get())
    pPara=10**(int(pScale.get()))
    lambdaPara=10**(int(lambdaScale.get()))
    selectedStrip=int(stripSelSpn.get())-1
    relHeight=float(relHeightScale.get())
    

        
def intensityModeChange():
    intensityMode=bool(intensityOpt.get())
    if (intensityMode):
        THfactor.set(50)
        minA.set(2000)
    else:
        THfactor.set(150)
        minA.set(5000)

        
def colorModeChange():   
    global image,r,artist1,ax1, dimension,cmapls,colorOpt,colorIndex
    if (dimension<3):
        image=r
        colorIndex=3
    elif (int(colorOpt.get()) < 3):
        colorIndex=int(colorOpt.get())
        image=r[:,:,colorIndex]
    else:
        image=np.dot(r[...,:3], [0.299, 0.587, 0.114])
        colorIndex=3
    
#    print(img_rescale)
    artist1.remove()
    artist1=ax1.imshow(image, cmap=cmapls[colorIndex])
    fig1.show()
    fig1.canvas.draw()

def rotateImage(): 
    global image,r,artist1,ax1, dimension,cmapls,colorOpt,colorIndex
    r=rotate(r,90, preserve_range=True)
    image=r
#    print(image.shape)
    artist1.remove()
    artist1=ax1.imshow(image)
    fig1.show()
    fig1.canvas.draw()
#    return image
    
def defaultsetting():
    THfactor.set(50)
    minA.set(500)
#    indexSpinner.invoke("buttondown")
#    indexSpinner.invoke("buttondown")
    intensityOpt.set(50)
    
    
def lengthen_Box ():
    global label_image,strip,count
    newLength=strip[:,3]*1.1
    addedLength=newLength-strip[:,3]
    strip[:,3]= newLength
    strip[:,0]= strip[:,0]-(addedLength/2)

#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()
    
    
    
    
def shorten_Box ():
    global label_image,strip,count
    newLength=strip[:,3]*0.9
    reducedLength=strip[:,3]-newLength
    strip[:,3]= newLength
    strip[:,0]= strip[:,0]+(reducedLength/2)
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()   
    
    
def shrink_Box ():
    global label_image,strip,count
    newWidth=strip[:,2]*0.9
    reducedWidth=strip[:,2]-newWidth
    strip[:,2]= newWidth
    strip[:,1]= strip[:,1]+(reducedWidth/2)
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()   
    
def expand_Box ():
    global label_image,strip,count
    newWidth=strip[:,2]*1.1
    addeddWidth=newWidth-strip[:,2]
    strip[:,2]= newWidth
    strip[:,1]= strip[:,1]-(addeddWidth/2)
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()       
    
    
def match_Length():
    global label_image,strip,count
    maxLength=np.max(strip[:,3])
#    print(maxLength)
    addedLength=maxLength-strip[:,3]
    strip[:,3]= maxLength
    strip[:,0]= strip[:,0]-(addedLength/2)

#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()
    
def match_Width():
    global label_image,strip,count
    maxWidth=np.max(strip[:,2])
#    print(maxLength)
    addedWidth=maxWidth-strip[:,2]
    strip[:,2]= maxWidth
    strip[:,1]= strip[:,1]-(addedWidth/2)

#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()
    
    
    
    
    
    
def move_Up():
    global label_image,strip,count
    strip[selectedStrip,0]= strip[selectedStrip,0]-(strip[selectedStrip,2]*0.5)
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()
    
    

def move_Down():
    global label_image,strip,count
    strip[selectedStrip,0]= strip[selectedStrip,0]+(strip[selectedStrip,2]*0.5)
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()

def move_Left():
    global label_image,strip,count
    strip[selectedStrip,1]= strip[selectedStrip,1]-(strip[selectedStrip,2]*0.1)
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()

def move_Right():
    global label_image,strip,count
    strip[selectedStrip,1]= strip[selectedStrip,1]+(strip[selectedStrip,2]*0.1)
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()


def alignTop():
    global label_image,strip,count
    strip[:,0]=np.min(strip[:,0])
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()
        
def alignBottom():
    global label_image,strip,count
    strip[:,0]=np.max(strip[:,0])
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()    
    
def add_Box():
    global label_image,strip,count
    maxWidth=np.max(strip[:,2])
    maxLength=np.max(strip[:,3])
    strip=np.insert(strip,0,[0,0,maxWidth,maxLength],axis=0)
    count=count+1
#    print(strip)
    for artist in plt.gca().lines + plt.gca().images + plt.gca().patches:
        artist.remove()
    plot_boxes()
    
    
    
    
    
def export_Profiles():
    global SampleIDs
    filenameSave =  filedialog.asksaveasfilename(initialdir = "/",
        title = "Select file",
        filetypes = (("text files","*.txt"),
        ("all files","*.*")))
    headernames=""
#    printarray= np.empty((profile_size,count*4), dtype='str')
    printarray=np.zeros((profile_size,count*4))
    for n in range(0, count):
        headernames=headernames + "Profile " + SampleIDs[n] + ',' + "Peak Location " +SampleIDs[n] + ','+ "Peak Height " + SampleIDs[n]  +  ','+ "Peak Area " + SampleIDs[n] + ','
        printarray[:,n*4:(n*4)+4]=np.column_stack((profiles[n,:].transpose(),
            peakLocation[n,:].transpose(),peakHeight[n,:].transpose(),
            peakArea[n,:].transpose()))
        
        
    if (len(filenameSave)> 1):
#        for l in range(0,count):
#        printarray[printarray==0]=''
        np.savetxt(filenameSave, printarray, fmt='%.2f' ,delimiter=',',header=headernames)

 
    
    
    
    
    
    
    
    
#def substract_BG():
#    global backgroundOpt,fig2
#    backgroundOpt = True
#    for artist in fig2.gca().lines:
#        artist.remove()
#    get_Profiles()
    
class DraggableRectangle:
    def __init__(self, rect):
        self.rect = rect
        self.press = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        global boxMoved
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.rect.axes: return

        contains, attrd = self.rect.contains(event)
        if not contains: return
        x0, y0 = self.rect.xy
        boxMoved=((np.where(strip[:,1]==x0)[0][0]))



        self.press = x0, y0, event.xdata, event.ydata
    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.rect.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #      (x0, xpress, event.xdata, dx, x0+dx))
        self.rect.set_x(x0+dx)
        self.rect.set_y(y0+dy)
#        self.rect.figure.canvas.draw()
        
        strip[boxMoved,0]=self.rect.get_xy()[1]
        strip[boxMoved,1]=self.rect.get_xy()[0]

    def on_release(self, event):
        global boxMoved

        
        'on release we reset the press data'
        self.press = None
        self.rect.figure.canvas.draw()


    def disconnect(self):
        'disconnect all the stored connection ids'
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)


def kill():
     plt.close("all")
     window.destroy()



cmapls=["Reds","Greens","Blues", "gray"]
backgroundOpt=False
count=0
drs = []
     
window = tk.Tk()  
window.title("LFA Analysis")
window.geometry("700x900+0+0")
#canvas = FigureCanvasTkAgg(fig1, master=window)
#plot_widget = canvas.get_tk_widget()
appHighlightFont = tk.font.Font(family='Helvetica', size=8)
#window.size(10,10)


openBtn=tk.Button(window,text="Open File",command=open_Image, font=appHighlightFont)
rotateBtn=tk.Button(window,text="Rotate Image",command=rotateImage, font=appHighlightFont)



intensityOpt = tk.IntVar(window)



Colorimetry=tk.Radiobutton(window, text="Colorimetry    ",command=intensityModeChange,variable=intensityOpt, value=1)
Luminescence=tk.Radiobutton(window, text="Luminescence",command=intensityModeChange, variable=intensityOpt, value=0)




colorIndLbl=tk.Label(window, text="Color Selection", font=appHighlightFont)
threshLbl=tk.Label(window, text="Threshold  ", font=appHighlightFont)
minALbl=tk.Label(window, text="Minimum Area", font=appHighlightFont)

THfactor = tk.Scale(window, from_=0, to=200, resolution=0.5, command=settingChange,orient=tk.HORIZONTAL)
THfactor.set(50)
THfactor.configure(state=tk.DISABLED)

minA = tk.Scale(window, from_=100, to=50000, command=settingChange,orient=tk.HORIZONTAL)
minA.set(5000)
minA.configure(state=tk.DISABLED)

widthFracScale = tk.Scale(window, from_=5, to=100, command=settingChange,orient=tk.HORIZONTAL)
widthFracScale.set(50)
widthFracScale.configure(state=tk.DISABLED)

resetSetting=tk.Button(window,text="Default",command=defaultsetting, 
    font=appHighlightFont)
resetSetting.configure(state=tk.DISABLED)

analyzeBtn=tk.Button(window,text="Analyze",command=strip_detection, 
    font=appHighlightFont)
analyzeBtn.configure(state=tk.DISABLED)

reanalyzeBtn=tk.Button(window, text="Reanalyze",command=re_Analyze,
    font=appHighlightFont)
reanalyzeBtn.configure(state=tk.DISABLED)

addBoxBtn=tk.Button(window, text="Add Box",command=add_Box,
    font=appHighlightFont)
addBoxBtn.configure(state=tk.DISABLED)

stripCount=tk.StringVar(window)
stripCount.set("")
stripCountLbl=tk.Label(window, textvariable=stripCount, font=appHighlightFont)

countLlb=tk.Label(window, text="Strips found:", font=appHighlightFont)


lengthenBtn=tk.Button(window, text = "Lengthen", command = lengthen_Box, 
    font=appHighlightFont)
lengthenBtn.configure(state=tk.DISABLED)

shortenBtn=tk.Button(window, text = "Shorten", command = shorten_Box, 
    font=appHighlightFont)
shortenBtn.configure(state=tk.DISABLED)

matchLengthBtn=tk.Button(window, text = "Match L", command = match_Length, 
    font=appHighlightFont)
matchLengthBtn.configure(state=tk.DISABLED)

shrinkBtn=tk.Button(window, text= "Shrink", command =shrink_Box,
                    font=appHighlightFont )
shrinkBtn.configure(state=tk.DISABLED)

expandBtn=tk.Button(window, text= "Expand", command =expand_Box,
                    font=appHighlightFont )
expandBtn.configure(state=tk.DISABLED)

matchWidthBtn=tk.Button(window, text = "Match W", command = match_Width, 
    font=appHighlightFont)
matchWidthBtn.configure(state=tk.DISABLED)

alignTopBtn=tk.Button(window, text="Align Top",command=alignTop, font=appHighlightFont)
alignTopBtn.configure(state=tk.DISABLED)
alignBottomBtn=tk.Button(window, text="Align Bottom",command=alignBottom, font=appHighlightFont)
alignBottomBtn.configure(state=tk.DISABLED)


stripSelVar=tk.StringVar(window)
stripSelVar.set("1")
stripSelSpn=tk.Spinbox(window, from_=1, to=24,command=settingChange,width=4, textvariable=stripSelVar, font=appHighlightFont)
stripSelSpn.configure(state=tk.DISABLED)

upBtn=tk.Button(window, text = "  Up  ", command = move_Up, 
    font=appHighlightFont)
upBtn.configure(state=tk.DISABLED)

downBtn=tk.Button(window, text = "Down", command = move_Down, 
    font=appHighlightFont)
downBtn.configure(state=tk.DISABLED)


leftBtn=tk.Button(window, text = "   Left   ", command = move_Left, 
    font=appHighlightFont)
leftBtn.configure(state=tk.DISABLED)

rightBtn=tk.Button(window, text = "  Right  ", command = move_Right, 
    font=appHighlightFont)
rightBtn.configure(state=tk.DISABLED)

tk.Label(window, text="SampleIDs       ", font=appHighlightFont).grid(row=24,column=2,columnspan=2)
tk.Label(window, text="Profile Width(%)", font=appHighlightFont).grid(row=25,column=2,columnspan=2)
tk.Label(window, text="Prominence      ", font=appHighlightFont).grid(row=26,column=2,columnspan=2)
tk.Label(window, text="Distance        ", font=appHighlightFont).grid(row=27,column=2,columnspan=2)
tk.Label(window, text="Relative Height ", font=appHighlightFont).grid(row=28,column=2,columnspan=2)
tk.Label(window, text="P, Asymmetry    ", font=appHighlightFont).grid(row=29,column=2,columnspan=2)
tk.Label(window, text="L, Smoothness   ", font=appHighlightFont).grid(row=30,column=2,columnspan=2)

#We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with
# positive peaks) and 10^2 ≤ λ ≤ 10^9 ,


SampleIDsEntry = tk.Entry(window,width=30, font=appHighlightFont)
SampleIDsEntry.grid(row=24,column=3,columnspan=5)


prominenceScale = tk.Scale(window, from_=0, to=100, resolution=0.5,command=settingChange,orient=tk.HORIZONTAL)
prominenceScale.set(1)
prominenceScale.configure(state=tk.DISABLED)

distanceScale = tk.Scale(window, from_=0, to=500, command=settingChange,orient=tk.HORIZONTAL)
distanceScale.set(20)
distanceScale.configure(state=tk.DISABLED)

relHeightScale=tk.Scale(window, from_=0.1, to=1.0, resolution=0.05, command=settingChange,orient=tk.HORIZONTAL)
relHeightScale.set(1)
relHeightScale.configure(state=tk.DISABLED)



pScale = tk.Scale(window, from_=-3, to=-1, resolution=0.1,command=settingChange,orient=tk.HORIZONTAL)
pScale.set(-2)
pScale.configure(state=tk.DISABLED)

lambdaScale = tk.Scale(window, from_=2, to=9, resolution=0.1, command=settingChange,orient=tk.HORIZONTAL)
lambdaScale.set(7)
lambdaScale.configure(state=tk.DISABLED)



getProfileBtn=tk.Button(window, text = "Get Profile", command = get_Profiles, 
    font=appHighlightFont)
getProfileBtn.configure(state=tk.DISABLED)

exportBtn=tk.Button(window, text = "Export Data", command = export_Profiles, 
    font=appHighlightFont)
exportBtn.configure(state=tk.DISABLED)

#plotBGsubstracBtn=tk.Button(window, text = "Plot Backgroud Adjusted", command = substract_BG, 
#    font=appHighlightFont)

exitBtn=tk.Button(window, text = "EXIT", command = kill, font=appHighlightFont)







openBtn.grid(row=0, column=1,columnspan=2)
rotateBtn.grid(row=1,column=1,columnspan=2)

resetSetting.grid(row=1, column=7,columnspan=2)


Colorimetry.grid(row=1,column=5)
Luminescence.grid(row=2,column=5)



colorIndLbl.grid(row=0,column=5)
threshLbl.grid(row=4,column=2)
minALbl.grid(row=5,column=2)
THfactor.grid(row=4,column=3,columnspan=5)
minA.grid(row=5,column=3,columnspan=5)

analyzeBtn.grid(row=9, column=1,columnspan=2)

reanalyzeBtn.grid(row=9, column=5,columnspan=1)

addBoxBtn.grid(row=9, column=7,columnspan=1)
stripCountLbl.grid(row=10,column=3)
countLlb.grid(row=9,column=3)

lengthenBtn.grid(row=12, column=2,columnspan=1)
shortenBtn.grid(row=13, column=2,columnspan=1)
matchLengthBtn.grid(row=14, column=2,columnspan=1)

shrinkBtn.grid(row=12,column=3,columnspan=1)
expandBtn.grid(row=13,column=3,columnspan=1)
matchWidthBtn.grid(row=14, column=3,columnspan=1)


alignTopBtn.grid(row=12, column=7,columnspan=1)
alignBottomBtn.grid(row=13, column=7,columnspan=1)


stripSelSpn.grid(row=11,column=5,columnspan=2)

upBtn.grid(row=12, column=5,columnspan=2)
downBtn.grid(row=14, column=5,columnspan=2)
leftBtn.grid(row=13, column=5,columnspan=1)
rightBtn.grid(row=13, column=6,columnspan=1)





widthFracScale.grid(row=25,column=5,columnspan=8)
prominenceScale.grid(row=26,column=5,columnspan=8)
distanceScale.grid(row=27,column=5,columnspan=8)
relHeightScale.grid(row=28,column=5,columnspan=8)
pScale.grid(row=29,column=5,columnspan=8)
lambdaScale.grid(row=30,column=5,columnspan=8)


getProfileBtn.grid(row=32, column=3,columnspan=2)

exportBtn.grid(row=33, column=3,columnspan=2)


exitBtn.grid(row=34, column=3,columnspan=2)
window.mainloop()
    


if __name__ == '__main__':
    import sys
from PyQt5 import QtWidgets
if not QtWidgets.QApplication.instance():
    app = QtWidgets.QApplication(sys.argv)
else:
    app = QtWidgets.QApplication.instance() 

app.exec_()
