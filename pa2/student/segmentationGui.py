# segmentationGui.py - Provides a GUI for the segmentation assignment
# Author: Jimmy Briggs (Spring 2018)
# Cornell University CS 4670/5670: Intro Computer Vision
import sys
import traceback
import numpy as np
import PIL
from PIL import Image, ImageTk, ImageDraw
import cv2

# Load correct GUI depending on version of python.
if sys.version_info.major >= 3: 
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    from tkinter import ttk, filedialog
    from tkinter import Text
    from importlib import reload 
    
else: # Python 2.7
    import Tkinter as tk
    from ScrolledText import ScrolledText
    import tkFileDialog  as filedialog
    import ttk
    from Tkinter import Text


    
class LogRedirector(object):
    def __init__(self,text_widget):
        self.text_space = text_widget

    def write(self,string):
        self.text_space.insert('end', string)
        self.text_space.see('end')
        
    def flush():
        self.text_space.insert('end', "\n")

class ImageWidget(tk.Canvas):
    '''This class represents a Canvas on which OpenCV images can be drawn.
       The canvas handles shrinking of the image if the image is too big,
       as well as writing of the image to files. '''

    def __init__(self, parent):
        self.imageCanvas = tk.Canvas.__init__(self, parent)
        self.originalImage = None
        self.bind("<Configure>", self.redraw)

    def convertCVToTk(self, cvImage):
        height, width = cvImage.shape[0:2]
        if height == 0 or width == 0:
            return 0, 0, None
        if len(cvImage.shape) > 2:
            img = Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_GRAY2RGB))
        return height, width, ImageTk.PhotoImage(img)

    def fitImageToCanvas(self, cvImage):
        height, width = cvImage.shape[0:2]
        if height == 0 or width == 0:
            return cvImage
        ratio = width / float(height)
        if self.winfo_height() < height:
            height = self.winfo_height()
            width = int(ratio * height)
        if self.winfo_width() < width:
            width = self.winfo_width()
            height = int(width / ratio)
        dest = cv2.resize(cvImage, (width, height),
            interpolation=cv2.INTER_LANCZOS4)
        return dest

    def drawCVImage(self, cvImage):
        self.originalImage = cvImage
        height, width, img = self.convertCVToTk(self.fitImageToCanvas(cvImage))
        if height == 0 or width == 0:
            return
        self.tkImage = img # prevent the image from being garbage collected
        self.delete("all")
        x = (self.winfo_width() - width) / 2.0
        y = (self.winfo_height() - height) / 2.0
        self.create_image(x, y, anchor=tk.NW, image=self.tkImage)

    def redraw(self, _):
        if self.originalImage is not None:
            self.drawCVImage(self.originalImage)

    def writeToFile(self, filename):
        if self.originalImage is not None:
            cv2.imwrite(filename, self.originalImage)
         
class imageManipulationFrame(tk.Frame):
    def __init__(self, parent):
        self.Frame = tk.Frame.__init__(self,parent)
        self.name = "Image manipulation template"
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.imageWidget = ImageWidget(self)
        self.imageWidget.grid(row=0, columnspan=2,sticky=tk.W+tk.E+tk.N+tk.S)
        
        self.loadButton = tk.Button(self, text="Open", command=self.imageLoadDialog)
        self.loadButton.grid(row=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
    
        self.saveButton = tk.Button(self, text="Save", command=self.imageSaveDialog)
        self.saveButton.grid(row=1, column=1, sticky=tk.W+tk.E+tk.N+tk.S)

        self.loadImage("../images/hexagon.jpg")
        
        # Extend this with the tools per frame
        self.toolFrame = tk.Frame(self)
        self.toolFrame.grid(row=2,columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)

    def resetFields():
        pass
    
    def loadImage(self, filename):
        self.image = cv2.imread(filename)
        self.imageWidget.drawCVImage(self.image)
    
    def imageLoadDialog(self): 
        try:
            self.filename = filedialog.askopenfilename(initialdir = "../images",title = "Open file",filetypes = (("png files","*.png"),("all files","*.*")))
            self.loadImage(self.filename)
            self.resetFields()
            print("Loaded " + self.filename)            
        except:
            print("Failed to open " + self.filename)
            
    def imageSaveDialog(self):
        f = filedialog.asksaveasfile(mode='w',initialdir = "../images",title = "Save file", defaultextension=".png")
        if f is None:
            return
        self.imageWidget.writeToFile(f.name)
        print("Saved " + f.name)
        
    def onVisibility(self, event):
        parent.title("segmentationGui.py |" + " " + self.name)
        
class edgeDetectFrame(imageManipulationFrame):
    def __init__(self, parent):
        self.imageManipulationFrame = imageManipulationFrame.__init__(self, parent)
        
        # Set up radio buttons and select the first one
        radioButtonLabels = ["Original", "x-Derivative", "y-Derivative", "Gradient Magnitude"]
        self.radioButtons = []
        self.selection = tk.IntVar()
        for i, label in enumerate(radioButtonLabels):
            self.radioButtons.append(tk.Radiobutton(self.toolFrame, 
               text=label,
               state=tk.ACTIVE,
               variable=self.selection,
               padx=20, 
               value=i,
               command = lambda : self.edgeDetect(self.selection)))
            self.radioButtons[-1].grid(row=2, column=i)        
        self.resetFields()
        
    def resetFields(self):
        self.radioButtons[0].select()
                 
    def edgeDetect(self,stateVar):
        state = stateVar.get()
        try:
            import segment; reload(segment)
            if state == 1:
                self.xDerivativeImage = segment.getDisplayGradient(segment.takeXGradient(self.image))
                self.imageWidget.drawCVImage(self.xDerivativeImage)
            elif state == 2:
                self.yDerivativeImage = segment.getDisplayGradient(segment.takeYGradient(self.image))
                self.imageWidget.drawCVImage(self.yDerivativeImage)
            elif state == 3:
                tempGradImage = segment.takeGradientMag(self.image)
                self.gradMagImage = segment.normalizeImage(tempGradImage, 0, np.max(tempGradImage), 0, 255).astype(np.uint8)  
                self.imageWidget.drawCVImage(self.gradMagImage)                
            else:
                self.imageWidget.drawCVImage(self.image)               
        except:
            traceback.print_exc()
            return

class kMeansSegmentationFrame(imageManipulationFrame):
    DEFAULT_K = 3;
    def __init__(self, parent):
        self.imageManipulationFrame = imageManipulationFrame.__init__(self,parent)
        
        self.toolFrame.grid_columnconfigure(0, weight=1)
        self.toolFrame.grid_columnconfigure(1, weight=1)
        
        self.useHsv = tk.BooleanVar(self)
        self.useHsvBox = tk.Checkbutton(self.toolFrame, text="Segment in HSV Space", variable=self.useHsv)
        self.useHsvBox.grid(row=0,column=0, sticky=tk.W+tk.E+tk.N+tk.S)
        
        self.kLabel = tk.Label(self.toolFrame, text="Number of Means:")
        self.kLabel.grid(row=0, column=1, sticky=tk.E+tk.N+tk.S)
        
        self.k = tk.StringVar(self)
        self.k.set(kMeansSegmentationFrame.DEFAULT_K)       
        self.kOptionMenu = ttk.Combobox(self.toolFrame, textvariable=self.k)
        self.kOptionMenu['values'] = (2,3,4,5,6,7,8,9)
        self.kOptionMenu.grid(row=0,column=2, sticky=tk.E+tk.N+tk.S)
        
        self.doSegment = tk.BooleanVar(self)
        self.segmentBox = tk.Checkbutton(self.toolFrame, text="Segment Image", variable=self.doSegment, command=lambda : self.kMeansSegment())
        self.segmentBox.grid(row=1,columnspan=3,sticky=tk.W+tk.E+tk.N+tk.S)
    
        self.resetFields()
    
    def resetFields(self):
        self.segmentBox.deselect()
    
    def kMeansSegment(self):
        if not self.doSegment.get():
            self.imageWidget.drawCVImage(self.image)
            return
        if not self.k.get().isdigit() or int(self.k.get()) < 1:
            self.k.set(kMeansSegmentationFrame.DEFAULT_K)
            print("Setting k to default value of " + str(kMeansSegmentationFrame.DEFAULT_K) + ".")
        
        iter = 0
        try:
            import segment; reload(segment)
            self.segmentedImage, iter = segment.kMeansSegmentation(self.image, int(self.k.get()), self.useHsv.get(), 0.01)
        except:
            traceback.print_exc()
            return
        print("k_Means finished in " + str(iter) + " iterations.")
        self.imageWidget.drawCVImage(self.segmentedImage)        
        

class normalizedCutsFrame(imageManipulationFrame):
    DEFAULT_SIGMA_X = 6
    DEFAULT_SIGMA_F = 5
    
    def __init__(self, parent):
        self.imageManipulationFrame = imageManipulationFrame.__init__(self,parent)
        self.loadImage('../images/hexagon-small.jpg')
        
        self.doSegment = tk.BooleanVar(self)
        self.segmentBox = tk.Checkbutton(self.toolFrame, text="Segment with Normalized Cuts", variable=self.doSegment, command=lambda : self.nCutSegment())
        self.segmentBox.grid(row=0,columnspan=2,sticky=tk.W+tk.E+tk.N+tk.S)
        
        self.xLabel = tk.Label(self.toolFrame, text="Sigma_X:")
        self.xLabel.grid(row=0, column=2, sticky=tk.E+tk.N+tk.S)
        
        self.toolFrame.grid_columnconfigure(0, weight=1)
        self.toolFrame.grid_columnconfigure(1, weight=1)
        self.toolFrame.grid_columnconfigure(2, weight=1)
        self.toolFrame.grid_columnconfigure(3, weight=1)
        self.toolFrame.grid_columnconfigure(4, weight=1)
        self.toolFrame.grid_columnconfigure(5, weight=1)
        
        self.xBox = Text(self.toolFrame, height=1, width=7)        
        self.xBox.grid(row=0, column=3, sticky=tk.W+tk.N+tk.S)
        
        self.fLabel = tk.Label(self.toolFrame, text="Sigma_F")
        self.fLabel.grid(row=0, column=4, sticky=tk.E+tk.N+tk.S)

        self.fBox = Text(self.toolFrame, height=1, width=7)
        self.fBox.grid(row=0, column=5, sticky=tk.W+tk.N+tk.S)
        
        self.resetFields()
        
        
    def resetFields(self):
        self.xBox.delete(1.0,tk.END)
        self.xBox.insert(tk.END,str(normalizedCutsFrame.DEFAULT_SIGMA_X))
        self.fBox.delete(1.0,tk.END)
        self.fBox.insert(tk.END,str(normalizedCutsFrame.DEFAULT_SIGMA_F))
        self.segmentBox.deselect()
        
    def nCutSegment(self):
        if not self.doSegment.get():
            self.imageWidget.drawCVImage(self.image)
            return
        else:
            sigmaFText = self.fBox.get("1.0",tk.END)
            sigmaXText = self.xBox.get("1.0",tk.END)
            
            if not sigmaFText.isdigit() or not float(sigmaFText) > 0:
                self.fBox.delete("1.0", tk.END)
                self.fBox.insert(tk.END, str(normalizedCutsFrame.DEFAULT_SIGMA_F))
                print("Setting Sigma_F to default value of " + str(normalizedCutsFrame.DEFAULT_SIGMA_F) + ".")
            sF = float(self.fBox.get("1.0",tk.END))

            if not sigmaXText.isdigit() or not float(sigmaXText) > 0:
                self.xBox.delete("1.0", tk.END)
                self.xBox.insert(tk.END, str(normalizedCutsFrame.DEFAULT_SIGMA_X))
                print("Setting Sigma_X to default value of " + str(normalizedCutsFrame.DEFAULT_SIGMA_X) + ".")
            sX = float(self.xBox.get("1.0",tk.END))            
                
            try:
                import segment; reload(segment)
                self.segmentedImage = segment.nCutSegmentation(self.image, sF, sX)
                self.imageWidget.drawCVImage(self.segmentedImage)
            except:
                traceback.print_exc()
                return
        
class mainWindow(tk.Frame): 
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("segmentationGui.py")
        
        self.frame = tk.Frame(self.root, width=700, height=700)
        self.frame.grid(row=1)
        
        self.notebook = ttk.Notebook(self.frame)
        self.edgeTab = edgeDetectFrame(self.notebook)
        self.kMeansTab = kMeansSegmentationFrame(self.notebook)
        self.nCutTab = normalizedCutsFrame(self.notebook)
        
        self.notebook.add(self.edgeTab, text = "Edge Detection",compound=tk.TOP)     
        self.notebook.add(self.kMeansTab, text = "k-Means")
        self.notebook.add(self.nCutTab, text = "Normalized Cuts")
        self.notebook.grid(row=1)
        
        self.log = ScrolledText(self.frame, wrap='word')
        self.log.grid(row=2)
        
        sys.stdout = LogRedirector(self.log)
        sys.stderr = LogRedirector(self.log)

        tk.mainloop()
        
if __name__ == "__main__":
    window = mainWindow()