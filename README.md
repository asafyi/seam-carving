# Seam Carving
Seam Carving implementation in python (Seam Removal and Seam Insertion) for performing content-aware image resizing.
The project is based on the paper ["Seam Carving for Content-Aware Image Resizing"](Seam_Carving_for_Content-Aware_Image_Resizing.pdf) by Shai Avidan and Ariel Shamir.   

The algorithm is using edge detection to detect objects in the image and choose the path for removal or insertion. Edge detection is implemented using Sobel Operator as described in the paper ["A Descriptive Algorithm for Sobel Image Edge Detection"](A_Descriptive_Algorithm_for_Sobel_Image_Edge_Detection%20.pdf).
For example:
<p align="center"><img src="./images/photo-1628707351135-e963f2aa4387.jpg" width="300">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="./images/trees_sobel.jpg" width="300"></p>
Then, using this image and Dynamic Programming, the algorithm will find the path with the least "energy" and remove it from both images:    
<p align="center"><img src="./images/seam-highland-view-bed-and.jpg"></p>
This stage is repeated until the image is in the requested size. </br>
The final result:   
<img src="./images/Animation.gif">
In the example above, the program gets an image in size 1601 x 664 and performs removal of vertical seams and insertion of horizontal seams 
(The algorithm of seam insertion is based on seam removal - performing removal of seams and then adding these seams to the original image).
After the calculations, we get a new image in size 1200 X 900 without losing and destroying important objects in the image.

## Requirements
* numpy
* scipy
* numba
* tkinter
* PIL

## Usage
```bash
python seam_carving.py
```
When the GUI opens, you will be able to select an image from your computer, the image will be displayed on the left side, along with its original size (in pixels). Afterward, you can choose a new width or height (if the text box is empty, the size in this dimension won't change) and start the algorithm. After computing, the new image will appear on the right side and you will be able to save it.
