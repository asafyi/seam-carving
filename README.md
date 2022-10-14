# Seam Carving
Seam Carving implementation in python (Seam Removal and Seam Insertion) for performing content aware resinig of images.
The project is based on the paper "Seam Carving for Content-Aware Image Resizing" by Shai Avidan and Ariel Shamir.  
Link to the paper: https://faculty.runi.ac.il/arik/scweb/imret/imret.pdf  

![alt text](./images/Animation.gif?raw=true)
In the example above, the program gets an image in size 1601 x 664 and performing removal of vertical seams and insertion of horizontal seams 
(The algorithm of seam inseartion is based on seam removal - performing removal of seams and then add these seams to the orginal image).
After the calculations, we get a new image in size 1200 X 900 without losing and destroying important objects in the image.

## Requirements
* numpy
* scipy
* numba
* tkinter
* PIL

## Usage
```
python seam_carving.py
```
