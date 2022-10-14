import numpy as np
from scipy import ndimage
from numba import jit
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfile
from PIL import ImageTk, Image

def edge_detectction(img):
    """"
    "Energy functin" - using the Sobel operator, the function preforming an edge detection on the given image
    """
    kernal_x = np.array([[-0.25, 0, 0.25],
                         [-0.5, 0, 0.5],
                         [-0.25, 0, 0.25]])
    kernal_y = np.transpose(kernal_x)
    img_gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
    edged_x = ndimage.convolve(img_gray,kernal_x)
    edged_y = ndimage.convolve(img_gray,kernal_y)
    img_sqrt =  np.sqrt(np.add(np.power(edged_x,2),np.power(edged_y,2)))
    img_edged = np.round(img_sqrt * (255/ np.max(img_sqrt)))
    img_edged = img_edged.astype(int)
    return img_edged


def compute_paths(img, img_edged ,window, output_label, new_size, transpose):
    """
    preforming the seam cariving - finding and dealeting paths from the image
    """
    img_show_loops= int(round(8000/img.shape[1]))
    indices_to_duplicate = None
    if new_size > img.shape[1]:
        img_with_indeces = np.pad(img, ((0,0),(0,0),(0,1)),'constant').astype(np.int32)
        img_with_indeces[:,:,3] += np.arange(img.shape[1])
        indices_to_duplicate = np.zeros((img.shape[0],img.shape[1]))
    else:
        img_with_indeces = img

    for i in range(np.abs(img.shape[1]-new_size)):
        img = img_with_indeces[:,:,0:3]
        min_energy, min_energy_index = create_min_energy_matrix(img_edged)
        path = discover_path(img_with_indeces, min_energy, min_energy_index, indices_to_duplicate)
        if i%img_show_loops == 0:
            if transpose:
                image = Image.fromarray(img.transpose((1,0,2)).astype(np.uint8))
            else:
                image = Image.fromarray(img.astype(np.uint8))
            update_img(window,image,output_label,False)
        img_with_indeces, img_edged = delete_path(path, img_with_indeces, img_edged)

    return img_with_indeces, img_edged, indices_to_duplicate


@jit
def create_min_energy_matrix(img_edged):
    """
    using Dynamic programming, we build a matrix which every cell represent the the sum of the path with the least energy from
    the last row to this current row. For cell [i,j] the previous cell in the path is one of [i+1,j-1], [i+1,j], [i+1,j+1]
    (when starting from the last row). In addition, creating a matrix that saving -1 or 0 or 1 according the previes cell in the path.
    """
    min_energy = np.empty(img_edged.shape, dtype=np.int32)
    min_energy_index = np.empty(img_edged.shape, dtype=np.int32)
    columns = img_edged.shape[1]
    for row in range(img_edged.shape[0]-2 , -1, -1):
        for col in range(1,columns-1):
            min_energy_index[row,col] = np.argmin(min_energy[row+1,col-1:col+2]) - 1
            min_energy[row,col] = min_energy[(row+1),(col + min_energy_index[row,col])] + img_edged[row,col]
        min_energy_index[row,columns-1] = np.argmin(min_energy[row+1,columns-2:columns]) -1
        min_energy[row,columns-1] = min_energy[(row+1),(columns-1 + min_energy_index[row,columns-1])] + img_edged[row,columns-1]
        min_energy_index[row,0] = np.argmin(min_energy[row+1,0:1])
        min_energy[row,0] = min_energy[(row+1),(min_energy_index[row,0])] + img_edged[row,0]
    return min_energy, min_energy_index


def discover_path(img, min_energy, min_energy_index, indices_to_duplicate):
    """
    function that finding the path with the least energy using the matrices from the previus functions, creating a matrix when all
    the cells in the path are marked as False. In addition, we change the color of the cells in the path to pink.
    """
    seam_insertion_bool = indices_to_duplicate is not None
    path = np.ones(min_energy.shape, dtype=bool)
    col = np.argmin(min_energy[0,:])
    if seam_insertion_bool:
            indices_to_duplicate[0,img[0,col,3]] = 1
    path[0,col] = 0
    pink = [255,0,255]
    for row in range(1,min_energy.shape[0]):
        col = col + min_energy_index[row-1,col]
        if seam_insertion_bool:
            indices_to_duplicate[row,img[row,col,3]] = 1
        path[row,col] = 0
        img[row,col,0:3] = pink
    return path


def delete_path(path, img, img_edged):
    """
    Deleting the path from the original image and the image with edge detection
    """
    new_img = img[np.stack([path]*img.shape[2], axis=2)].reshape((img.shape[0],img.shape[1]-1, img.shape[2]))
    new_img_edged = img_edged[path].reshape((img.shape[0],img.shape[1]-1))
    return new_img, new_img_edged


def seam_insertion(img, img_edged, new_size, indices_to_duplicate):
    """
    The function implements seam insertion according to the pixels which chosen to 'delete' and were marked
    int the matrix indeces_toduplicate
    """
    updated_img = np.empty((img.shape[0],new_size,3), dtype=np.int32)
    updated_img_edged = np.empty((img.shape[0],new_size), dtype=np.int32)
    for row in range(updated_img.shape[0]):
        index = 0
        for col in range(updated_img.shape[1]):
            updated_img[row,col,:] = img[row,index,0:3]
            updated_img_edged[row,col] = img_edged[row,index]
            if indices_to_duplicate[row,index] == 0:
                index += 1
            else:
                indices_to_duplicate[row,index] = 0
    return updated_img, updated_img_edged


def main():
    window = tk.Tk()
    window.title("Seam Carving")
    window.resizable(height=False,width=False)
    top_frame = tk.Frame(master=window, height=100)
    top_frame.pack(fill=tk.X, side=tk.TOP)

    left_frame = tk.Frame(master=window, width=500, height=400 ,bg="white")
    left_frame.pack(fill=tk.Y, side=tk.LEFT, padx=4, pady=4)

    right_frame = tk.Frame(master=window, width=500, bg="white")
    right_frame.pack(fill=tk.Y, side=tk.RIGHT, padx=4, pady=4)

    filepath=[""]
    str_size = tk.StringVar()
    str_size.set("Orginal size:\nHeight:\nWidth:")
    str_computing = tk.StringVar()
    tk.Label(top_frame, textvariable=str_size).place(x=90, y=-1)
    tk.Label(right_frame, textvariable=str_computing, font=("Arial",12), bg="white").place(x=210, y=0)
    width_entery = tk.Entry(top_frame)
    width_entery.place(x=590,y=24)
    height_entery = tk.Entry(top_frame)
    height_entery.place(x=590,y=3)

    tk.Label(top_frame, text="New height:").place(x=517, y=3)
    tk.Label(top_frame, text="New width:").place(x=517, y=24)
    img_label=tk.Label(left_frame)
    output_label=tk.Label(right_frame)
    open_btn = tk.Button(top_frame,text="Open",command=lambda:open_file(window, img_label,str_size, filepath))
    open_btn.grid(padx=5, pady=10)
    out=[""]
    compute_btn = tk.Button(top_frame,text="compute", command=lambda:compute(window, output_label, filepath, width_entery, height_entery, str_computing, out))
    compute_btn.place(x=730, y=10)
    save_btn = tk.Button(top_frame,text="Save", command=lambda:save_file(out[0]))
    save_btn.place(x=950, y=10)
    window.mainloop()



def save_file(img):
    """
    Saving the output image
    """
    file = asksaveasfile(mode='w', defaultextension=".jpg", filetypes=[("jpg file",".jpg"),("png file",".png"), ("bmp file",".bmp")])
    img.save(file.name)



def compute(window, output_label, filepath, width_entery, height_entery, str_computing, out):
    """
    The funcion read the needed new values for the image size, and calling compute_paths for width and height if needed
    """
    str_computing.set("Computing...")
    img = np.asarray(Image.open(filepath[0]),dtype=np.int32)
    img_edged = edge_detectction(img)

    try:
        new_width = int(width_entery.get())
    except:
        new_width = None
    try:
        new_height = int(height_entery.get())
    except:
        new_height = None

    if new_width is not None:
        original_img = img.copy().astype(np.int32)
        original_img_edged = img_edged.copy().astype(np.int32)
        img, img_edged, indices_to_duplicate = compute_paths(img, img_edged ,window, output_label, new_width,transpose=False)

        if indices_to_duplicate is not None:
                img, img_edged = seam_insertion(original_img, original_img_edged ,new_width, indices_to_duplicate)

    if new_height is not None:
        img_edged = img_edged.transpose()
        img = img.transpose((1,0,2))
        img_copy = img.copy().astype(np.int32)
        img_copy_edged = img_edged.copy().astype(np.int32)
        img, img_edged, indices_to_duplicate = compute_paths(img, img_edged ,window, output_label, new_height,transpose=True)
        if indices_to_duplicate is not None:
            img, img_edged = seam_insertion(img_copy, img_copy_edged, new_height, indices_to_duplicate)
        img = img.transpose((1,0,2))

    str_computing.set("")
    image = Image.fromarray(img.astype(np.uint8))
    out[0] = image
    update_img(window,image,output_label)
    window.mainloop()



def update_img(window,image, label, final= True):
    """
    function which manges how and where the image shows in the GUI
    """
    if image.size[0] > image.size[1]:
        height = int((350/image.size[0])*image.size[1])
        img = ImageTk.PhotoImage(image.resize((450,height)))
        label.configure(image=img)
        label.pack(padx=25, pady=(400-height)/2, expand=True)
    else:
        width = int((450/image.size[1])*image.size[0])
        img = ImageTk.PhotoImage(image.resize((width,350)))
        label.configure(image=img)
        label.pack(padx=(500-width)/2, pady=25, expand=True)
    if final:
        window.mainloop()
    else:
        window.update()


def open_file(window, img_label,str_size, filepath):
    """
    choosing an image from the computer
    """
    filepath[0] = askopenfilename(filetypes=[("image Files", ("*.png","*.jpg","*.jpeg","*.bmp"))])
    if filepath[0] == "":
        window.title(f"Seam Carving")
        img_label.pack_forget()
        str_size.set("Orginal size:\nHeight:\nWidth:")
    else:
        window.title(f"Seam Carving - {filepath[0]}")
        image = Image.open(filepath[0])
        str_size.set(f"Orginal size:\nHeight: {image.size[1]}\nWidth: {image.size[0]}")
        update_img(window,image,img_label)
    window.mainloop()

if __name__ == "__main__":
    main()