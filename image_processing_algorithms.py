import numpy as np
from scipy import ndimage
from numba import jit
from PIL import ImageTk, Image



def edge_detectction(img):
    """"
    "Energy functin" - using the Sobel operator, the function performing an edge detection on the given image
    """
    kernal_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
    kernal_y = np.transpose(kernal_x)
    img_gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
    edged_x = ndimage.convolve(img_gray,kernal_x)
    edged_y = ndimage.convolve(img_gray,kernal_y)
    img_sqrt =  np.sqrt(np.add(np.power(edged_x,2),np.power(edged_y,2)))
    img_edged = np.round(img_sqrt * (255/ np.max(img_sqrt)))
    img_edged = img_edged.astype(int)
    return img_edged




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

