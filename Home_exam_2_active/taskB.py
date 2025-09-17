import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    #Using the code provided to read the data
    # define paths to data and header file
    data_file = "image.dat"
    header_file = "image.txt"
    # number of pixels from meta data
    Ny = 1759
    Nx = 501
    # read the image data as vector and reshape to image
    dat = np.fromfile(data_file, dtype=np.csingle)
    img = dat.reshape(Ny,Nx)

    #Finding the absolute values of the data
    img_abs_val = np.abs(img)