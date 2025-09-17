import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import constants as spc

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

    #Plotting
    fig, ax = plt.subplots(figsize = (6, 6))
    im = ax.imshow(img_abs_val, cmap = 'gray', aspect = 'auto', origin = 'lower')
    fig.colorbar(im, ax = ax)
    ax.set_title('Absolute value of the complex image')
    ax.set_xlabel('Range (pixels)')
    ax.set_ylabel('Azimuth (pixels)')
    plt.savefig('abs_val_SLC.png')
    plt.close()

    '''---- Task A1 ----'''
    
    #Finding the real and imaginary parts of the image data, making it into 1D
    real_data = np.real(img).flatten()
    imaginary_data = np.imag(img).flatten()

    #Plotting histograms of real and imag parts
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    axes[0].hist(real_data, bins = 100, color = 'gray', edgecolor = 'black')
    axes[0].set_title('Histogram of the real part of the data')
    axes[0].set_xlabel('Amplitude')
    axes[0].set_ylabel('Number of pixels')

    axes[1].hist(imaginary_data, bins = 100, color = 'gray', edgecolor = 'black')
    axes[1].set_title('Histogram of the imaginary part of the data')
    axes[1].set_xlabel('Amplitude')
    plt.tight_layout()
    plt.savefig('histograms_taskA1')
    plt.close()

    '''---- Task A2 ----'''

    #Calculating the intensity
    intensity = img_abs_val **2

    #Plotting
    fig, ax = plt.subplots(figsize = (7, 6))
    ax.hist(intensity.flatten(), bins = 100, color = 'gray', edgecolor = 'black')
    ax.set_title('Histogram of image intensity')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Number of pixels')
    ax.set_xbound(0, 3.6)
    plt.savefig('intensity_hist_taskA2')
    plt.close()

    #Calculating the mean and variance of the intensity
    intensity_mean = np.mean(intensity)
    intensity_var = np.var(intensity)

    #Using these to calculate the normalized variance: var / (mean)^2
    norm_var_intensity = intensity_var / (intensity_mean**2)
    print(f'The normalized variance is {norm_var_intensity}')

    '''---- Task A3 ----'''

    intensity_smooth = ndimage.uniform_filter(intensity, size = 5)

    #plotting
    fig, ax = plt.subplots(figsize = (7, 6))
    ax.hist(intensity_smooth.flatten(), bins = 100, color = 'gray', edgecolor = 'black')
    ax.set_title('Histogram of image intensity')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Number of pixels')
    plt.savefig('smooth_intensity_hist_taskA3')
    plt.close()

    #Calculating the mean and variance of the intensity after filtering
    smooth_intensity_mean = np.mean(intensity_smooth)
    smooth_intensity_var = np.var(intensity_smooth)

    #Finding the normalized variance
    smooth_norm_var_intensity = smooth_intensity_var / (smooth_intensity_mean**2)
    smooth_2 = np.mean(img_abs_val)**2 / 25
    print(smooth_2)
    print(f'The normalized variance is {smooth_norm_var_intensity}')   

    '''---- Task B1 ----'''
    #Finding the needed data from image.txt, defining it here
    sample_freq_y = 1686.0674 #Hz
    sample_frec_x = 19207680. #Hz
    ground_velocity = 6716.7298 #m/s
    inc_angle = 22.779148 #degrees

    #Finding image resolution
    delta_y = ground_velocity / sample_freq_y #m
    delta_x = spc.c / (2 * sample_frec_x * np.sin(inc_angle)) #m

    #Finding the max wavenumber
    k_x_max = np.pi / delta_x
    k_y_max = np.pi / delta_y

    #Defining an array for the wavenumbers
    k_x = np.linspace(-k_x_max, k_x_max, Nx)
    k_y = np.linspace(-k_y_max, k_y_max, Ny)

    #taking the square root of the mean intensity
    sqrt_intensity_mean = np.sqrt(intensity_mean)

    #Normalizing the image
    norm_img = img / sqrt_intensity_mean

    #Fourier transforming only in azimuth direction and shifting it using fftshift
    fft_img = np.fft.fft(norm_img, axis = 0)
    shifted_fft_img = np.fft.fftshift(fft_img, axes = 0)

    #Taking the absolute value (magnitude) of the fourier transformed image
    abs_fft_img = np.abs(shifted_fft_img)

    #Plotting
    fig, ax = plt.subplots(figsize = (6, 6))
    im = ax.imshow(abs_fft_img, cmap = 'gray', aspect = 'auto', 
                   origin = 'lower', extent = [0, Nx, - k_y_max, k_y_max])
    fig.colorbar(im, ax = ax)
    ax.set_title('Absolute value of fourier transformed image')
    ax.set_xlabel('Range (pixels)')
    ax.set_ylabel('Azimuth wavenumber (rad/m)')
    plt.savefig('abs_fft_im_taskB1.png')
    plt.close()

    #Shape not in the middle of the azimuth, shifted antenna pattern

    '''---- Task B2 ----'''
    #Averaging along range axis
    azimuth_spectral_profile = np.mean(abs_fft_img, axis = 1)

    #Finding the frquencies
    # frequencies = np.linspace(- 0.5 * sample_freq_y, 0.5 * sample_freq_y, Ny)
    frequencies_y = np.fft.fftshift(np.fft.fftfreq(Ny, d = 1 / sample_freq_y))

    fig, ax = plt.subplots(figsize = (6, 6))
    ax.plot(frequencies_y, azimuth_spectral_profile)
    ax.set_title('Spectral profile azimuth')
    ax.set_xlabel('PRF (Hz)')
    ax.set_ylabel('Spectral magnitude')
    plt.savefig('spectralprofile_taskB2.png')
    plt.close()

    #Very shifted spectral profile

    '''---- Task B3 ----'''
    #Finding the maximum value of the spectral profile
    max_index = np.argmax(azimuth_spectral_profile)
    middle_index = int(len(azimuth_spectral_profile) / 2)
    idx_shift = middle_index - max_index #Index and pixels have the same value

    azimuth_shift_image = np.roll(shifted_fft_img, idx_shift - 30, axis = 0)
    centered_spectral_profile = np.mean(np.abs(azimuth_shift_image), axis = 1)

    #Getting the minimum and maximum azimuth frequencies
    print(f'Maximum azimuth frequency: {k_y_max}')
    print(f'Minimun azimuth frequency: {-k_y_max}')

    #Plotting, we have transposed the array as the task asked for azimuth freq along the x-axis
    fig, ax = plt.subplots(figsize = (6, 6))
    im = ax.imshow(np.transpose(np.abs(azimuth_shift_image)), cmap = 'gray', aspect = 'auto', 
                   origin = 'lower', extent = [- k_y_max, k_y_max, 0, Nx])
    fig.colorbar(im, ax = ax)
    ax.set_title('Absolute value of fourier transformed image')
    ax.set_ylabel('Range (pixels)')
    ax.set_xlabel('Azimuth wavenumber [rad/m]')
    plt.savefig('abs_fft_im_taskB3.png')
    plt.close()

    #Plotting the spectral profile
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.plot(k_y, centered_spectral_profile)
    ax.set_title('Spectral profile azimuth')
    ax.set_xlabel('Azimuth wavenumber [rad/m]')
    ax.set_ylabel('Spectral magnitude')
    plt.savefig('centered_spectralprofile_taskB3.png')
    plt.close()

    '''---- Task B4 ----'''
    #Defining the 3 different looks
    look_size = Ny // 3
    look1 = azimuth_shift_image[0: look_size, :]
    look2 = azimuth_shift_image[look_size: 2*look_size, :]
    look3 = azimuth_shift_image[2*look_size: 3*look_size, :]

    inv_fft_look1 = np.fft.ifft(np.fft.ifftshift(look1, axes = 0), axis = 0)
    inv_fft_look2 = np.fft.ifft(np.fft.ifftshift(look2, axes = 0), axis = 0)
    inv_fft_look3 = np.fft.ifft(np.fft.ifftshift(look3, axes = 0), axis = 0)

    #making new wavenumber arrays because of the slicing
    k_x = np.linspace(-np.pi / delta_x, np.pi / delta_x, Nx)
    k_y = np.linspace(-np.pi / delta_y, np.pi / delta_y, look_size)

    #Plotting all three looks intensity images
    fig, axes = plt.subplots(1, 3, figsize = (16, 7))

    axes[0].imshow(np.abs(inv_fft_look1)**2, cmap = 'gray', origin = 'lower', aspect = 'auto')
    axes[0].axis('off')
    axes[0].set_title("Look 1", fontsize = 20)

    axes[1].imshow(np.abs(inv_fft_look2)**2, cmap = 'gray', origin = 'lower', aspect = 'auto')
    axes[1].axis('off')
    axes[1].set_title("Look 2", fontsize = 20)

    axes[2].imshow(np.abs(inv_fft_look3)**2, cmap = 'gray', origin = 'lower', aspect = 'auto')
    axes[2].axis('off')
    axes[2].set_title("Look 3", fontsize = 20)

    plt.tight_layout()
    plt.savefig('Looks_taskB4.png')
    plt.close()

    '''---- Task B5 ----'''
    #Intensities and means for the 3 looks
    intensity_look1 = np.abs(inv_fft_look1) **2
    mean_intensity1 = np.mean(intensity_look1)

    intensity_look2 = np.abs(inv_fft_look2) **2
    mean_intensity2 = np.mean(intensity_look2)

    intensity_look3 = np.abs(inv_fft_look3) **2
    mean_intensity3 = np.mean(intensity_look3)

    #Normalizing: (I - <I>) / I
    look1_norm = (intensity_look1 - mean_intensity1) / mean_intensity1
    look2_norm = (intensity_look2 - mean_intensity2) / mean_intensity2
    look3_norm = (intensity_look3 - mean_intensity3) / mean_intensity3

    #Fourier transforming
    fft_look1_norm = np.fft.fftshift(np.fft.fft2(look1_norm))
    fft_look2_norm = np.fft.fftshift(np.fft.fft2(look2_norm))
    fft_look3_norm = np.fft.fftshift(np.fft.fft2(look3_norm))

    #Multiplying the spectras with each others
    #Co spectra
    co_1x1 = fft_look1_norm * np.conj(fft_look1_norm)
    co_2x2 = fft_look2_norm * np.conj(fft_look2_norm)
    co_3x3 = fft_look3_norm * np.conj(fft_look3_norm)

    #cross spectra
    cr_1x2 = fft_look1_norm * np.conj(fft_look2_norm)
    cr_2x3 = fft_look2_norm * np.conj(fft_look3_norm)
    cr_1x3 = fft_look1_norm * np.conj(fft_look3_norm)

    avg_co = (co_1x1 + co_2x2 + co_3x3) / 3
    avg_cross = (cr_1x2 + cr_2x3) / 2

    #Defining axes for plotting
    extent = [k_x[0], k_x[-1], k_y[0], k_y[-1]]

    fig, axes = plt.subplots(1, 3, figsize = (16,7))
    im1 = axes[0].imshow(np.log1p(np.abs(avg_co)), cmap = 'gray',
               extent = extent, aspect = 'auto', vmin = 10, origin = 'lower')
    axes[0].set_title("Co-spectrum average")
    axes[0].set_xlabel("Range Wavenumber $k_x$ [rad/m]")
    axes[0].set_ylabel("Azimuth Wavenumber $k_y$ [rad/m]")
    fig.colorbar(im1, ax = axes[0], orientation = 'vertical')
    
    im2 = axes[1].imshow(np.log1p(np.abs(avg_cross)), cmap = 'gray',
                         extent = extent, aspect = 'auto', vmin = 10,
                          origin = 'lower')
    axes[1].set_title("Cross-spectrum average")
    axes[1].set_xlabel("Range Wavenumber $k_x$ [rad/m]")
    axes[1].set_ylabel("Azimuth Wavenumber $k_y$ [rad/m]")
    fig.colorbar(im2, ax = axes[1], orientation = 'vertical')
    
    im3 = axes[2].imshow(np.log1p(np.abs(cr_1x3)), cmap = 'gray',
                         extent = extent, aspect = 'auto', vmin = 10,
                          origin = 'lower')
    axes[2].set_title("Cross-spectrum (S13)")
    axes[2].set_xlabel("Range Wavenumber $k_x$ [rad/m]")
    axes[2].set_ylabel("Azimuth Wavenumber $k_y$ [rad/m]")
    fig.colorbar(im3, ax = axes[2], orientation = 'vertical')

    plt.tight_layout()
    plt.savefig('taskB5.png')
    plt.close()

    '''---- Task C1 ----'''
    #Finding wavenumber bin sizes
    delta_k_x = (2 * np.pi) / (Nx * delta_x) 
    delta_k_y = (2 * np.pi) / (look_size * delta_y)  

    #Defining an array for the wavenumbers
    kx = np.arange(-k_x_max, k_x_max, delta_k_x)
    ky = np.arange(-k_y_max, k_y_max, delta_k_y)

    #Plotting
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))
    im1 = axes[0].imshow(np.real(avg_co), extent = extent,
                          cmap = 'gray', aspect = 'auto',
                          vmin = 10)
    axes[0].set_xlim(-0.1, 0.1)
    axes[0].set_ylim(-0.1, 0.1)
    axes[0].set_title('Real Part of co-spectrum')
    axes[0].set_xlabel("Range wavenumber $k_x$ [rad/m]")
    axes[0].set_ylabel("Azimuth wavenumber $k_y$ [rad/m]")
    fig.colorbar(im1, ax = axes[0])

    im2 = axes[1].imshow(np.imag(avg_co), extent = extent, 
                         cmap = 'gray', aspect = 'auto'
                        )
    axes[1].set_xlim(-0.1, 0.1)
    axes[1].set_ylim(-0.1, 0.1)
    axes[1].set_title('Imaginary Part of co-spectrum')
    axes[1].set_xlabel("Range wavenumber $k_x$ [rad/m]")
    axes[1].set_ylabel("Azimuth wavenumber $k_y$ [rad/m]")
    fig.colorbar(im2, ax = axes[1])

    plt.tight_layout()
    plt.savefig('avgco_real_imag.png')
    plt.close()

    fig2, axes2 = plt.subplots(1, 2, figsize = (12, 6))

    im1 = axes2[0].imshow(np.real(avg_cross), extent = extent,
                          cmap = 'gray', aspect = 'auto')
    axes2[0].set_xlim(-0.1, 0.1)
    axes2[0].set_ylim(-0.1, 0.1)
    axes2[0].set_title('Real Part of cross-spectrum')
    axes2[0].set_xlabel("Range wavenumber $k_x$ [rad/m]")
    axes2[0].set_ylabel("Azimuth wavenumber $k_y$ [rad/m]")
    fig2.colorbar(im1, ax = axes2[0])

    image_to_plot = np.flipud(np.imag(avg_cross))

    im2 = axes2[1].imshow(np.imag(avg_cross), extent = extent, 
                         cmap = 'gray', aspect = 'auto')
    axes2[1].set_xlim(-0.1, 0.1)
    axes2[1].set_ylim(-0.1, 0.1)
    axes2[1].set_title('Imaginary Part of cross-spectrum')
    axes2[1].set_xlabel("Range wavenumber $k_x$ [rad/m]")
    axes2[1].set_ylabel("Azimuth wavenumber $k_y$ [rad/m]")
    fig2.colorbar(im2, ax = axes2[1])

    plt.tight_layout()
    plt.savefig('avgcross_real_imag.png')
    plt.close()

    fig3, axes3 = plt.subplots(1, 2, figsize = (12, 6))

    im1 = axes3[0].imshow(np.real(cr_1x3), extent = extent,
                          cmap = 'gray', aspect = 'auto')
    axes3[0].set_xlim(-0.1, 0.1)
    axes3[0].set_ylim(-0.1, 0.1)
    axes3[0].set_title('Real Part of two-step cross-spectrum')
    axes3[0].set_xlabel("Range wavenumber $k_x$ [rad/m]")
    axes3[0].set_ylabel("Azimuth wavenumber $k_y$ [rad/m]")
    fig3.colorbar(im1, ax = axes3[0])

    im2 = axes3[1].imshow(np.imag(cr_1x3), extent = extent, 
                         cmap = 'gray', aspect = 'auto')
    axes3[1].set_xlim(-0.1, 0.1)
    axes3[1].set_ylim(-0.1, 0.1)
    axes3[1].set_title('Imaginary Part of two-step cross-spectrum')
    axes3[1].set_xlabel("Range wavenumber $k_x$ [rad/m]")
    axes3[1].set_ylabel("Azimuth wavenumber $k_y$ [rad/m]")
    fig3.colorbar(im2, ax = axes3[1])

    plt.tight_layout()
    plt.savefig('twostep_cross_real_imag.png')
    plt.close()

    #Finding the max spectral value from the 2d spectra
    abs_val_co = np.abs(avg_co)
    max_spectral_index1 = np.unravel_index(np.argmax(abs_val_co), abs_val_co.shape)

    abs_val_cross = np.abs(avg_cross)
    max_spectral_index2 = np.unravel_index(np.argmax(abs_val_cross), abs_val_cross.shape)

    abs_val_1x3 = np.abs(cr_1x3)
    max_spectral_index3 = np.unravel_index(np.argmax(abs_val_1x3), abs_val_1x3.shape)

    print(abs_val_1x3.shape)

    #Extracting the wavenumber of the maximum value
    max_kx1 = k_x[max_spectral_index1[1]]
    max_ky1 = k_y[max_spectral_index1[0]]

    max_kx2 = k_x[max_spectral_index2[1]]
    max_ky2 = k_y[max_spectral_index2[0]]

    max_kx3 = k_x[max_spectral_index3[1]]
    max_ky3 = k_y[max_spectral_index3[0]]

    #Finding the magnitude of the wavenumber vector
    mag_k1 = np.sqrt(max_kx1**2 + max_ky1**2)
    mag_k2 = np.sqrt(max_kx2**2 + max_ky2**2)
    mag_k3 = np.sqrt(max_kx3**2 + max_ky3**2)

    #taking the average wavenumber
    mag_k = (mag_k1 + mag_k2 + mag_k3) / 3

    #Calculating the wavlength
    wavelength = (2 * np.pi) / mag_k

    #Defining the values of kx and ky seen in the plot
    k_x_value = -0.02
    k_y_value = 0.03

    #Finding the magnitude of k
    mag_k = np.sqrt(k_x_value**2 + k_y_value**2)

    #Calculating the wavlength
    wavelength = (2 * np.pi) / mag_k

    print(f'The wavelength of the max spectral energy is {wavelength}')





    

