import numpy as np

#Making functions for the surface roughness spectrums, Gaussian and exponential
def Gaussian_correlation_func(corr_length, theta, wavelength) -> float:

    '''
    Computes the Gaussian surface roughness spectrum W_g for a given correlation length, incidence angle, and wavelength.

    Parameters:
    - corr_length (float): Correlation length of the surface roughness (in meters).
    - theta (float): Local incidence angle in radians.
    - wavelength (float): Wavelength of the radar signal (in meters).

    Returns:
    - W_g (float): The normalized Gaussian correlation function value for the given parameters.
    '''

    k = (2 * np.pi) / wavelength
    kx2 = 2 * k * np.sin(theta)
    ky2 = 0
    
    W_g = (corr_length**2 / np.pi) * np.exp(- ((kx2 + ky2)**2 * corr_length**2) / 4)

    return W_g

def exp_correlation_func(corr_length, theta, wavelength) -> float:

    '''
    Computes the exponential surface roughness spectrum W_e for a given correlation length, incidence angle, and wavelength.

    Parameters:
    - corr_length (float): Correlation length of the surface roughness (in meters).
    - theta (float): Local incidence angle in radians.
    - wavelength (float): Wavelength of the radar signal (in meters).

    Returns:
    - W_e (float): The normalized Gaussian correlation function value for the given parameters.    

    '''

    k = (2 * np.pi) / wavelength
    kx2 = 2 * k * np.sin(theta)
    ky2 = 0

    W_e = (2 * corr_length**2) / (np.pi * (1 + (kx2 + ky2)**2 * corr_length**2))

    return W_e



