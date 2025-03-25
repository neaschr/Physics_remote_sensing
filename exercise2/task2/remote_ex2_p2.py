import numpy as np
import matplotlib.pyplot as plt

def reflection_coeff_h(theta: np.ndarray, n_in: float, n_out: float) -> np.ndarray:

    '''
    Calculates the reflection coefficient for horizontally polarized light at the interface between two media.

    Parameters:
        theta (numpy.ndarray): Angle of incidence in radians (measured from the normal to the surface).
        n_in (float): Refractive index of the incident medium.
        n_out (float): Refractive index of the transmitted medium.

    Returns:
        R_sq (numpy.ndarray): Reflection coefficient squared for horizontal polarization.
    '''

    #Computing the sine of the transmission angle using Snell's Law
    sin_theta_t = n_in * np.sin(theta) / n_out

    #Avoiding invalid arcsin values
    sin_theta_t = np.clip(sin_theta_t, -1, 1) 

    #Finding transmission angles in radians 
    theta_t = np.arcsin(sin_theta_t)

    #Fresnel equation, with handling for normal incidence
    specialcase = (theta == 0)
    R_sq = np.empty_like(theta)
    R_sq[~specialcase] = ((np.sin(theta[~specialcase] - 
                                  theta_t[~specialcase]))**2) / ((np.sin(theta[~specialcase] + theta_t[~specialcase]))**2)
    R_sq[specialcase] = ((n_in-n_out)/(n_in+n_out))**2 

    return R_sq


def reflection_coeff_v(theta: np.ndarray, n_in: float, n_out: float) -> np.ndarray:
    '''
    Calculates the reflection coefficient for vertically polarized light at the interface between two media.

    Parameters:
        theta (numpy.ndarray): Angle of incidence in radians (measured from the normal to the surface).
        n_in (float): Refractive index of the incident medium.
        n_out (float): Refractive index of the transmitted medium.

    Returns:
        R_sq (numpy.ndarray): Reflection coefficient squared for vertical polarization.
    '''

    #Computing the sine of the transmission angle using Snell's Law
    sin_theta_t = n_in * np.sin(theta) / n_out

    #Avoiding invalid arcsin values
    sin_theta_t = np.clip(sin_theta_t, -1, 1)  

    #Finding transmission angles in radians 
    theta_t = np.arcsin(sin_theta_t)

    #Fresnel equation, with handling for normal incidence
    specialcase = (theta == 0)
    R_sq = np.empty_like(theta)
    R_sq[~specialcase] = ((np.tan(theta[~specialcase] - 
                                  theta_t[~specialcase]))**2) / ((np.tan(theta[~specialcase] + theta_t[~specialcase]))**2)
    R_sq[specialcase] = ((n_in-n_out)/(n_in+n_out))**2 

    return R_sq

if __name__ == '__main__':

    #Defining the refractions indices
    n_1 = 1
    n_2 = 1.7
    n_3 = 9

    #Making arrays for angles, first in radians then in degrees for plotting
    angles = np.linspace(0.0, np.pi/2, 10000)
    angles2 = (angles/np.pi) * 180

    #Calculating
    wave1_h = reflection_coeff_h(angles, n_1, n_2)
    wave2_h = reflection_coeff_h(angles, n_1, n_3)
    wave1_v = reflection_coeff_v(angles, n_1, n_2)
    wave2_v = reflection_coeff_v(angles, n_1, n_3)

    #Plotting
    fig, ax = plt.subplots(figsize = (9, 6))
    ax.plot(angles2, wave1_h, label = r'$n = 1.7, |R_h|^2$', color = 'blue', linestyle = 'dashed')
    ax.plot(angles2, wave2_h, label = r'$n = 9, |R_h|^2$', color = 'purple', linestyle = 'dashed')
    ax.plot(angles2, wave1_v, label = r'$n = 1.7, |R_v|^2$', color = 'blue')
    ax.plot(angles2, wave2_v, label = r'$n = 9, |R_v|^2$', color = 'purple')

    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Reflection Coefficient (Power)")
    ax.set_title("Reflection Coefficients for Different Polarizations and Refractive Indices")
    ax.grid(True)

    plt.savefig('refraction.png')
    plt.legend()
    plt.show()