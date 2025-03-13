import numpy as np
import matplotlib.pyplot as plt
import correlationfunc as cf

#Making a function to calculate scattering crossection
def scattering_crossec(wavelength: float, theta: float, W: float, h: float, epsilon: float) -> tuple:
    """
    Computes the radar scattering cross-sections using the Small Perturbation Model (SPM).

    Parameters:
    - wavelength (float): Radar wavelength in meters.
    - theta (float): Local incidence angle in radians.
    - W (float): Normalized surface roughness spectrum.
    - h (float): Root mean square of surface roughness in meters.
    - epsilon (float): Dielectric constant of the surface.

    Returns:
    - sigma_hh (float): Scattering cross-section for horizontally polarized waves (HH).
    - sigma_vv (float): Scattering cross-section for vertically polarized waves (VV).

    """

    k = 2* np.pi / wavelength

    alpha_hh = (epsilon - 1) / ((np.cos(theta) + np.sqrt(epsilon - (np.sin(theta))**2))**2)
    alpha_vv = ((epsilon - 1) * ((epsilon - 1) * (np.sin(theta))**2 + epsilon)) / ((epsilon * np.cos(theta) + np.sqrt(epsilon - (np.sin(theta))**2))**2)
    
    sigma_hh = 4 * np.pi * k**4 * h**2 * (np.cos(theta)) * np.abs(alpha_hh)**2 * W
    sigma_vv = 4 * np.pi * k**4 * h**2 * (np.cos(theta)) * np.abs(alpha_vv)**2 * W

    return sigma_hh, sigma_vv

if __name__ == "__main__":

    incident_angles = np.linspace(0, np.pi, 100) #Angles in radians from 0 to pi
    correlation_length = 0.1 #m
    rms_height = 0.01 #m
    diel_const = 3
    wavel = 0.35 #m 

    #making arrays to store the calculated data
    scattering_gauss_hh = np.zeros(len(incident_angles))
    scattering_gauss_vv = np.zeros(len(incident_angles))
    scattering_exp_hh = np.zeros(len(incident_angles))   
    scattering_exp_vv = np.zeros(len(incident_angles))   
    
    #making a counter
    i = 0

    #Finding the surface function and then the scattering crossections
    for incident_angle in incident_angles:
        gauss_surface = cf.Gaussian_correlation_func(correlation_length, incident_angle, wavel)
        exp_surface = cf.exp_correlation_func(correlation_length, incident_angle, wavel)

        scattering_gauss_hh[i] = scattering_crossec(wavel, incident_angle, gauss_surface, rms_height, diel_const)[0]
        scattering_gauss_vv[i] = scattering_crossec(wavel, incident_angle, gauss_surface, rms_height, diel_const)[1]
        scattering_exp_hh[i] = scattering_crossec(wavel, incident_angle, exp_surface, rms_height, diel_const)[0]
        scattering_exp_vv[i] = scattering_crossec(wavel, incident_angle, exp_surface, rms_height, diel_const)[1]
        i += 1

    #Convert incident angles to degrees for plotting
    incident_angles_deg = np.degrees(incident_angles)

    #Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(incident_angles_deg, 10 * np.log10(scattering_gauss_hh), label="HH - Gaussian", linestyle='-')
    ax.plot(incident_angles_deg, 10 * np.log10(scattering_gauss_vv), label="VV - Gaussian", linestyle='--')
    ax.plot(incident_angles_deg, 10 * np.log10(scattering_exp_hh), label="HH - Exponential", linestyle='-.')
    ax.plot(incident_angles_deg, 10 * np.log10(scattering_exp_vv), label="VV - Exponential", linestyle=':')

    ax.set_xlabel("Incident Angle (degrees)")
    ax.set_ylabel("Scattering crossection (dB)")
    ax.set_title("Scattering crossection as a function of incident angle")
    ax.legend()
    ax.grid(True)

    plt.show()