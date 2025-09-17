import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def data_reader(file_name: str, column_names: list, delimiter_in: str = None) ->  tuple[pd.DataFrame, np.ndarray]:
    '''Function that takes in the name a data file and the names of the columns and returns
    the data as a pandas dataframe

    Parameters:
        file_name (str): Name of the file that will be read
        column_names (list): List containg the names wanted for the columns of data
        delimiter_in (str): Delimiter for the file, such as comma or whitespace, if no argument is passed, whitespace is assumed
    
    Returns:
        data (pd.DataFrame): Pandas dataframe with the data from the file
        data2 (np.ndarray): Numpy array containing the data from the file
    '''
    
    #Finding the first line starting with a number to find the data we are interested in
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() and line.strip()[0].isdigit():
                skip_rows = i
                break

    #Reading the data in the file given as an argument
    if delimiter_in:
        data = pd.read_csv(file_name, comment = '%', header = None, delimiter = delimiter_in, 
                           names = column_names, skiprows = skip_rows)
    else:
        data = pd.read_csv(file_name, comment = '%', header = None, delim_whitespace = True, 
                           names = column_names, skiprows = skip_rows)
    
    data2 = data.to_numpy()

    return data, data2

def spectral_profile_sat(spectral_profile: np.ndarray, satellite_band_central_wavelength: np.ndarray, 
                         sat_bandwidth: np.ndarray)-> np.ndarray:

    '''
    A function that takes in a spectral profile and band specifications of a satellite and averages the 
    spectral values over the satellite bands such as seen by the satellite

    All wavelengths should be in meters

    Parameters:
        spectral_profile (np.ndarray): 2d array containing wavelenghts (m) and irradiance
        satellite_band_central_wavelength (np.ndarray): Array containg the central wavelengths (m) of the satellites bands 
        sat_bandwidth (np.ndarray): Array containg the satellites bandwidths (m), has the same length as the array 
            containg central wavelengths

    Returns:
        band_averages (np.ndarray): An array of the same length as the spectral values containing the
            values avaraged over the satellites bands
        avg_band_values (np.ndarray): AN array of the same length as the number of spectral bands
            containing the average value over each band
    '''
    
    #Defining the bands
    i = 0
    bands = np.zeros((len(satellite_band_central_wavelength), 2))
    
    #Finding the bamds of the satellite by +- hakf the bandwidth from each central wavelength
    for central_freq in satellite_band_central_wavelength:
        band_min = central_freq - (sat_bandwidth[i] / 2)
        band_max = central_freq + (sat_bandwidth[i] / 2)
        bands[i] = [band_min, band_max]
        i += 1

    wavelengths = spectral_profile[0]
    values = spectral_profile[1]
    band_averages = np.zeros(len(values))
    avg_band_values = np.zeros(len(satellite_band_central_wavelength))
    i = 0

    #Finding the numbers inside the band using masking
    for band in bands:
        mask = (wavelengths >= band[0]) & (wavelengths <= band[1])

        if np.any(mask):
            band_averages[mask] = np.mean(values[mask])
            avg_band_values[i] = np.mean(values[mask])
        i += 1

    return (band_averages, avg_band_values)

def linear_mixing(R_observed: np.ndarray, R_soil: np.ndarray, R_grass: np.ndarray
                  )-> tuple[np.ndarray, float]:

    '''
    Using a spectral mixing model to estimate the fraction of grass

    Parameters:
        R_obs (np.ndarray): Observed reflectance (13 bands)
        R_grass (np.ndarray): Grass reflectance (13 bands)
        R_soil (np.ndarray): Soil reflectance (13 bands)

    Returns:
        f_grass (np.ndarray): fraction of grass 
        f_grass_mean (float): Mean grass fraction across bands
    '''

    fraction_grass = (R_observed - R_soil) / (R_grass - R_soil)
    fraction_grass_avg = np.mean(fraction_grass)


    return fraction_grass, fraction_grass_avg

if __name__ == '__main__':

    '''---- Task 2 a) ----'''

    #Defining column names and reading the data from the three files containing the data
    grass_col_names = ['wavelength [micrometer]', 'reflectance [percent]']
    spectral_profile_grass = data_reader(file_name = 'jhu.becknic.vegetation.grass.green.solid.gras.spectrum.txt',
                                        column_names = grass_col_names)[1]
    
    soil_col_names = ['wavelength [micrometer]', 'reflectance [percent]']
    spectral_profile_soil = data_reader(file_name = 'jhu.becknic.soil.alfisol.paleustalf.coarse.87P2410.spectrum.txt',
                                        column_names = soil_col_names)[1]
    
    atmos_col_names = ['wavelength [nanometer]', 'etr [watts per square meter per nm]', 
                       'global tilt [watts per square meter per nm]', 'direct circumsolar [watts per square meter per nm]']
    spectral_profile_atmosphere = data_reader(file_name = 'ASTMG173.csv',column_names = atmos_col_names,
                                              delimiter_in = ',')[1]

    #Plotting all wavelengths in nanometers
    #Grass plotting
    fig1, ax1 = plt.subplots(figsize = (6, 4))
    ax1.plot(spectral_profile_grass[:, 0] * 1e3, spectral_profile_grass[:, 1], color = 'green')
    ax1.set_xlabel('Wavelenght [nm]')
    ax1.set_ylabel('Reflectance [%]')
    ax1.set_title('Spectral profile of grass')
    ax1.grid(True)
    plt.savefig('spectral_profile_grass.png')
    plt.show()

    #Soil plotting
    fig2, ax2 = plt.subplots(figsize = (6, 4))
    ax2.plot(spectral_profile_soil[:, 0] *1e3, spectral_profile_soil[:, 1], color = 'saddlebrown')
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Reflectance [%]')
    ax2.set_title('Spectral profile of bare soil')
    ax2.grid(True)
    plt.savefig('spectral_profile_soil.png')
    plt.show()

    #Atmpspheric plotting
    #We have divided irradiance at ground level by irradiance at above atmosphere to 
    #get a percentage that is transmitted through the atmosphere
    fig3, ax3 = plt.subplots(figsize = (6,4))
    ax3.plot(spectral_profile_atmosphere[:, 0], 
             100*spectral_profile_atmosphere[:, 2]/spectral_profile_atmosphere[:, 1], color='blue')
    ax3.set_xlabel('Wavelength [nm]')
    ax3.set_ylabel('Transmission [%]')
    ax3.set_title('Normalized atmospheric transmission profile')
    ax3.grid(True)
    plt.savefig('spectral_profile_atmosphere2.png')
    plt.show()
    
    # #Atmospheric plotting, plotting the three different irradiances
    # fig3, ax3 = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # #Extraterrestrial irradiance
    # ax3[0].plot(spectral_profile_atmosphere[:, 0], spectral_profile_atmosphere[:, 1], color='cyan')
    # ax3[0].set_ylabel(r'Etr [W m$^{-2}$ nm$^{-1}$]')
    # ax3[0].set_title('Extraterrestrial Irradiance')
    # ax3[0].grid(True)

    # #Global tilt
    # ax3[1].plot(spectral_profile_atmosphere[:, 0], spectral_profile_atmosphere[:, 2], color='blue')
    # ax3[1].set_ylabel(r'Global Tilt [W m$^{-2}$ nm$^{-1}$]')
    # ax3[1].set_title('Global Tilt Irradiance')
    # ax3[1].grid(True)

    # #Direct + circumsolar
    # ax3[2].plot(spectral_profile_atmosphere[:, 0], spectral_profile_atmosphere[:, 3], color='purple')
    # ax3[2].set_ylabel(r'Direct + Circumsolar [W m$^{-2}$ nm$^{-1}$]')
    # ax3[2].set_title('Direct + Circumsolar Irradiance')
    # ax3[2].set_xlabel('Wavelength [nm]')
    # ax3[2].grid(True)
    # plt.tight_layout()
    # plt.savefig('spectral_profile_atmosphere.png')
    # plt.show()

    '''---- Task 2 b) ----'''

    #Reading the data file containing information about the S2A spectral bands
    s2_column_names = ['Band', 'Name', 'Central Wavelength (nm)', 'Bandwidth (nm)', 'Spatial Resolution (m)']
    s2a_spectrals = data_reader('s2spec.csv', s2_column_names, ',')[1] 
    s2a_central_wl = s2a_spectrals[:, 2] / 1e9 #Converted to SI unit meters
    s2a_bandwidths = s2a_spectrals[:, 3] / 1e9 #m

    #Converting wavelengths to meters for the 3 spectral profiles
    spectral_profile_grass_m = np.vstack([spectral_profile_grass[:, 0] / 1e6, spectral_profile_grass[:, 1]])
    spectral_profile_soil_m = np.vstack([spectral_profile_soil[:, 0] / 1e6, spectral_profile_soil[:, 1]])

    atmos_profile_percent = 100*spectral_profile_atmosphere[:, 2]/spectral_profile_atmosphere[:, 1]
    spectral_profile_atmos_m = np.vstack([spectral_profile_atmosphere[:, 0] / 1e9, atmos_profile_percent])

    #Plotting the values of grass as seen by the satellite, converting to nm for plotting
    fig4, ax4 = plt.subplots(figsize = (6, 4))
    ax4.plot(spectral_profile_grass[:, 0] * 1e3,  spectral_profile_sat(spectral_profile_grass_m, s2a_central_wl, s2a_bandwidths)[0]
             , color = 'green', label = 'spectral profile')
    ax4.vlines(s2a_central_wl * 1e9, 0, 60, colors= 'red', linestyles= 'dashed', label = 'central wavelength')
    ax4.set_xlim(None, 2500)
    ax4.set_xlabel('Wavelength [nm]')
    ax4.set_ylabel('Reflectance [%]')
    ax4.set_title('Spectral profile of grass as seen by S2A')
    ax4.grid(True)
    plt.legend()
    plt.savefig('grass_s2a.png')
    plt.show()

    #Plotting the values of bare soil as seen by the satellite, converting to nm for plotting
    fig5, ax5 = plt.subplots(figsize = (6, 4))
    ax5.plot(spectral_profile_soil[:, 0] * 1e3,  spectral_profile_sat(spectral_profile_soil_m, s2a_central_wl, s2a_bandwidths)[0]
             , color = 'saddlebrown', label = 'spectral profile')
    ax5.vlines(s2a_central_wl * 1e9, 0, 60, colors= 'red', linestyles= 'dashed', label = 'central wavelength')
    ax5.set_xlim(None, 2500)
    ax5.set_xlabel('Wavelength [nm]')
    ax5.set_ylabel('Reflectance [%]')
    ax5.set_title('Spectral profile of soil as seen by S2A')
    ax5.grid(True)
    plt.legend()
    plt.savefig('soil_s2a.png')
    plt.show()

    #Plotting the values of the atmosphere as seen by the satellite, converting to nm for plotting
    fig6, ax6 = plt.subplots(figsize = (6, 4))
    ax6.plot(spectral_profile_atmosphere[:, 0],  spectral_profile_sat(spectral_profile_atmos_m, s2a_central_wl, s2a_bandwidths)[0]
             , color = 'blue', label = 'spectral profile')
    ax6.vlines(s2a_central_wl * 1e9, 0, 100, colors= 'red', linestyles= 'dashed', label = 'central wavelength')
    ax6.set_xlim(None, 2500)
    ax6.set_xlabel('Wavelength [nm]')
    ax6.set_ylabel('Transmission [%]')
    ax6.set_title('Spectral profile of the atmosphere as seen by S2A')
    ax6.grid(True)
    plt.legend()
    plt.savefig('atmosphere_s2a.png')
    plt.show()

    '''---- Task 2 d) ----'''
    #To plot only the 10 m resolution bands we make an array containing only these bands, using masking
    s2a_resolutions = s2a_spectrals[:, 4]
    mask = (s2a_resolutions == 10)
    s2a_10m_resolution = s2a_spectrals[mask]

    s2a_central_wl_10m = s2a_10m_resolution[:, 2] / 1e9 #Converted to SI unit meters
    s2a_bandwidths_10m = s2a_10m_resolution[:, 3] / 1e9 #m

    fig7, ax7 = plt.subplots(figsize = (6, 4))

    # Plot grass profile
    ax7.plot(spectral_profile_grass[:, 0] * 1e3, spectral_profile_sat(spectral_profile_grass_m, s2a_central_wl_10m, 
                                                                    s2a_bandwidths_10m)[0],
                                                                    color = 'green', label = 'Grass')

    # Plot soil profile
    ax7.plot(spectral_profile_soil[:, 0] * 1e3, spectral_profile_sat(spectral_profile_soil_m, 
                                                                     s2a_central_wl_10m, s2a_bandwidths_10m)[0],
                                                                     color = 'saddlebrown', label = 'Soil')
    ax7.vlines(s2a_central_wl * 1e9, 0, 60, colors = 'red', linestyles = 'dashed', label = 'central wavelength')
    ax7.set_xlim(None, 2500)
    ax7.set_xlabel('Wavelength [nm]')
    ax7.set_ylabel('Reflectance [%]')
    ax7.set_title('Spectral Profiles of Grass and Soil as Seen by S2A with a 10m resolution')
    ax7.grid(True)
    ax7.legend()
    plt.savefig('grass_soil_s2a_10m.png')
    plt.show()


    '''---- Task 2 f) ----'''

    test_field_col_names = ['Name',	'B2', 'B3', 'B4', 'B8']
    test_field_data = pd.read_csv('TestFields.txt', comment = '%', header = None, delim_whitespace= True, 
                           names = test_field_col_names, skiprows = None).to_numpy()
    
    float_data_fields_percent = 100*test_field_data[1:, 1:].astype(float)

    avg_grass_percentage = []
    for field in float_data_fields_percent:
        grass_field = linear_mixing(field, 
                                    spectral_profile_sat(spectral_profile_soil_m, s2a_central_wl_10m, s2a_bandwidths_10m)[1],
                                    spectral_profile_sat(spectral_profile_grass_m, s2a_central_wl_10m, s2a_bandwidths_10m)[1])

        avg_grass_percentage.append(grass_field[1])
        print(grass_field[1])

    field_names = [f'Field {i}' for i in range(1,6)]

    fig8, ax8 = plt.subplots(figsize = (6, 4))
    ax8.bar(field_names, avg_grass_percentage, color = 'green')
    ax8.set_ylabel('Fraction of grass')
    ax8.set_title('Average Grass Fraction per Field')
    plt.savefig('grass_fraction_barplot.png')
    plt.show()

    avg_grass_percentage = []
    for field in float_data_fields_percent[:, [2, 3]]:
        grass_field = linear_mixing(field, 
                                    spectral_profile_sat(spectral_profile_soil_m, 
                                                         s2a_central_wl_10m[2:], s2a_bandwidths_10m[2:])[1],
                                    spectral_profile_sat(spectral_profile_grass_m,
                                                         s2a_central_wl_10m[2:], s2a_bandwidths_10m[2:])[1])

        avg_grass_percentage.append(grass_field[1])
        print(grass_field[1])

    fig9, ax9 = plt.subplots(figsize = (6, 4))
    ax9.bar(field_names, avg_grass_percentage, color = 'green')
    ax9.set_ylabel('Fraction of grass')
    ax9.set_title('Average Grass Fraction per Field')
    plt.savefig('grass_fraction_barplot2.png')
    plt.show()