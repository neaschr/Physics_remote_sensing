import numpy as np
import scipy.constants as spc
import scipy.integrate as i

#Defining constants and variables
T = 6000 #K
c = spc.c #m/s
h = spc.Planck #Js
k = spc.k #J/K

def sol_en(low_lim, high_lim, temp):
    '''Function that return the blackbody energy between to wavelengths
    by integrating Plancks formula over the wavelengths given. 
    low_lim: The lowest wavelength
    high_lim: The highest wavelength
    T: temperature of the black body'''

    S = lambda x: (2 * np.pi * h * c**2) / (x**5) * 1 / (np.exp(c * h / (k * x * temp)) - 1) #Plancks formula
    energy = i.quad(S, low_lim, high_lim)

    return energy

ultraviolet_energy = sol_en(0, 0.4E-6, T)[0] #Finding the energy in the UV spectrum
visible_energy = sol_en(0.4E-6, 0.7E-6, T)[0] #Finding the energy in the visible spectrum
infrared_energy = sol_en(0.7E-6, 10E-6, T)[0] #Finding the energy in the infrared spectrum
therm_submil_energy = sol_en(10E-6, 0.003, T)[0] #Finding the energy in the thermal infrared and submillimeter
microwave_energy = sol_en(0.003, 0.3, T)[0] #Finding the energy in the microwave spectrum

total_energy = sol_en(0, np.inf, T)[0]

#Defining a dictionary for the values of energy from the spectrum
specs = {'UV': ultraviolet_energy, 'Visible': visible_energy, 'Infrared': infrared_energy, 
         'Thermal infrared and submillimeter': therm_submil_energy, 'Microwave' : microwave_energy}

#Finding and printing the percentage of the total energy for each energy
for k, v in specs.items(): 
    print(f'The energy from the {k} spectrum is {v/total_energy} percent of the total solar energy emittance')