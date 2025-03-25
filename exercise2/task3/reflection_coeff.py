import numpy as np
from matplotlib import pyplot as plt

def normal_reflection_coeff(n_r: float, alpha: float,
                            nu_div: np.ndarray, nu_s_rel: float) -> np.ndarray:


    n_i = alpha * np.exp(-((nu_div - 1)**2 / (nu_div**2)) * (nu_s_rel**2))
    ref_coeff = ((n_r - 1)**2 + n_i**2) / ((n_r + 1)**2 + n_i**2)

    return ref_coeff

if __name__ == '__main__':

    #Defining the values we will use
    n_r_value = 3
    alpha_value = 0.1
    nu_nu_s = 0.05
    nu_div_nu0 = np.linspace(0.5, 1.2, 1000)

    #Plotting
    fig, ax = plt.subplots(figsize = (9, 6))
    ax.plot(nu_div_nu0, normal_reflection_coeff(n_r_value, alpha_value, nu_div_nu0, nu_nu_s))
    ax.set_xlabel(r'$\nu / \nu_0$')
    ax.set_ylabel(r'Reflection coefficient')
    ax.set_title('Reflection coefficient')
    ax.grid(True)

    plt.savefig('Reflectioncoefftask3.png')
    plt.show()


    # Parameters
    alpha = 0.1
    nu_nu_s = 0.05

    # Frequency range: avoid zero to prevent division by zero
    x = np.linspace(0.8, 1.2, 1000)  # x = nu / nu_0

    # Proper ni expression
    n_i = alpha * np.exp(-((x - 1)**2 / x**2) * nu_nu_s**2)

    # Plot n_i
    plt.figure(figsize=(8, 5))
    plt.plot(x, n_i)
    plt.xlabel(r'$\nu / \nu_0$', fontsize=14)
    plt.ylabel(r'$n_i(\nu)$', fontsize=14)
    plt.title(r'Imaginary Part of Refractive Index', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()