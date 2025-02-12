import numpy as np
import matplotlib.pyplot as plt

#Defining functions that find the reflection index for vertical and horisontal polarizes waves
# def reflection_coeff_h(theta, n_in, n_out):

#     sin_theta_t = n_in * np.sin(theta) / n_out
#     sin_theta_t = np.clip(sin_theta_t, -1, 1)  #Ensure valid arcsin input
#     theta_t = np.arcsin(sin_theta_t)
#     R_sq = np.zeros(len(theta))
    
#     for n in range(len(theta)):            
#         if np.sin(theta[n] + theta_t[n]) == 0:
#             R_sq[n] = 1
#         else:
#             R_sq[n] = ((np.sin(theta[n] - theta_t[n]))**2) / ((np.sin(theta[n] + theta_t[n]))**2)

#     return R_sq

# def reflection_coeff_v(theta, n_in, n_out):

#     sin_theta_t = n_in * np.sin(theta) / n_out
#     sin_theta_t = np.clip(sin_theta_t, -1, 1)  #Ensure valid arcsin input
#     theta_t = np.arcsin(sin_theta_t)
#     R_sq = np.zeros(len(theta))

#     for n in range(len(theta)):
#         if np.tan(theta[n] + theta_t[n]) == 0:
#             R_sq[n] = 1
#         else:
#             R_sq[n] = ((np.tan(theta[n] - theta_t[n]))**2) / ((np.tan(theta[n] + theta_t[n]))**2)
            
#     return R_sq

def reflection_coeff_h(theta, n_in, n_out):

    sin_theta_t = n_in * np.sin(theta) / n_out
    sin_theta_t = np.clip(sin_theta_t, -1, 1)  #Ensure valid arcsin input
    theta_t = np.arcsin(sin_theta_t)

    R_sq = ((np.sin(theta - theta_t))**2) / ((np.sin(theta + theta_t))**2)

    return R_sq


def reflection_coeff_v(theta, n_in, n_out):

    sin_theta_t = n_in * np.sin(theta) / n_out
    sin_theta_t = np.clip(sin_theta_t, -1, 1)  #Ensure valid arcsin input
    theta_t = np.arcsin(sin_theta_t)

    R_sq = ((np.tan(theta - theta_t))**2) / ((np.tan(theta + theta_t))**2)

    return R_sq


n_1 = 1
n_2 = 1.7
n_3 = 9

angles = np.linspace(0.001, np.pi/2, 10000)
angles2 = (angles/np.pi) * 180
wave1_h = reflection_coeff_h(angles, n_1, n_2)
wave2_h = reflection_coeff_h(angles, n_1, n_3)
wave1_v = reflection_coeff_v(angles, n_1, n_2)
wave2_v = reflection_coeff_v(angles, n_1, n_3)
 
fig, ax = plt.subplots(figsize = (9, 6))
ax.plot(angles2, wave1_h, label = r'$n = 1.7, |R_h|^2$', color = 'blue', linestyle = 'dashed')
ax.plot(angles2, wave2_h, label = r'$n = 9, |R_h|^2$', color = 'purple', linestyle = 'dashed')
ax.plot(angles2, wave1_v, label = r'$n = 1.7, |R_v|^2$', color = 'blue')
ax.plot(angles2, wave2_v, label = r'$n = 9, |R_v|^2$', color = 'purple')

ax.set_xlabel("Angle (deg)")
ax.set_ylabel("Reflection Coefficient (Power)")
ax.set_title("Reflection Coefficients for Different Polarizations and Refractive Indices")
ax.grid(True)

plt.legend()
plt.show()