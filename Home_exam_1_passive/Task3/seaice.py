import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import cartopy
import cartopy.crs as ccrs


def sea_ice_extent(T_b_water, T_b_ice, T_b_observed):

    '''Function that uses a two mixing model to calculate the sea ice concentration.

    Parameters:
        T_b_water (float or array): Brightness temperature for open water.
        T_b_ice (float or array): Brightness temperature for sea ice.
        T_b_observed (float or array): Observed brightness temperature from satellite data.

    Returns:
        SIC (float or array): Sea ice concentration
    '''

    SIC = np.clip((T_b_observed - T_b_water) / (T_b_ice - T_b_water), 0, 1)

    return SIC

def gradient_ratio(T_b1, T_b2):

    '''Function that calculates the gradient ratio.

    Parameters:
        T_b1(float or array): Brightness temperature for the first frequency.
        T_b2 (float or array): Brightness temperature for the second frequency.
        

    Returns:
        grad_ratio (float or array): Sea ice concentration
    '''

    grad_ratio = (T_b1 - T_b2) / (T_b1 + T_b2)

    return grad_ratio

if __name__ == '__main__':

    '''---- Task 3 b) ----'''

    data = xr.open_dataset('FYS-3001 amsr2File_20181112.nc')

    #Finding the data i wnat to use and converting to np arrays
    # latitudes = np.array(data['lat'].values)
    # longitudes = np.array(data['lon'].values)
    # bright_t_19GHz = np.array(data['tb19v'].values)[0]
    # bright_t_37GHz = np.array(data['tb37v'].values)[0]
    latitudes = data['lat'].values
    longitudes = data['lon'].values

    bright_t_19GHz = data['tb19v'][0].values
    bright_t_37GHz = data['tb37v'][0].values
    bright_t_85GHz = data['tb85v'][0].values

    #Using masking to find the right region
    mask = (latitudes >= 75) & (latitudes <= 85) & (longitudes>= -10) & (longitudes <= 50)
    indices = np.argwhere(mask)

    ymin, xmin = indices.min(axis = 0)
    ymax, xmax = indices.max(axis = 0) + 1        

    #Finding the coordinates and brightness temperatures at the range given
    svalbard_latitudes = latitudes[ymin:ymax, xmin:xmax]
    svalbard_longitudes =  longitudes[ymin:ymax, xmin:xmax]
    bright_t_svalbard_19GHz = bright_t_19GHz[ymin:ymax, xmin:xmax]
    bright_t_svalbard_37GHz = bright_t_37GHz[ymin:ymax, xmin:xmax]
    bright_t_svalbard_85GHz = bright_t_85GHz[ymin:ymax, xmin:xmax]

    svalbard_latitudes_nan = np.where(mask, latitudes, np.nan)
    svalbard_longitudes_nan = np.where(mask, longitudes, np.nan)
    bright_t_svalbard_19GHz_nan = np.where(mask, bright_t_19GHz, np.nan)
    bright_t_svalbard_37GHz_nan = np.where(mask, bright_t_37GHz, np.nan)
    bright_t_svalbard_85GHz_nan = np.where(mask, bright_t_85GHz, np.nan)

    #Finding max and min latitudes and longitudes
    lon_min_geo = np.nanmin(svalbard_longitudes)
    lon_max_geo = np.nanmax(svalbard_longitudes)

    lat_min_geo = np.nanmin(svalbard_latitudes)
    lat_max_geo = np.nanmax(svalbard_latitudes)

    #Finding the central latitude and longitude
    central_lat = (lat_max_geo - lat_min_geo) / 2
    central_long = (lon_max_geo - lon_min_geo) / 2

    # define projections for the figure and the original set of coordinates
    orig_projection = ccrs.PlateCarree()
    target_projection = ccrs.Stereographic(central_longitude = central_long,
                                           central_latitude = central_lat)

    vmin = min((np.nanmin(bright_t_19GHz), np.nanmin(bright_t_37GHz), np.nanmin(bright_t_85GHz)))
    vmax = max((np.nanmax(bright_t_19GHz), np.nanmax(bright_t_37GHz), np.nanmax(bright_t_85GHz)))

    #Plotting
    fig, ax = plt.subplots(1, 3, figsize = (12, 4), constrained_layout = True)

    #Plotting 18.7 GHz
    im1 = ax[0].imshow(bright_t_svalbard_19GHz_nan, cmap = 'magma', vmin = vmin, vmax = vmax)
    ax[0].set_title('Brightness Temperature (18.7 GHz)')
    ax[0].set_axis_off()

    #Plotting 36.5 GHz
    im2 = ax[1].imshow(bright_t_svalbard_37GHz_nan, cmap = 'magma', vmin = vmin, vmax = vmax)
    ax[1].set_title('Brightness Temperature (36.5 GHz)')
    ax[1].set_axis_off()

    #Plotting 85 GHz
    im3 = ax[2].imshow(bright_t_svalbard_85GHz_nan, cmap = 'magma', vmin = vmin, vmax = vmax)
    ax[2].set_title('Brightness Temperature (85.0 GHz)')
    ax[2].set_axis_off()

    cbar = fig.colorbar(im1, shrink = 0.8, pad = 0.02)
    cbar.set_label('Brightness Temperature [K]')
    plt.savefig('Svalbard_imshow_19GHz_37GHz.png')
    plt.show()


    '''---- Task 3 c) ----'''

    #Defining the emissivity for each frequency/polarization using the plot from the task
    water_emissivity_19V = 0.63
    water_emissivity_37V = 0.69
    water_emissivity_85V = 0.76

    FYI_emissivity_19V = 0.95
    FYI_emissivity_37V = 0.93
    FYI_emissivity_85V = 0.92

    MYI_emissivity_19V = 0.83
    MYI_emissivity_37V = 0.67
    MYI_emissivity_85V = 0.60

    #Defining the surface temperature / actual temperature
    surface_temp = 273 #K

    #Using this to calculate the brightness temperatre for ice and water
    T_i_19V = surface_temp * FYI_emissivity_19V
    T_o_19V = surface_temp * water_emissivity_19V

    #Finding the SIC
    sea_ice_19V = sea_ice_extent(T_o_19V, T_i_19V, bright_t_svalbard_19GHz_nan)

    #Plottign
    fig, ax = plt.subplots(figsize = (7, 5))
    im = ax.imshow(100* sea_ice_19V, cmap = 'magma', vmin = 0, vmax = 100)
    ax.set_title('Sea ice concentracion (18.7 GHz)')
    ax.set_axis_off()
    cbar = fig.colorbar(im, shrink = 0.8, pad = 0.02)
    cbar.set_label('SIC [%]')
    plt.savefig('SIC_svaldbard.png')
    plt.show()

    '''---- Task 3 d) ----'''

    T_FYI_19V = surface_temp * FYI_emissivity_19V
    T_FYI_37V = surface_temp * FYI_emissivity_37V
    T_FYI_85V = surface_temp * FYI_emissivity_85V

    T_i_37V = surface_temp * MYI_emissivity_37V
    T_i_85V = surface_temp * MYI_emissivity_85V

    calc_GR_FYI = gradient_ratio(T_FYI_37V, T_FYI_19V)
    calc_GR_MYI = gradient_ratio(T_i_37V, T_i_19V)

    calc_GR_FYI = gradient_ratio(T_FYI_85V, T_FYI_37V)
    calc_GR_MYI = gradient_ratio(T_i_85V, T_i_37V)

    # print(calc_GR_FYI)
    # print(calc_GR_MYI)

    observed_GR_37_19V = gradient_ratio(bright_t_svalbard_37GHz_nan, bright_t_svalbard_19GHz_nan)

    fig, ax = plt.subplots(figsize = (7, 5))
    im = ax.imshow(observed_GR_37_19V, cmap = 'magma')
    ax.set_title('Gradient ratio using 36.5 GHz and 18.7 GHz')
    ax.set_axis_off()
    cbar = fig.colorbar(im, pad = 0.02)
    cbar.set_label('GR')

    plt.savefig('grad_ratio_37_19.png')
    plt.show()


    # '''---- Task 3 e) ----'''

    #Calculating the observed GR using data with no nan values
    observed_GR_37_19V = gradient_ratio(bright_t_svalbard_37GHz, bright_t_svalbard_19GHz)

    #Maksing for the different materials
    water_mask = (observed_GR_37_19V >= 0.02) 
    FYI_mask = (observed_GR_37_19V > -0.02) & (observed_GR_37_19V < 0.02)
    MYI_mask = (observed_GR_37_19V <= -0.02)

    GR_37_19_changed = np.full_like(observed_GR_37_19V, fill_value = np.nan)
    
    #Setting the values as constant within the materials
    GR_37_19_changed[water_mask] = 0
    GR_37_19_changed[FYI_mask] = 1
    GR_37_19_changed[MYI_mask] = 2

    #Plotting, inspiration from 
    #https://github.com/CryosphereVirtualLab/public-notebooks/blob/main/S1_ice_water_classification/load_and_calibrate_S1_scene.ipynb
    cmap = colors.ListedColormap(['purple','cyan', 'blue'])
    fig2, ax2 = plt.subplots(figsize = (7, 5),  subplot_kw={'projection': target_projection})
    ax2.set_extent([lon_min_geo-1,lon_max_geo+1, lat_min_geo, lat_max_geo])
    ax2.add_feature(cartopy.feature.OCEAN)
    ax2.add_feature(cartopy.feature.LAND, facecolor = 'gray', zorder = 2)
    gl0 = ax2.gridlines(color='white',draw_labels=True, y_inline=False)
    ax2.coastlines(color='black')
    pc2 = ax2.pcolormesh(svalbard_longitudes, svalbard_latitudes, GR_37_19_changed, 
                         cmap = cmap, transform=orig_projection, zorder = 1)
    gl0.top_labels = False
    gl0.right_labels = False
    cbar = plt.colorbar(pc2, cmap = cmap, ticks = [0.33, 1, 1.66])
    cbar.ax.set_yticklabels(['Water', 'FYI', 'MYI'])

    plt.savefig('GR_3_color_fyi_myi_W')
    plt.show()