    # -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:45:02 2024

@author: jonas
"""

#%% SETUP
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt

#%% METEO DATA
# MeteoStation Parse Parameters 
# The table below maps the physical parameter to its given axis. 
 
# Axis | Parameter | NetCDF Parameter Name | Unit | Description 
# x | Time | time | seconds since 1970-01-01 00:00:00 | Time 
# y | Air Temperature | AirTC | °C | Air temperature 
# y1 | Relative Humidity | RH | % | Relative humidity (RH) is the ratio of the partial pressure of water vapor to the equilibrium vapor pressure of water at a given temperature. Relative humidity depends on temperature and the pressure of the system of interest. The same amount of water vapor results in higher relative humidity in cool air than warm air. A related parameter is the dew point. 
# y2 | Solar Irradiance | Slrw | W m-2 | The quantity with standard name solar_irradiance, often called Total Solar Irradiance (TSI), is the radiation from the sun integrated over the whole electromagnetic spectrum and over the entire solar disk. The quantity applies outside the atmosphere, by default at a distance of one astronomical unit from the sun, but a coordinate or scalar coordinate variable of distance_from_sun can be used to specify a value other than the default. "Irradiance" means the power per unit area (called radiative flux in other standard names), the area being normal to the direction of flow of the radiant energy. 
# y3 | Wind Speed | WS | m s-1 | Speed is the magnitude of velocity. Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component. (Vertical motion in the atmosphere has the standard name upward_air_velocity.) The wind speed is the magnitude of the wind velocity. 
# y4 | Wind Direction | WindDir | deg | Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component. (Vertical motion in the atmosphere has the standard name upward_air_velocity.) In meteorological reports, the direction of the wind vector is usually (but not always) given as the direction from which it is blowing (wind_from_direction) (westerly, northerly, etc.). In other contexts, such as atmospheric modelling, it is often natural to give the direction in the usual manner of vectors as the heading or the direction to which it is blowing (wind_to_direction) (eastward, southward, etc.) "from_direction" is used in the construction X_from_direction and indicates the direction from which the velocity vector of X is coming. 
# y5 | Rainfall | Rain | mm | "Amount" means mass per unit area. The construction thickness_of_[X_]rainfall_amount means the accumulated "depth" of rainfall i.e. the thickness of a layer of liquid water having the same mass per unit area as the rainfall amount. 
# y6 | Air Pressure | BP | mbar | Air pressure is the force per unit area which would be exerted when the moving gas molecules of which the air is composed strike a theoretical surface of any orientation. 
# y7 | Binary Error Mask | time_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y8 | Binary Error Mask | Batt_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y9 | Binary Error Mask | Ptemp_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y10 | Binary Error Mask | AirTC_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y11 | Binary Error Mask | RH_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y12 | Binary Error Mask | Slrw_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y13 | Binary Error Mask | Slrm_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y14 | Binary Error Mask | WS_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y15 | Binary Error Mask | WindDir_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y16 | Binary Error Mask | Rain_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  
# y17 | Binary Error Mask | BP_qual | 0 = nothing to report, 1 = more investigation | data_binary_mask has 1 where condition X is met, 0 elsewhere. 0 = high quality, 1 = low quality.  

# Directory containing the NetCDF files
data_dir = 'C:/Users/jonas/Documents/uni/mastor2/ML/datasets/MeteoStation'

# Prepare a list to store data from all files
data_list = []

# Loop through each NetCDF file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.nc'):
        file_path = os.path.join(data_dir, filename)

        # Open the NetCDF file
        with Dataset(file_path, 'r') as nc_file:
            # Extract variables
            unix_time = nc_file.variables['time'][:]
            air_temp = nc_file.variables['AirTC'][:]
            solar_radiation = nc_file.variables['Slrw'][:]
            wind_speed = nc_file.variables['WS'][:]

            # Convert Unix time to datetime
            datetime_times = [datetime.utcfromtimestamp(t) for t in unix_time]

            # Combine into a structured array and append to list
            file_data = np.array(list(zip(datetime_times, air_temp, solar_radiation, wind_speed)),
                                 dtype=[('time', 'O'), ('AirTC', 'f4'), ('SlrW', 'f4'), ('WS', 'f4')])
            data_list.append(file_data)

# Combine all data into a single NumPy array
meteo = np.concatenate(data_list)

# Verify contents
print(meteo[:5])  # Display the first 5 records

#%% THETIS DATA
# Datalakes Parse Parameters 
# The table below maps the physical parameter to its given axis. 
 
# Axis | Parameter | NetCDF Parameter Name | Unit | Description 
# y | Depth | depth | m | Depth is the vertical distance below the surface. 
# x | Time | time | seconds since 1970-01-01 00:00:00 | Time 
# z | Water Temperature | temp | degC | Lake water temperature is the in situ temperature of the lake water. To specify the depth at which the temperature applies use a vertical coordinate variable or scalar coordinate variable. There are standard names for lake_surface_temperature, lake_surface_skin_temperature, lake_surface_subskin_temperature and lake_surface_foundation_temperature which can be used to describe data located at the specified surfaces. 
# z1 | Conductivity | cond | microS/cm | Electrical conductivity (EC) estimates the amount of total dissolved salts (TDS), or the total amount of dissolved ions in the water. 
# z2 | Salinity | sal | mg/l | Salinity is the saltiness or amount of salt dissolved in a body of water, called saline water. 
# z3 | Conductivity (20degC) | cond20 | microS/cm | Electrical conductivity (EC) estimates the amount of total dissolved salts (TDS), or the total amount of dissolved ions in the water. This is normalized to 20 degC. 
# z4 | Dissolved Oxygen | do | mg/L | Dissolved oxygen refers to the level of free, non-compound oxygen present in water or other liquids. 
# z5 | Oxygen Saturation | dosat | %sat | Fractional saturation is the ratio of some measure of concentration to the saturated value of the same quantity. 
# z6 | PAR | par | umol s-1 m-2 | Designates the spectral range (wave band) of solar radiation from 400 to 700 nanometers that photosynthetic organisms are able to use in the process of photosynthesis. 
# z7 | Backscattering | bb440 | m-1 | Scattering of radiation is its deflection from its incident path without loss of energy. Backwards scattering refers to the sum of scattering into all backward angles i.e. scattering_angle exceeding pi/2 radians. A scattering_angle should not be specified with this quantity.  
# z8 | Backscattering | bb532 | m-1 | Scattering of radiation is its deflection from its incident path without loss of energy. Backwards scattering refers to the sum of scattering into all backward angles i.e. scattering_angle exceeding pi/2 radians. A scattering_angle should not be specified with this quantity.  
# z9 | Backscattering | bb630 | m-1 | Scattering of radiation is its deflection from its incident path without loss of energy. Backwards scattering refers to the sum of scattering into all backward angles i.e. scattering_angle exceeding pi/2 radians. A scattering_angle should not be specified with this quantity.  
# z10 | Backscattering | bb700 | m-1 | Scattering of radiation is its deflection from its incident path without loss of energy. Backwards scattering refers to the sum of scattering into all backward angles i.e. scattering_angle exceeding pi/2 radians. A scattering_angle should not be specified with this quantity.  
# z11 | Chlorophyll A | chla | mg m-3 | 'Mass concentration' means mass per unit volume and is used in the construction mass_concentration_of_X_in_Y, where X is a material constituent of Y. A chemical or biological species denoted by X may be described by a single term such as 'nitrogen' or a phrase such as 'nox_expressed_as_nitrogen'. Chlorophylls are the green pigments found in most plants, algae and cyanobacteria; their presence is essential for photosynthesis to take place. There are several different forms of chlorophyll that occur naturally. All contain a chlorin ring (chemical formula C20H16N4) which gives the green pigment and a side chain whose structure varies. The naturally occurring forms of chlorophyll contain between 35 and 55 carbon atoms. Chlorophyll-a is the most commonly occurring form of natural chlorophyll. The chemical formula of chlorophyll-a is C55H72O5N4Mg. 
# z12 | CDOM | cdom | ppb | null 
# z13 | Absorption | a700 | m-1 | null 
# z14 | Scattering | b700 | m-1 | null 
# z15 | Attenuation | c700 | m-1 | null 
# z16 | Absorption Line Height | aLH550 | m-1 | null 
# z17 | Absorption Line Height | aLH676 | m-1 | null 
# z18 | Spectral Attenuation Slope | Sk |  | null 

# Directory containing the NetCDF files
data_dir = 'C:/Users/jonas/Documents/uni/mastor2/ML/datasets/ThetisData'

# Prepare a list to store data from all files
data_list = []

# Loop through each NetCDF file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.nc'):
        file_path = os.path.join(data_dir, filename)
        # Open the NetCDF file
        with Dataset(file_path, 'r') as nc_file:
            # Extract variables
            unix_time = nc_file.variables['time'][:]
            water_temp = nc_file.variables['temp'][:]
            chla = nc_file.variables['chla'][:] if 'chla' in nc_file.variables else None
            depth = nc_file.variables['depth'][:] if 'depth' in nc_file.variables else None
            depth_mask = (depth >= 0) & (depth <= 10)

            # Convert Unix time to datetime
            datetime_times = [datetime.utcfromtimestamp(t) for t in unix_time]
            
            # Prepare lists for storing relevant data
            water_temp_surface = []
            chla_surface = []
            chla_0_10m = []

            for i in range(len(unix_time)):
                # Get surface water temperature and chlorophyll measurement
                if water_temp is not None:
                    water_temp_surface.append(water_temp[i][0])  # Assuming 0th index is surface
                else:
                    water_temp_surface.append(np.nan)  # Append NaN if no temp data

                if chla is not None and chla[i].size > 0:
                    chla_surface.append(chla[i][0])  # Assuming 0th index is surface
                else:
                    chla_surface.append(np.nan)  # Append NaN if no chla data
                        
                if chla is not None and chla[:, i].size > 0 and depth is not None and depth.size > 0:
                    # Apply the mask to the depth dimension of chla for the current time step
                    valid_chla = chla[depth_mask, i]  # Chlorophyll values within 0-10 m
                    valid_depths = depth[depth_mask]  # Depth values within 0-10 m
            
                    if valid_depths.size > 1:
                        # Calculate depth intervals
                        depth_intervals = np.diff(valid_depths, prepend=valid_depths[0])  # Include first depth
                        chla_integrated = np.sum(valid_chla * depth_intervals)  # Integrate chlorophyll-a
                        chla_0_10m.append(chla_integrated)
                    else:
                        chla_0_10m.append(np.nan)  # Append NaN if no valid depths
                else:
                    chla_0_10m.append(np.nan)  # Append NaN if no chla or depth data
            # Combine into a structured array and append to list
            file_data = np.array(list(zip(datetime_times, water_temp_surface, chla_surface, chla_0_10m)),
                                 dtype=[('time', 'O'), ('water_temp_surface', 'f4'), ('chla_surface', 'f4'), ('chla_0_10m', 'f4')])
            data_list.append(file_data)

# Combine all data into a single NumPy array
tethis = np.concatenate(data_list)

# Verify contents
print(tethis[:5])  # Display the first 5 records


#%% PLOTTING DATA

# Extract the necessary data from the structured array
wind_speed_data = meteo['WS']
water_temp_surface_data = tethis['water_temp_surface']
solar_radiation_data = meteo['SlrW']
chla_surface_data = tethis['chla_surface']
chla_0_10m_data = tethis['chla_0_10m']

# Create a figure with subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Plot Wind Speed
axs[0].plot(meteo['time'], wind_speed_data, marker='o', linestyle='-', color='b')
axs[0].set_title('Wind Speed Over Time')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Wind Speed (m/s)')
axs[0].grid()

# Plot Water Temperature (Surface and 15m)
axs[1].plot(tethis['time'], water_temp_surface_data, marker='o', linestyle='-', color='g', label='Surface Water Temp')
axs[1].set_title('Water Temperature Over Time')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Temperature (°C)')
axs[1].grid()
axs[1].legend()

# Plot Solar Radiation
axs[2].plot(meteo['time'], solar_radiation_data, marker='o', linestyle='-', color='orange')
axs[2].set_title('Solar Radiation Over Time')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Solar Radiation (W/m²)')
axs[2].grid()

# Plot Chlorophyll-a Concentrations (Surface and 15m)
axs[3].plot(tethis['time'], chla_surface_data, marker='o', linestyle='-', color='b', label='Chla Surface')
axs[3].plot(tethis['time'], chla_0_10m_data, marker='o', linestyle='-', color='g', label='Chla between 0 and 10m depth')
axs[3].set_title('Chlorophyll-a Concentration Over Time')
axs[3].set_xlabel('Time')
axs[3].set_ylabel('Chlorophyll-a (mg/m³)')
axs[3].grid()
axs[3].legend()

# Adjust layout
plt.tight_layout()
plt.show()

#%% MERGING DATA
# Create a new structured array to hold the merged data
merged_dtype = np.dtype([
    ('time', 'O'),
    ('AirTC', 'f4'),
    ('SlrW', 'f4'),
    ('WS', 'f4'),
    ('water_temp_surface', 'f4'),
    ('chla_surface', 'f4'),
    ('chla_0_10m', 'f4')
])

# Prepare an empty list to hold merged records
merged_data = []

# Loop through each entry in the thetis array
for thetis_entry in tethis:
    # Find the closest time in the meteo array
    time_diff = np.abs(meteo['time'] - thetis_entry['time'])
    closest_index = time_diff.argmin()  # Get index of closest time

    # Get the closest meteorological data
    closest_meteo_entry = meteo[closest_index]

    # Append the merged entry
    merged_data.append((thetis_entry['time'], 
                        closest_meteo_entry['AirTC'], 
                        closest_meteo_entry['SlrW'], 
                        closest_meteo_entry['WS'], 
                        thetis_entry['water_temp_surface'], 
                        thetis_entry['chla_surface'], 
                        thetis_entry['chla_0_10m']))

# Convert the list of tuples to a structured array
merged_array = np.array(merged_data, dtype=merged_dtype)

# Verify contents
print(merged_array[:5])  # Display the first 5 records

#%% SAVING FOR FUTURE USE

df = pd.DataFrame(merged_array)
df.to_csv("C:/Users/jonas/Documents/uni/mastor2/ML/datasets/data.csv", index = False)
