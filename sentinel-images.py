# # -*- coding: utf-8 -*-
# """
# Created on Sun Oct 13 15:45:16 2024

# @author: jonas
# """

# import rasterio
# import matplotlib.pyplot as plt
# import numpy as np

# # File path to the image
# file_path = 'C:/Users/jonas/Documents/uni/mastor2/ML/datasets/Sentinel-2/Sentinel2_2019-02-19_11.tif'

# # Open the image using rasterio
# with rasterio.open(file_path) as src:
#     # Read the RGB bands (Band 4: Red, Band 3: Green, Band 2: Blue)
#     red = src.read(4)   # Band 4: Red
#     green = src.read(3) # Band 3: Green
#     blue = src.read(2)  # Band 2: Blue
    
#     # Read the NIR band (Band 8: Near-Infrared)
#     nir = src.read(8)   # Band 8: NIR

# # Normalize the RGB bands to 0-1 for visualization
# red_normalized = red / red.max()
# green_normalized = green / green.max()
# blue_normalized = blue / blue.max()

# # Stack the normalized RGB bands to create a 3-band image (height x width x 3)
# rgb = np.dstack((red_normalized, green_normalized, blue_normalized))

# # Display the RGB composite image
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)  # First subplot (left)
# plt.imshow(rgb)
# plt.title("Sentinel-2 RGB Composite")
# plt.axis('off')

# # Display the NIR band
# plt.subplot(1, 2, 2)  # Second subplot (right)
# plt.imshow(nir, cmap='gray')  # Display NIR as grayscale
# plt.title("Near-Infrared (NIR) Band")
# plt.axis('off')

# plt.tight_layout()
# plt.show()


# # Import necessary libraries
# import rasterio
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime
# import os

# # List of file paths for multiple images (e.g., different timestamps)
# sentinel_folder = 'C:/Users/jonas/Documents/uni/mastor2/ML/datasets/Sentinel-2/'
# file_paths = [os.path.join(sentinel_folder, f) for f in os.listdir(sentinel_folder) if f.endswith('.tif')]

# # Initialize an empty list to store the results
# chla_data = []

# # Loop over each file
# for file_path in file_paths:
#     # Extract the timestamp from the file name or other source
#     # Assuming filename format "Sentinel2_YYYY-MM-DD_HH.tif"
#     timestamp_str = file_path.split('/')[-1].split('_')[1]  # Extract date
#     time = datetime.strptime(timestamp_str, "%Y-%m-%d")  # Convert to datetime

#     # Open the image using rasterio
#     with rasterio.open(file_path) as src:
#         # Read the Red and NIR bands
#         red = src.read(4).astype(float)
#         nir = src.read(8).astype(float)
        
#         # Avoid division by zero
#         ndci = (nir - red) / (nir + red + 1e-10)  # Calculate NDCI
        
#         # Calculate the mean NDCI value as a proxy for chlorophyll-a
#         chla_mean = np.nanmean(ndci)  # Use nanmean to ignore any NaN values
    
#     # Append the timestamp and chlorophyll-a value to the list
#     chla_data.append({'time': time, 'chla': chla_mean})

# # Convert the list of dictionaries to a DataFrame
# chla_df = pd.DataFrame(chla_data)

# # Display the DataFrame
# print(chla_df)

# # Optional: Plot the time series of chlorophyll-a concentration
# plt.figure(figsize=(10, 5))
# plt.plot(chla_df['time'], chla_df['chla'], color='b')
# plt.xlabel("Time")
# plt.ylabel("Chlorophyll-a (NDCI)")
# plt.title("Chlorophyll-a Concentration Over Time")
# plt.grid(True)
# plt.show()

# #%% SAVING FOR FUTURE USE

# df = pd.DataFrame(chla_df)
# df.to_csv("C:/Users/jonas/Documents/uni/mastor2/ML/datasets/sat-data.csv", index = False)

#%% yee

# Import necessary libraries
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# List of file paths for multiple images (e.g., different timestamps)
sentinel_folder = 'C:/Users/jonas/Documents/uni/mastor2/ML/datasets/Sentinel-2/'
file_paths = [os.path.join(sentinel_folder, f) for f in os.listdir(sentinel_folder) if f.endswith('.tif')]

# Initialize an empty list to store the results
chla_data = []

# Loop over each file
for file_path in file_paths:
    # Extract the timestamp from the file name or other source
    # Assuming filename format "Sentinel2_YYYY-MM-DD_HH.tif"
    timestamp_str = file_path.split('/')[-1].split('_')[1]  # Extract date
    time = datetime.strptime(timestamp_str, "%Y-%m-%d")  # Convert to datetime

    # Open the image using rasterio
    with rasterio.open(file_path) as src:
        # Read the Red and NIR bands
        red = src.read(4).astype(float)
        nir = src.read(8).astype(float)
        
        # Define the central region (e.g., 20% of the image size)
        height, width = red.shape
        center_height = height // 2
        center_width = width // 2
        region_size = min(height, width) // 5  # Define size as 20% of the smallest dimension
        
        # Extract the central region
        red_central = red[center_height - region_size:center_height + region_size,
                          center_width - region_size:center_width + region_size]
        nir_central = nir[center_height - region_size:center_height + region_size,
                          center_width - region_size:center_width + region_size]
        
        # Avoid division by zero
        ndci_central = (nir_central - red_central) / (nir_central + red_central + 1e-10)  # Calculate NDCI
        
        # Calculate the mean NDCI value for the central region
        chla_mean_central = np.nanmean(ndci_central)  # Use nanmean to ignore any NaN values
    
    # Append the timestamp and chlorophyll-a value to the list
    chla_data.append({'time': time, 'chla': chla_mean_central})

# Convert the list of dictionaries to a DataFrame
chla_df = pd.DataFrame(chla_data)

# Display the DataFrame
print(chla_df)

# Optional: Plot the time series of chlorophyll-a concentration
plt.figure(figsize=(10, 5))
plt.plot(chla_df['time'], chla_df['chla'], color='b')
plt.xlabel("Time")
plt.ylabel("Chlorophyll-a (NDCI)")
plt.title("Chlorophyll-a Concentration Over Time")
plt.grid(True)
plt.show()

#%% SAVING FOR FUTURE USE

df = pd.DataFrame(chla_df)
df.to_csv("C:/Users/jonas/Documents/uni/mastor2/ML/datasets/sat-data.csv", index=False)