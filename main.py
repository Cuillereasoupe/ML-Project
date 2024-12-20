# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:48:02 2024

@author: jonas
"""
#%% SETUP
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import shap
import xgboost as xgb

with open('C:/Users/jonas/Documents/uni/mastor2/ML/datasets/data.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
data = pd.DataFrame(data, columns = data[0])
data = data.iloc[1:].reset_index(drop=True)

data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.replace('', np.nan)
data = data.astype(float)
data = data.resample('D').mean()

data = data.dropna(subset=['AirTC', 'SlrW', 'WS', 'water_temp_surface', 'chla_0_10m'])

with open('C:/Users/jonas/Documents/uni/mastor2/ML/datasets/sat-data.csv', 'r') as f:
    reader = csv.reader(f)
    satdata = list(reader)
    
satdata = pd.DataFrame(satdata, columns = satdata[0])
satdata = satdata.iloc[1:].reset_index(drop=True)

#%% IMPUTATION
# Assuming `data` is indexed by a datetime index
data.index = pd.to_datetime(data.index)

# Convert `satdata` to pandas DataFrame
satdata['time'] = pd.to_datetime(satdata['time'])
satdata['chla'] = pd.to_numeric(satdata['chla'])

# Ensure `satdata` has unique dates by averaging duplicate values for the same day
satdata_daily = satdata.groupby(satdata['time'].dt.date).mean()

def find_closest_date(row_time, sat_times):
    return sat_times.iloc[(sat_times - row_time).abs().argmin()]

# Match each row in `data` with the closest date in `satdata`
sat_times = satdata_daily['time']
data['sat_chla'] = data.index.map(lambda x: 
    satdata_daily.loc[
        satdata_daily['time'] == find_closest_date(x, sat_times),
        'chla'
    ].values[0]
)
    
#%% SPLITTING DATA RANDOM
# Set random seed for reproducibility
random_seed = 30

# Split the dataset into features (X) and target variable (y) for modeling.
# Use `train_test_split` to create train, validation, and test datasets, ensuring reproducibility.
# Train: 70%, Validation: 15%, Test: 15%

# Features and target
X = data[['AirTC', 'SlrW', 'WS', 'water_temp_surface', 'sat_chla']]
y = data['chla_0_10m']

# First, split data into training (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_seed)

# Then, split the temp data into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)

# Print shapes to verify
print("Training data:", X_train.shape, y_train.shape)
print("Validation data:", X_val.shape, y_val.shape)
print("Testing data:", X_test.shape, y_test.shape)


#%% Kernel density plots
# Convert data to DataFrames for easier access
feature_names = X_train.columns
titles = [
    "Air temperature", "Solar irradiance", "Wind speed", 
    "Surface water temperature", "NDCI", "Output chl-a"
]

train_df = pd.DataFrame(X_train, columns=feature_names)
val_df = pd.DataFrame(X_val, columns=feature_names)
test_df = pd.DataFrame(X_test, columns=feature_names)

plt.rcParams.update({
    'font.size': 16,           # Default font size for all text
    'axes.titlesize': 20,      # Font size for subplot titles
    'axes.labelsize': 18,      # Font size for x and y labels
    'legend.fontsize': 16,     # Font size for legend
    'xtick.labelsize': 16,     # Font size for x-axis tick labels
    'ytick.labelsize': 16      # Font size for y-axis tick labels
})

# Create kernel density plots for each feature
plt.figure(figsize=(15, 10))
for i, (feature, title) in enumerate(zip(feature_names, titles)):
    plt.subplot(2, 3, i + 1)  # Adjust grid size based on the number of features
    sns.kdeplot(train_df[feature], label="Train", fill=False, linestyle='--', linewidth=2)
    sns.kdeplot(val_df[feature], label="Validation", fill=True, alpha=0.3)
    sns.kdeplot(test_df[feature], label="Test", fill=True, alpha=0.3)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Density")

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for the legend

# Kernel density plot for the target variable
plt.subplot(2, 3, i + 2)  # Adjust grid size for the target variable
sns.kdeplot(y_train, label="Train", fill=False, linestyle='--', linewidth=2)
sns.kdeplot(y_val, label="Validation", fill=True, alpha=0.3)
sns.kdeplot(y_test, label="Test", fill=True, alpha=0.3)
plt.title("Output chl-a")
plt.xlabel("Chl-a in the Water")
plt.ylabel("Density")

handles, labels = plt.gca().get_legend_handles_labels()
plt.figlegend(handles, labels, loc='lower center', ncol=3, fontsize=16)
plt.show()


#%% NORMALIZING
# Normalize the training data
scaler = StandardScaler()

# Fit the scaler on training data and transform test- and validation-sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#%% Check data quality
# Check for outliers using boxplot


# Create a figure with three vertical subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 18))
# Plot for Training Data
sns.boxplot(data=X_train_scaled, orient="h", ax=axes[0])
axes[0].set_title("Boxplot of Features in Training Data (Outlier Check)")
# Plot for Validation Data
sns.boxplot(data=X_val_scaled, orient="h", ax=axes[1])
axes[1].set_title("Boxplot of Features in Validation Data (Outlier Check)")
# Plot for Testing Data
sns.boxplot(data=X_test_scaled, orient="h", ax=axes[2])
axes[2].set_title("Boxplot of Features in Testing Data (Outlier Check)")
# Adjust layout
plt.tight_layout()
# Show the plot
plt.show()

# Check for feature scaling (summary statistics)
print("Summary statistics for training features:")
print(X_train.describe())

#%% RANDOM FOREST with grid search

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],         # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],        # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],        # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],          # Minimum samples required at a leaf node
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider at each split
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                            cv=5, scoring='neg_mean_squared_error', 
                            verbose=2, n_jobs=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Extract the best parameters and model
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict with the tuned model
y_train_pred_best = best_rf.predict(X_train)
y_val_pred_best = best_rf.predict(X_val)
y_test_pred_best = best_rf.predict(X_test)

# Calculate metrics
train_mse_best = mean_squared_error(y_train, y_train_pred_best)
train_r2_best = r2_score(y_train, y_train_pred_best)
val_mse_best = mean_squared_error(y_val, y_val_pred_best)
val_r2_best = r2_score(y_val, y_val_pred_best)
test_mse_best = mean_squared_error(y_test, y_test_pred_best)
test_r2_best = r2_score(y_test, y_test_pred_best)

print("\nTuned Random Forest Results:")
print(f"Training MSE: {train_mse_best:.4f}, R^2: {train_r2_best:.4f}")
print(f"Validation MSE: {val_mse_best:.4f}, R^2: {val_r2_best:.4f}")
print(f"Testing MSE: {test_mse_best:.4f}, R^2: {test_r2_best:.4f}")

#%% XGBoost
# Parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Initialize XGBoost model
xgb_model = xgb.XGBRegressor(random_state=random_seed)

# Grid Search with Cross Validation
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    scoring='r2',
    cv=5,
    verbose=1
)

# Fit the grid search model
grid_search_xgb.fit(X_train_scaled, y_train)

# Best parameters and best estimator
best_params_xgb = grid_search_xgb.best_params_
best_xgb = grid_search_xgb.best_estimator_

# Predictions using the best model
y_train_pred = best_xgb.predict(X_train_scaled)
y_val_pred = best_xgb.predict(X_val_scaled)
y_test_pred = best_xgb.predict(X_test_scaled)

# Evaluation metrics
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Print results
print("\nXGBoost Grid Search Results:")
print(f"Best Parameters: {best_params_xgb}")
print(f"Training MSE: {train_mse:.4f}, R^2: {train_r2:.4f}")
print(f"Validation MSE: {val_mse:.4f}, R^2: {val_r2:.4f}")
print(f"Testing MSE: {test_mse:.4f}, R^2: {test_r2:.4f}")
#%% FEATURE IMPORTANCE
feature_importances = best_xgb.feature_importances_
features = X_train.columns
features = ["Air temperature", "Solar irradiance", "Wind speed", "Surface water temperature", "NDCI"]
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
#plt.title("Feature Importance from the XGBoost model")
plt.show()

# Correlation Matrix with custom variable names
correlation_matrix = data[['AirTC', 'SlrW', 'WS', 'water_temp_surface', 'sat_chla', 'chla_surface']].corr()
# Custom column and row titles for the correlation matrix
correlation_titles = ['Air temp', 'Solar irradiance', 'Wind speed', 'Water temp', 'Satellite chl-a', 'Chl-a surface']
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=correlation_titles, yticklabels=correlation_titles)
plt.title("Correlation Matrix")
plt.show()

explainer = shap.Explainer(best_xgb, X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, feature_names=features)

#%% PLOTTING DATA

# Extract the necessary data from the structured array
AirTC_data = data['AirTC']
wind_speed_data = data['WS']
water_temp_surface_data = data['water_temp_surface']
solar_radiation_data = data['SlrW']
#chla_surface_data = data['chla_surface']
chla_0_10m_data = data['chla_0_10m']
sat_chla_data = data['sat_chla']

# Create a figure with subplots
fig, axs = plt.subplots(6, 1, figsize=(10, 12))

# Plot Wind Speed
axs[0].plot(data.index, wind_speed_data, linestyle='-', color='b')
axs[0].set_title('Wind Speed Over Time')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Wind Speed (m/s)')
axs[0].grid()

# Plot Water Temperature (Surface and 15m)
axs[1].plot(data.index, water_temp_surface_data, linestyle='-', color='g', label='Surface Water Temp')
#axs[1].plot(data.index, water_temp_15m_data, marker='o', linestyle='-', color='r', label='Water Temp at 15m')
axs[1].set_title('Water Temperature Over Time')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Temperature (°C)')
axs[1].grid()
axs[1].legend()

# Plot Solar Radiation
axs[2].plot(data.index, solar_radiation_data, linestyle='-', color='orange')
axs[2].set_title('Solar Radiation Over Time')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Solar Radiation (W/m²)')
axs[2].grid()

# Plot Chlorophyll-a Concentrations (Surface)
#axs[3].plot(data.index, chla_surface_data, linestyle='-', color='b', label='Chla Surface')
axs[3].plot(data.index, chla_0_10m_data, marker='o', linestyle='-', color='g', label='Chla 0-10m depth')
axs[3].set_title('Chlorophyll-a Concentration Over Time')
axs[3].set_xlabel('Time')
axs[3].set_ylabel('Chlorophyll-a (mg/m³)')
axs[3].grid()
#axs[3].legend()

# Plot Chl-a
axs[4].plot(data.index, sat_chla_data, linestyle='-', color='blue')
axs[4].set_title('NDCI imputed over time')
axs[4].set_xlabel('Time')
axs[4].set_ylabel('NDCI')
axs[4].grid()

# Plot Air TC
axs[5].plot(data.index, AirTC_data, linestyle='-', color='blue')
axs[5].set_title('Air temperature over time')
axs[5].set_xlabel('Time')
axs[5].set_ylabel('Temperature [°C]')
axs[5].grid()

# Adjust layout
plt.tight_layout()
plt.show()


#%% CHLA REAL VS MODELS

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest: Measured vs Predicted
axes[0].scatter(y_test, y_test_pred_best, alpha=0.7, color='blue', label="Random Forest Predictions")
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label="1:1 Line")
axes[0].set_title("Random Forest Model")
axes[0].set_xlabel("Measured Chlorophyll-a")
axes[0].set_ylabel("Predicted Chlorophyll-a")
axes[0].legend()
axes[0].grid(True)

# XGBoost: Measured vs Predicted
axes[1].scatter(y_test, y_test_pred, alpha=0.7, color='green', label="XGBoost Predictions")
axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label="1:1 Line")
axes[1].set_title("XGBoost Model")
axes[1].set_xlabel("Measured Chlorophyll-a")
axes[1].set_ylabel("Predicted Chlorophyll-a")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

