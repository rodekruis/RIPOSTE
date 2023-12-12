import geopandas as gpd
import pandas as pd
import os
import ee
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.mask import mask
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point
import seaborn as sns

########################## Definitions ########################
# # Log into Google Earth Engine
# ee.Authenticate()
# ee.Initialize()

# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')

# Set spatial resolution and set the common column to merge the data spatially
common_column = "ADM1_FR"

# Set temporal resolution
temporal = "monthly"

# Load admin boundaries
# Administrative level SHP file
admin_shp_path = "Administrative Boundaries\cmr_admbnda_inc_20180104_SHP\cmr_admbnda_adm1_inc_20180104.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
print("Loaded admin boundaries")

#Load data
master_df = pd.read_csv('complete_dataset_df_'+temporal+'.csv')
index_columns = ['start_date', 'end_date', common_column]
# Remove time periods that are missing incidence data
master_df_clean = master_df.dropna(subset=['Attack Rate'])

# Load functions
def normalize_minmax(df, column, min=None, max=None):
    normalized_df = df
    if min is None:
        min = df[column].min()
    if max is None:
        max = df[column].max()

    normalized_df[column] = ((df[column] - min) / (max - min)) * 10
    return normalized_df

def inform_class_thresholds(value, thresholds):
    if value <= thresholds[0]:
        return "Very low"
    for i in range(1, len(thresholds)):
        if thresholds[i - 1] < value <= thresholds[i]:
            return category_labels[i]
    return "Very high"

################### Depict data on graphs #####################
## Timeseries plots of indicators
# for dataset in [col for col in master_df.columns if col not in index_columns]:
#     print(dataset)
#     plt.figure(figsize=(8,6))
#     plt.title(f'Graph for {dataset}')
#     subset_df = master_df.set_index(index_columns)[dataset].unstack(common_column)
#     subset_df.plot(kind='line', ax=plt.gca())
#     plt.xlabel('Date Range')
#     plt.ylabel('Values')
#     plt.legend(title='Indicator data', loc='upper right')
#     plt.show()
# print("Plotted graphs")

# ## Plots of indicators vs incidence
# # Get the number of unique y datasets
# unique_datasets = [col for col in master_df_clean.columns if col not in ['start_date', 'end_date', common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
# # Calculate the number of rows and columns for subplots
# num_rows = len(unique_datasets) // 3 + (len(unique_datasets) % 3 > 0)
# num_cols = min(len(unique_datasets), 3)
# # Create base plot
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 12))
# # Loop through datasets
# for i, dataset in enumerate(unique_datasets):
#     row, col = divmod(i, num_cols)
#     ax = axs[row, col]
#     ax.scatter(master_df_clean['Attack Rate'], master_df_clean[dataset], marker='o', label='Data')
#     # Calculate the coefficients of the best-fit line
#     coefficients = np.polyfit(master_df_clean['Attack Rate'], master_df_clean[dataset], 1)
#     slope, intercept = coefficients
#     # Create the best-fit line equation
#     line_equation = f'Best-fit Line: y = {slope:.2f}x + {intercept:.2f}'
#     # Plot the best-fit line
#     ax.plot(master_df_clean['Attack Rate'], slope * master_df_clean['Attack Rate'] + intercept, color='red', linestyle='--',
#             label=line_equation)
#     ax.set_title(f'Scatter Plot for Attack Rate vs {dataset}')
#     ax.set_xlabel('Attack Rate')
#     ax.set_ylabel(dataset)
#     ax.legend(loc='upper left')
#     ax.grid(True)
# # Remove any empty subplots
# for i in range(len(unique_datasets), num_rows * num_cols):
#     fig.delaxes(axs.flatten()[i])
# plt.tight_layout()
# plt.show()

## Plots of indicators distribution
# Get the number of unique y datasets
unique_datasets = [col for col in master_df_clean.columns if col not in ['start_date', 'end_date', common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
# Calculate the number of rows and columns for subplots
num_rows = len(unique_datasets) // 3 + (len(unique_datasets) % 3 > 0)
num_cols = min(len(unique_datasets), 3)
# Create base plot
fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 12))
# Loop through datasets
for i, dataset in enumerate(unique_datasets):
    row, col = divmod(i, num_cols)
    ax = axs[row, col]
    sns.histplot(master_df_clean[dataset], kde=True, ax=ax)
    ax.set_title(f'Distribution plot for {dataset}')
    ax.set_xlabel(dataset)
    ax.set_ylabel("Frequency")
    ax.grid(True)
# Remove any empty subplots
for i in range(len(unique_datasets), num_rows * num_cols):
    fig.delaxes(axs.flatten()[i])
plt.tight_layout()
plt.show()

################### Outlier removal #####################
outliers_removed_count = 0
for col in ["cases"]:
  Q1 = master_df[col].quantile(0.25)
  Q3 = master_df[col].quantile(0.75)
  IQR = Q3 - Q1
  threshold = 3 * IQR
  # Create a boolean mask for identifying outliers
  outlier_mask = (master_df[col] < (Q1 - threshold)) | (master_df[col] > (Q3 + threshold))
  # Print the dataset name and the number of outliers removed
  outliers_removed = outlier_mask.sum()
  print(f'Dataset: {col}, Outliers Removed: {outliers_removed}')
  # Print the index values of the removed outliers
  removed_outlier_indices = master_df.index[outlier_mask].tolist()
  print(f'Removed Outlier Indices: {removed_outlier_indices}')

  # Remove outliers from the DataFrame
  master_df[col] = master_df[col][~outlier_mask]
  outliers_removed_count += outliers_removed
print(f"Total Outliers Removed: {outliers_removed_count}")
print("Outliers removed")

################### Normalize df of indicators ######################
# NB: Currently not setting fixed min and max values for any dataset, this might need to be changed. Also normalization might need to be done per region or time period.
normalized_df = master_df
for dataset in [col for col in master_df.columns if col not in ['start_date', 'end_date', common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]:
  normalized_df = normalize_minmax(master_df, dataset)
# Inverse datasets that were not initially indexes and are not yet with the negative influence being the highest value
normalized_df['FOSA/1000_hab.'] = (10 - normalized_df['FOSA/1000_hab.'])
normalized_df.rename(columns={'FOSA/1000_hab.': 'Lack_of_PH_Training'}, inplace=True)
print("Datasets normalized")

##################### Aggregation #########################
## Combine all the incidence data for each admin level
# Only find the mean of numeric columns
numeric_columns = normalized_df.select_dtypes(include=['number']).columns.tolist()
# Make the order of the regions fixed
normalized_df[common_column] = pd.Categorical(normalized_df[common_column], categories=normalized_df[common_column].unique(), ordered=True)
# Find the mean per admin level
time_aggregated_df = normalized_df.groupby(common_column)[numeric_columns].mean()
print(time_aggregated_df)
print("Aggregated to remove temporal resolution")

# #####################  INFORM - Hierarchical Model ################
# ## Combine the datasets into values per dimension
# # Define the dimensions
# dimensions = {
#     'Hazard and Exposure': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies'],
#     'Vulnerability': ['Poverty', 'Percentage_Vulnerable_Population', 'Conflicts', 'Avg_HH_size'],
#     'Lack of Coping Capacity': ['Pop_HCFs', 'Lack_of_PH_Training'],
#     'Risk': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Percentage_Vulnerable_Population', 'Conflicts', 'Avg_HH_size', 'Pop_HCFs', 'Lack_of_PH_Training']
# }
# # Create empty dataframe for the dimension means
# risk_aggregated_df = pd.DataFrame()
# # Loop through every dimension to calculate the mean of all the relevant columns
# for dimension, columns_to_agg in dimensions.items():
#     # Group by 'Category' and calculate the mean for all columns
#     dimension_mean = time_aggregated_df[columns_to_agg].mean(axis=1)
#     # Append the aggregated data to the result dataframe
#     risk_aggregated_df[dimension] = dimension_mean
# # Print the aggregated dataframe
# # Create CSV of risk scores per indicator
# risk_score_df = pd.concat([time_aggregated_df, risk_aggregated_df],axis=1)
# risk_score_df.to_csv('INFORM_risk_score_df_'+temporal+'.csv', index=True)
# print("Created risk scores")
#
# ##################### Draw the INFORM risk maps ########################
# # Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
# merged_gdf = admin_boundaries.merge(risk_aggregated_df, on=common_column)
# merged_time_gdf = admin_boundaries.merge(time_aggregated_df, on=common_column)
# # Define different thresholds for each dimension (customize these values)
# dimension_thresholds = {
#     "Hazard and Exposure": [1.4, 2.6, 4.0, 6.0, 10.0],
#     "Vulnerability": [1.9, 3.2, 4.7, 6.3, 10.0],
#     "Lack of Coping Capacity": [3.1, 4.6, 5.9, 7.3, 10.0],
#     "Risk": [1.9, 3.4, 4.9, 6.4, 10.0]
# }
# # Define the risk categories
# category_labels = ["Very low", "Low", "Medium", "High","Very high"]
# category_colors = [(1.0, 0.9607843137254902, 0.9411764705882353, 1.0),
#                    (1.0, 0.8784313725490196, 0.8235294117647058, 1.0),
#                    (1.0, 0.7098039215686275, 0.5490196078431373, 1.0),
#                    (0.9921568627450981, 0.4196078431372549, 0.23529411764705882, 1.0),
#                    (0.796078431372549, 0.09411764705882353, 0.11372549019607843, 1.0)]
#
# # Create a single figure with subplots for each dimension
# num_dimensions = len(risk_aggregated_df.columns)
# fig, axs = plt.subplots(1, num_dimensions, figsize=(16, 8))
# # Create map per dimension
# for i, dimension in enumerate(risk_aggregated_df.columns):
#     ax = axs[i]
#     colormap = plt.cm.colors.ListedColormap(category_colors)
#     thresholds = dimension_thresholds.get(dimension, [0])
#     merged_gdf[dimension + '_Category'] = merged_gdf[dimension].apply(lambda x: inform_class_thresholds(x, thresholds))
#     # Use the 'dimension_Category' values to map to colors using the colormap
#     norm = plt.Normalize(vmin=0, vmax=len(category_labels) - 1)
#     merged_gdf['Color'] = merged_gdf[dimension + '_Category'].apply(lambda x: norm(category_labels.index(x)))
#     # Plot using the 'Color' column to assign colors based on categories
#     merged_gdf.plot(column='Color', cmap=colormap, legend=False, ax=ax)
#     ax.set_title(dimension + ' Map')
#     ax.set_axis_off()
# # Label the map with region names
# for idx, row in merged_gdf.iterrows():
#     label = row[common_column]
#     for ax in axs:
#         ax.annotate(text=label, xy=row['geometry'].centroid.coords[0], horizontalalignment='center', fontsize=8, color='black')
# # Create a custom legend
# legend_labels = [mpatches.Patch(color=color, label=label) for color, label in zip(category_colors, category_labels)]
# fig.legend(handles=legend_labels, title='Risk Level', loc='upper right')
# # Show the maps
# plt.show()

################  Weighted Index ################
# Complete Pearson correlation to determine the coefficients that will be the weights in the index
columns_to_agg = ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Percentage_Vulnerable_Population', 'Conflicts', 'Avg_HH_size', 'Pop_HCFs', 'Lack_of_PH_Training']
# Group by the common column and calculate the mean, handling NaN values
time_aggregated_df = time_aggregated_df.groupby(common_column)[columns_to_agg + ['Attack Rate']].agg(np.nanmean)
# Drop rows with NaN values in any of the selected columns
clean_aggregated_df = time_aggregated_df.dropna(subset=columns_to_agg + ['Attack Rate'])
# Calculate correlation coefficients separately for each column
correlation_coefficients = [np.corrcoef(clean_aggregated_df[col], clean_aggregated_df['Attack Rate'], rowvar=False)[0, 1] for col in columns_to_agg]
print(correlation_coefficients)
# Multiply the correlation coefficients with the indicator values
weighted_indicators = [coef * time_aggregated_df[col] for coef, col in zip(correlation_coefficients, columns_to_agg)]
print(weighted_indicators)
# Combine the weighted indicators into a DataFrame
risk_scores_df = pd.DataFrame(weighted_indicators).transpose()
# Calculate the mean of risk scores per common column and add to dataframe with all the subindicators
risk_scores_df['Risk'] = risk_scores_df.mean(axis=1)
# Create new dataframe with only the mean risk scores column
risk_aggregated_df = pd.DataFrame(risk_scores_df.reset_index()[[common_column, 'Risk']])
print(risk_aggregated_df)
# Create CSV of risk scores per common column
risk_aggregated_df.to_csv('weighted_risk_score_df_'+temporal+'.csv', index=True)
print("Created risk scores")

##################### Draw the weighted risk maps ########################
# Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
merged_gdf = admin_boundaries.merge(risk_aggregated_df, on=common_column)
merged_time_gdf = admin_boundaries.merge(time_aggregated_df, on=common_column)
# Define the risk categories
category_labels = ["Very low", "Low", "Medium", "High","Very high"]
category_colors = [(1.0, 0.9607843137254902, 0.9411764705882353, 1.0),
                   (1.0, 0.8784313725490196, 0.8235294117647058, 1.0),
                   (1.0, 0.7098039215686275, 0.5490196078431373, 1.0),
                   (0.9921568627450981, 0.4196078431372549, 0.23529411764705882, 1.0),
                   (0.796078431372549, 0.09411764705882353, 0.11372549019607843, 1.0)]
# Create a figure for the map
fig, ax = plt.subplots(figsize=(12, 12))
print(merged_gdf['Risk'])
print(merged_gdf['Risk'].isnull().sum())
print(len(merged_gdf['Risk']))
print(merged_gdf['Risk'].dtype)
#Divide data into 5 quantiles with equal number of data points
merged_gdf['Risk_Category'] = pd.qcut(merged_gdf['Risk'], q=5, labels=category_labels)
# Plot using the 'Color' column to assign colors based on categories
colormap = plt.cm.colors.ListedColormap(category_colors)
merged_gdf.plot(column='Risk_Category', cmap=colormap, legend=True, ax=ax)
# Set axis off for cleaner map
ax.set_axis_off()
# Label the map with region names
for idx, row in merged_gdf.iterrows():
    label = row[common_column]
    ax.annotate(text=label, xy=row['geometry'].centroid.coords[0], horizontalalignment='center', fontsize=8, color='black')
# Create a custom legend
legend_labels = [mpatches.Patch(color=color, label=label) for color, label in zip(category_colors, category_labels)]
ax.legend(handles=legend_labels, title='Risk Level', loc='upper right', bbox_to_anchor=(1.2, 1))
# Show the map
plt.show()

############# Plot risk index accuracy - risk score vs incidence rate ##################
# Create scatter plot
for region in master_df[common_column].unique():
    region_incidence = merged_time_gdf[merged_time_gdf[common_column] == region]['Attack Rate']
    region_risk = merged_gdf[merged_gdf[common_column] == region]['Risk']
    plt.scatter(region_risk, region_incidence, label=region, s=50)
plt.xlabel('Risk Score')
plt.ylabel('Cholera Attack Rate')
plt.title('Correlation between risk level and cholera incidence')
plt.legend()
plt.show()

# Create a box and whisker plot
boxplot_incidence = []
boxplot_risk_category = []
for region in master_df[common_column].unique():
    region_incidence = merged_time_gdf[merged_time_gdf[common_column] == region]['Attack Rate']
    boxplot_incidence.extend(region_incidence.tolist())
    region_risk_category = merged_gdf[merged_gdf[common_column] == region]['Risk_Category']
    boxplot_risk_category.extend(region_risk_category.tolist())
# Create a DataFrame from the lists
data = pd.DataFrame({
    'Risk_Category': boxplot_risk_category,
    'Attack Rate': boxplot_incidence
})
# Specify the order of risk categories
category_order = ["Very low", "Low", "Medium", "High", "Very high"]
# Create a box and whisker plot with specified order
plt.figure(figsize=(12, 8))
sns.boxplot(x='Risk_Category', y='Attack Rate', data=data, order=category_order)
plt.xlabel('Risk Level')
plt.ylabel('Cholera Attack Rate')
plt.title('Box and Whisker Plot of Cholera Attack Rate by Risk Level')
plt.show()

############ Model Cross-Validation #########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Assuming your data is in a DataFrame named 'master_df' with columns including 'Attack Rate', 'Pop_Density', 'Poverty', 'Percentage_Vulnerable_Population', 'Percentage_displaced_population', 'Lack of HCF'

# Define features and target variable
features = ['Pop_Density', 'Poverty', 'Percentage_Vulnerable_Population', 'Percentage_displaced_population', 'Lack of HCF']
target = 'Attack Rate'
master_df.dropna(inplace=True)

# Extract features and target variable
X = master_df[features]
y = master_df[target]

# Define the linear regression model
model = LinearRegression()

# Perform k-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean RMSE:", np.mean(rmse_scores))

# Sensitivity Analysis
feature_importance = {}
base_rmse = np.mean(rmse_scores)

for feature in features:
    # Perturb the feature values
    perturbed_X = X.copy()
    perturbed_X[feature] += np.random.normal(0, 0.01, len(X))

    # Evaluate the model with perturbed feature
    perturbed_cv_scores = cross_val_score(model, perturbed_X, y, cv=5, scoring='neg_mean_squared_error')
    perturbed_rmse = np.sqrt(-perturbed_cv_scores)

    # Calculate sensitivity
    sensitivity = np.mean(perturbed_rmse) - base_rmse
    feature_importance[feature] = sensitivity

# Display feature importance
print("Sensitivity Analysis:")
for feature, sensitivity in feature_importance.items():
    print(f"{feature}: {sensitivity}")

# Plot sensitivity analysis results
plt.bar(feature_importance.keys(), feature_importance.values())
plt.xlabel('Features')
plt.ylabel('Sensitivity (Change in RMSE)')
plt.title('Sensitivity Analysis')
plt.show()