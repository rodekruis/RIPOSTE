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
import statsmodels.api as sm
import shap
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

########################## Definitions ########################

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
master_df = pd.read_csv('underreporting_complete_dataset_df_'+temporal+'.csv')
index_columns = ['start_date', 'end_date', common_column]

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

# Function to map legend labels
def map_legend_labels(label):
    return legend_label_mapping.get(label, label)
# English to French legend label mapping
legend_label_mapping = {
    "Very low": "Très faible",
    "Low": "Faible",
    "Medium": "Moyen",
    "High": "Élevé",
    "Very high": "Très élevé"
}

def risk_dimensions(dimension):
    return risk_dimensions_mapping.get(dimension, dimension)
# English to French dimension mapping
risk_dimensions_mapping = {
    "Hazard and Exposure": "Aléa et Exposition",
    "Vulnerability": "Vulnérabilité",
    "Lack of Coping Capacity": "Manque de Capacité d’Adaptation",
}

################### Depict data on graphs #####################
# # Timeseries plots of indicators
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

## Incidence Graph for French Presentation
plt.figure(figsize=(8,6))
plt.title(f'Incidence de Choléra')
subset_df = master_df.set_index(index_columns)['Attack Rate'].unstack(common_column)
subset_df.plot(kind='line', ax=plt.gca())
plt.xlabel('Dates')
plt.ylabel('Taux d\'Attaque')
plt.legend(title='Régions', loc='upper right')
plt.show()

# ## Incidence, Precipitation, Temperature Graph for French Presentation
# plt.figure(figsize=(8,6))
# plt.title(f'Incidence de Choléra')
# subset_df = master_df.set_index(index_columns)['Attack Rate'].unstack(common_column)
# subset_df.plot(kind='line', ax=plt.gca(), label='Taux d\'Attaque', linestyle='-', marker='o')
# ax2 = plt.gca().twinx()
# subset_df2 = master_df.set_index(index_columns)['total_precipitation_sum'].unstack(common_column)
# subset_df2.plot(kind='line', ax=ax2, color='orange', label='Précipitation', linestyle='--', marker='^')  # Line2 is the label for the second line
# ax3 = plt.gca().twinx()
# ax3.spines['right'].set_position(('outward', 60))  # Adjust the position of the third y-axis
# subset_df3 = master_df.set_index(index_columns)['skin_temperature'].unstack(common_column)
# subset_df3.plot(kind='line', ax=ax3, color='green', label='Température', linestyle=':', marker='s')  # Line3 is the label for the third line
# plt.xlabel('Dates')
# plt.ylabel('Taux d\'Attaque')
# plt.legend(title='Régions', loc='upper right')
# plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.title(f'Incidence, Précipitation, Température')

# Get unique colors for each region
region_colors = plt.cm.get_cmap('tab10', len(master_df[common_column].unique()))

# Plot Incidence (Taux d'Attaque)
subset_df = master_df.set_index(index_columns)['Attack Rate'].unstack(common_column)
for i, region in enumerate(subset_df.columns):
    subset_df[region].plot(kind='line', ax=plt.gca(), label=f'Taux d\'Attaque - {region}', linestyle='-', marker='o', color=region_colors(i))

# Plot Précipitation
ax2 = plt.gca().twinx()
subset_df2 = master_df.set_index(index_columns)['total_precipitation_sum'].unstack(common_column)
for i, region in enumerate(subset_df2.columns):
    subset_df2[region].plot(kind='line', ax=ax2, label=f'Précipitation - {region}', linestyle='--', marker='^', color=region_colors(i))

# Plot Température
ax3 = plt.gca().twinx()
ax3.spines['right'].set_position(('outward', 60))
subset_df3 = master_df.set_index(index_columns)['skin_temperature'].unstack(common_column)
for i, region in enumerate(subset_df3.columns):
    subset_df3[region].plot(kind='line', ax=ax3, label=f'Température - {region}', linestyle=':', marker='s', color=region_colors(i))

# Set labels and legend
plt.xlabel('Dates')
plt.ylabel('Taux d\'Attaque / Précipitation / Température')
plt.legend(title='Régions', loc='upper right')

# Display the plot
plt.show()





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

# ## Plots of indicators distribution
# # Get the number of unique y datasets
# unique_datasets = [col for col in master_df.columns if col not in ['start_date', 'end_date', common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
# # Calculate the number of rows and columns for subplots
# num_rows = len(unique_datasets) // 3 + (len(unique_datasets) % 3 > 0)
# num_cols = min(len(unique_datasets), 3)
# # Create base plot
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 12))
# # Loop through datasets
# for i, dataset in enumerate(unique_datasets):
#     row, col = divmod(i, num_cols)
#     ax = axs[row, col]
#     sns.histplot(master_df[dataset], kde=True, ax=ax)
#     ax.set_title(f'Distribution plot for {dataset}')
#     ax.set_xlabel(dataset)
#     ax.set_ylabel("Frequency")
#     ax.grid(True)
# # Remove any empty subplots
# for i in range(len(unique_datasets), num_rows * num_cols):
#     fig.delaxes(axs.flatten()[i])
# plt.tight_layout()
# plt.show()

################### Outlier removal #####################
outliers_removed_count = 0
for col in ["Attack Rate"]:
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
normalized_df.rename(columns={'FOSA/1000_hab.': 'Lack_of_PH_Facilities'}, inplace=True)
print("Datasets normalized")

##################### Aggregation #########################
## Combine all the incidence data for each admin level
# Only find the mean of numeric columns
numeric_columns = normalized_df.select_dtypes(include=['number']).columns.tolist() #includes Attck Rate, cases and deaths
columns_to_agg = ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Conflicts', 'Avg_HH_size', 'Lack_of_PH_Facilities']
# Make the order of the regions fixed
normalized_df[common_column] = pd.Categorical(normalized_df[common_column], categories=normalized_df[common_column].unique(), ordered=True)
# # Find the mean per admin level
# time_aggregated_df = normalized_df.groupby(common_column)[numeric_columns].mean()
# Group by the common column and calculate the mean per region, handling NaN values
time_aggregated_df = normalized_df.groupby(common_column)[numeric_columns].agg(np.nanmean)
print("Aggregated to remove temporal resolution")

# #####################  INFORM - Hierarchical Model ################
# ## Combine the datasets into values per dimension
# # Define the dimensions
# dimensions = {
#     'Hazard and Exposure': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies'],
#     'Vulnerability': ['Poverty', 'Fraction_Vulnerable_Population', 'Conflicts', 'Avg_HH_size'],
#     'Lack of Coping Capacity': ['Pop_HCFs', 'Lack_of_PH_Training'],
#     'Risk': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Fraction_Vulnerable_Population', 'Conflicts', 'Avg_HH_size', 'Pop_HCFs', 'Lack_of_PH_Training']
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
# Drop rows with NaN values in any of the selected columns from the full, not time aggregated, dataset
clean_aggregated_df = normalized_df.dropna(subset=columns_to_agg + ['Attack Rate'])
### Complete regression analysis
X = clean_aggregated_df[columns_to_agg]  # Independent variables
y = clean_aggregated_df['Attack Rate']  # Dependent variable
# Display the number of data points
print(f'Number of data points for linear regression: {len(X)}')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define a list of regression models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
}
# Dictionary to store model instances
model_instances = {}
# Calculate average importance across all models for each feature
average_importance = pd.DataFrame(index=X.columns)
# Iterate over each model and evaluate its performance
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Store the trained model instance for later use
    model_instances[model_name] = model
    # Display R-squared values
    print(f'{model_name} Training R sq: {model.score(X_train, y_train)}')
    print(f'{model_name} Training R sq: {model.score(X_test, y_test)}')

    # Display feature importance
    if model_name == 'Random Forest':
        model_importance = model.feature_importances_
    else:
        model_importance = model.coef_
    feature_importance = pd.Series(model_importance, index=X.columns).abs()
    feature_importance.sort_values(ascending=False, inplace=True)
    average_importance[model_name] = feature_importance / feature_importance.sum()
    # # Plot feature importance
    # plt.figure(figsize=(8, 6))
    # sns.barplot(x=feature_importance, y=feature_importance.index, palette="dark:lightblue", hue=feature_importance.index, legend=False)
    # plt.title(f'{model_name} Feature Importances')
    # plt.xlabel('Relative Importance')
    # plt.show()
# Calculate average importance
average_importance['Average Importance'] = average_importance.mean(axis=1)
average_importance.sort_values(by='Average Importance', ascending=False, inplace=True)
# Plot average feature importance across all models
plt.figure(figsize=(10, 8))
sns.barplot(x='Average Importance', y=average_importance.index, data=average_importance, palette="dark:lightblue", hue=feature_importance.index, legend=False)
plt.title('Average Feature Importance Across All Models')
plt.xlabel('Relative Importance')
plt.show()

### Create risk scores with model resulting coefficients
lr_model = LinearRegression().fit(X_train, y_train)
print("Regression Coefficients:", lr_model.coef_)
# Multiply the correlation coefficients with the indicator values
weighted_indicators = [coef * time_aggregated_df[col] for coef, col in zip(lr_model.coef_, columns_to_agg)]
# Combine the weighted indicators into a DataFrame
risk_scores_df = pd.DataFrame(weighted_indicators).transpose()
# Define the dimensions
dimensions = {
    'Hazard and Exposure': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies'],
    'Vulnerability': ['Poverty', 'Conflicts', 'Avg_HH_size'],
    'Lack of Coping Capacity': ['Lack_of_PH_Facilities'],
    'Risk': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Conflicts', 'Avg_HH_size', 'Lack_of_PH_Facilities']
}
# Loop through every dimension to calculate the mean of all the relevant columns
risk_aggregated_df = pd.DataFrame()
for dimension, columns_to_agg in dimensions.items():
    # Group by 'Category' and calculate the mean for all columns
    risk_aggregated_df[dimension] = risk_scores_df[columns_to_agg].mean(axis=1)
risk_aggregated_df.to_csv('weighted_risk_score_df_'+temporal+'.csv', index=True)
print("Created risk scores")

##################### Draw the weighted risk maps ########################
# Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
merged_gdf = admin_boundaries.merge(risk_aggregated_df, on=common_column)
merged_time_gdf = admin_boundaries.merge(time_aggregated_df, on=common_column)
# Define the risk categories
category_labels = ["Very low", "Low", "Medium", "High", "Very high"]
category_colors = [(1.0, 0.9607843137254902, 0.9411764705882353, 1.0),
                   (1.0, 0.8784313725490196, 0.8235294117647058, 1.0),
                   (1.0, 0.7098039215686275, 0.5490196078431373, 1.0),
                   (0.9921568627450981, 0.4196078431372549, 0.23529411764705882, 1.0),
                   (0.796078431372549, 0.09411764705882353, 0.11372549019607843, 1.0)]
# Create a single figure with subplots for each dimension
num_dimensions = 4
fig, axs = plt.subplots(1, num_dimensions, figsize=(16, 8))
# Create map per dimension
for i, dimension in enumerate(["Hazard and Exposure", "Vulnerability", "Lack of Coping Capacity", "Risk"]):
    ax = axs[i]
    colormap = plt.cm.colors.ListedColormap(category_colors)
    # Divide data into 5 quantiles with equal number of data points
    merged_gdf[dimension + '_Category'] = pd.qcut(merged_gdf[dimension], q=5, labels=category_labels)
    # Use the 'dimension_Category' values to map to colors using the colormap
    norm = plt.Normalize(vmin=0, vmax=len(category_labels) - 1)
    merged_gdf['Color'] = merged_gdf[dimension + '_Category'].apply(lambda x: norm(category_labels.index(x)))
    # Plot using the 'Color' column to assign colors based on categories
    merged_gdf.plot(column='Color', cmap=colormap, legend=False, ax=ax)
    ax.set_title(risk_dimensions(dimension))
    ax.set_axis_off()
# Label the map with region names
for idx, row in merged_gdf.iterrows():
    label = row[common_column]
    for ax in axs:
        ax.annotate(text=label, xy=row['geometry'].centroid.coords[0], horizontalalignment='center', fontsize=8, color='black')
# Create a custom legend
legend_labels = [mpatches.Patch(color=color, label=map_legend_labels(label)) for color, label in zip(category_colors, category_labels)]
fig.legend(handles=legend_labels, title='Niveau de Risque', loc='upper right', bbox_to_anchor=(1, 1))
# Show the maps
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
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Risk_Category', y='Attack Rate', data=data, order=category_order)
# plt.xlabel('Risk Level')
# plt.ylabel('Cholera Attack Rate')
# plt.title('Box and Whisker Plot of Cholera Attack Rate by Risk Level')
# plt.show()
# print("Showed risk map accuracy")


