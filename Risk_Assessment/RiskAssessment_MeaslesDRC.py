import geopandas as gpd
import pandas as pd
import os
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

#set wd
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/510/Anticipatory Action/RIPOSTE/Measles DRC/Data')

#set variables
# Set spatial resolution and set the common column to merge the data spatially
zone = "Zone"
province = "Province"

# Create master dataframe to store all the datasets
index_columns = ['start_date', 'end_date', zone, province]

# Set temporal resolution
temporal = "static"

#load admin boundaries
# Administrative level SHP file
admin_shp_path = "Els/RDC_Zones de santé.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
full_admin_boundaries.rename(columns={'Nom': zone, 'PROVINCE': province},inplace=True)  # Fix the naming of columns
admin_boundaries = full_admin_boundaries[[zone, province, 'geometry']]
# Make all admin names upper case to match other dataset inputs
admin_boundaries[zone] = admin_boundaries[zone].str.upper()
admin_boundaries[province] = admin_boundaries[province].str.upper()
print("Loaded admin boundaries")

#Load data
master_df = pd.read_csv('underreporting_complete_dataset_df_'+temporal+'.csv')

#load functions
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
    "Risk": "Risque"
}

################### Outlier removal of incidence data #####################
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
for dataset in [col for col in master_df.columns if col not in ['start_date', 'end_date', zone, province, 'Shape_Leng', 'Shape_Area', 'geometry']]:
  normalized_df = normalize_minmax(master_df, dataset)
# Inverse datasets that were not initially indexes and are not yet with the negative influence being the highest value
normalized_df['HCF/pop'] = (10 - normalized_df['HCF/pop'])
normalized_df.rename(columns={'HCF/pop': 'Lack of HCF'}, inplace=True)
print("Datasets normalized")

##################### Aggregation ######################### -> only needed on non-static data
# ## Combine all the incidence data for each admin level
# # Only find the mean of numeric columns
# numeric_columns = normalized_df.select_dtypes(include=['number']).columns.tolist()
columns_to_agg = ['Pop_Density', 'Poverty', 'Fraction_Vulnerable_Population', 'Fraction_displaced_population', 'Lack of HCF', 'Proportion_Unvaccinated_Children', 'Malnourished', 'No School', 'No Handwashing', 'Old Household Chief', 'Female Household Chief', 'No Access HCF']
# # Make the order of the regions fixed
# normalized_df[zone] = pd.Categorical(normalized_df[zone], categories=normalized_df[zone].unique(), ordered=True)
# # Find the mean per admin level
# time_aggregated_df = normalized_df.groupby([zone, province])[numeric_columns].mean()
# print("Aggregated to remove temporal resolution")

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
    print(f'{model_name} Testing R sq: {model.score(X_test, y_test)}')

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
weighted_indicators = [coef * normalized_df[col] for coef, col in zip(lr_model.coef_, columns_to_agg)]
# Combine the weighted indicators into a DataFrame
risk_scores_df = pd.DataFrame(weighted_indicators).transpose()
risk_scores_df = pd.concat([normalized_df[[zone, province]], risk_scores_df], axis=1)
print(risk_scores_df)

## Risk score per indicator
risk_scores_df.to_csv('weighted_risk_score_per_indicator_df_'+temporal+'.csv', index=True)
print("Created indicator risk scores")

## Aggregate by dimension
dimensions = {
    'Hazard and Exposure': ['Pop_Density'],
    'Vulnerability': ['Poverty', 'Fraction_Vulnerable_Population', 'Fraction_displaced_population', 'Malnourished', 'No School',  'Old Household Chief', 'Female Household Chief'],
    'Lack of Coping Capacity': ['Lack of HCF', 'Proportion_Unvaccinated_Children', 'No Handwashing', 'No Access HCF'],
    'Risk': ['Pop_Density', 'Poverty', 'Fraction_Vulnerable_Population', 'Fraction_displaced_population', 'Lack of HCF', 'Proportion_Unvaccinated_Children', 'Malnourished', 'No School', 'No Handwashing', 'Old Household Chief', 'Female Household Chief', 'No Access HCF']
}
# Loop through every dimension to calculate the mean of all the relevant columns
risk_aggregated_df = pd.DataFrame()
for dimension, columns_to_agg in dimensions.items():
    # Group by 'Category' and calculate the mean for all columns
    risk_aggregated_df[dimension] = risk_scores_df[columns_to_agg].mean(axis=1)
risk_aggregated_df = pd.concat([normalized_df[[zone, province]], risk_aggregated_df], axis=1)
risk_aggregated_df.to_csv('weighted_risk_score_df_'+temporal+'.csv', index=True)
print("Created dimension risk scores")

##################### Draw the weighted risk maps per dimension ########################
# Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
merged_gdf = admin_boundaries.merge(risk_aggregated_df, on=[zone, province])
merged_time_gdf = admin_boundaries.merge(normalized_df, on=[zone, province])
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
# Create a custom legend
legend_labels = [mpatches.Patch(color=color, label=map_legend_labels(label)) for color, label in zip(category_colors, category_labels)]
fig.legend(handles=legend_labels, title='Niveau de Risque', loc='upper right', bbox_to_anchor=(1, 1))
# Show the maps
plt.show()
