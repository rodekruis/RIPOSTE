import geopandas as gpd
import pandas as pd
import os
import ee
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import rasterio
from rasterio.mask import mask
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.spatial.distance import pdist
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

########################## Definitions ########################

# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/510/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')

# Set spatial resolution and set the common column to merge the data spatially
common_column = "DISTRICT_S"

# Set temporal resolution
temporal = "monthly"

# Load admin boundaries
# Administrative level SHP file
admin_shp_path = "Administrative Boundaries\Health Boundaries\District_sante_2022.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column,"NOM_REGION", 'geometry']]
print("Loaded admin boundaries")

#Load data
master_df = pd.read_csv('complete_dataset_df_'+temporal+'_districts.csv')
index_columns = ['start_date', 'end_date', 'NOM_REGION', common_column]

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
for dataset in [col for col in master_df.columns if col not in ['start_date', 'end_date', common_column, 'NOM_REGION', 'Shape_Leng', 'Shape_Area', 'geometry']]:normalized_df = normalize_minmax(master_df, dataset)
# Inverse datasets that were not initially indexes and are not yet with the negative influence being the highest value
normalized_df['FOSA/1000_hab.'] = (10 - normalized_df['FOSA/1000_hab.'])
normalized_df.rename(columns={'FOSA/1000_hab.': 'Lack_of_PH_Facilities'}, inplace=True)
# Create a new column for the U-shaped Min-Max normalized values
normalized_df['Connected_Zones'] = 0
normalized_df['Poor_Access'] = 0
# Mask for values greater than 5
connected_zones = normalized_df['Roads'] > normalized_df['Roads'].mean()
# Perform inverted Min-Max normalization on values greater than 5 and store in 'Roads_U'
normalized_df.loc[connected_zones, 'Connected_Zones'] = ((normalized_df.loc[connected_zones, 'Roads'] - normalized_df.loc[connected_zones, 'Roads'].min()) / (normalized_df.loc[connected_zones, 'Roads'].max() - normalized_df.loc[connected_zones, 'Roads'].min()))*10
# Mask for values less than or equal to 5
poor_access = normalized_df['Roads'] <= normalized_df['Roads'].mean()
# Perform inverted Min-Max normalization on values less than or equal to 5 and store in 'Roads_U'
normalized_df.loc[poor_access, 'Poor_Access'] = 10 - (((normalized_df.loc[poor_access, 'Roads'] - normalized_df.loc[poor_access, 'Roads'].min()) /(normalized_df.loc[poor_access, 'Roads'].max() - normalized_df.loc[poor_access, 'Roads'].min())))*10
print("Datasets normalized")
normalized_df.to_csv('nomalized_df_'+temporal+'_districts.csv', index=False, float_format='%.8f')

##################### Aggregation #########################
## Combine all the incidence data for each admin level
# Only find the mean of numeric columns
numeric_columns = normalized_df.select_dtypes(include=['number']).columns.tolist() #includes Attck Rate, cases and deaths
columns_to_agg = ['total_precipitation', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Conflicts', 'Avg_HH_size', 'Lack_of_PH_Facilities', 'Prison', 'Connected_Zones', 'Border', 'PAMI']
# Make the order of the regions fixed
normalized_df[common_column] = pd.Categorical(normalized_df[common_column], categories=normalized_df[common_column].unique(), ordered=True)
# # Find the mean per admin level
# time_aggregated_df = normalized_df.groupby(common_column)[numeric_columns].mean()
# Group by the common column and calculate the mean per region, handling NaN values
time_aggregated_df = normalized_df.groupby(common_column)[numeric_columns].agg(np.nanmean)
print("Aggregated to remove temporal resolution")

#####################  INFORM - Hierarchical Model ################
## Combine the datasets into values per dimension
# Define the dimensions
dimensions = {
    'Hazard and Exposure': ['total_precipitation', 'Hazards', 'Pop_Density', 'Water_Bodies', 'Prison', 'Connected_Zones', 'Border'],
    'Vulnerability': ['Poverty', 'Conflicts', 'Avg_HH_size'],
    'Lack of Coping Capacity': ['Lack_of_PH_Facilities', 'Poor_Access', 'Insufficient_WASH'],
    'Risk': ['total_precipitation', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Conflicts', 'Avg_HH_size', 'Lack_of_PH_Facilities', 'Prison', 'Border', 'Connected_Zones', 'Poor_Access']
}
# Create empty dataframe for the dimension means
risk_aggregated_df = pd.DataFrame()
# Loop through every dimension to calculate the mean of all the relevant columns
for dimension, columns_to_agg in dimensions.items():
    # Group by 'Category' and calculate the mean for all columns
    dimension_mean = time_aggregated_df[columns_to_agg].mean(axis=1)
    # Append the aggregated data to the result dataframe
    risk_aggregated_df[dimension] = dimension_mean
# Print the aggregated dataframe
# Create CSV of risk scores per indicator
risk_score_df = pd.concat([time_aggregated_df, risk_aggregated_df],axis=1)
risk_score_df.to_excel('INFORM_risk_score_df_' + temporal + '.xlsx', index=True)
print("Created risk scores")
#
##################### Draw the INFORM risk maps ########################
# Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
merged_gdf = admin_boundaries.merge(risk_score_df, on=common_column)
merged_time_gdf = admin_boundaries.merge(time_aggregated_df, on=common_column)
# Define different thresholds for each dimension (customize these values)
dimension_thresholds = {
    "Hazard and Exposure": [1.4, 2.6, 4.0, 6.0, 10.0],
    "Vulnerability": [1.9, 3.2, 4.7, 6.3, 10.0],
    "Lack of Coping Capacity": [3.1, 4.6, 5.9, 7.3, 10.0],
    "Risk": [1.9, 3.4, 4.9, 6.4, 10.0]
}
# Define the risk categories
category_labels = ["Very low", "Low", "Medium", "High","Very high"]
category_colors = [(1.0, 0.9607843137254902, 0.9411764705882353, 1.0),
                   (1.0, 0.8784313725490196, 0.8235294117647058, 1.0),
                   (1.0, 0.7098039215686275, 0.5490196078431373, 1.0),
                   (0.9921568627450981, 0.4196078431372549, 0.23529411764705882, 1.0),
                   (0.796078431372549, 0.09411764705882353, 0.11372549019607843, 1.0)]

# French translation for the dimensions
dimension_translation = {
    "Hazard and Exposure": "Aléa et Exposition",
    "Vulnerability": "Vulnérabilité",
    "Lack of Coping Capacity": "Manque de Capacité d’Adaptation",
    "Risk": "Risque"
}

# 1. Create a plot with the three dimensions
fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # Adjusted size for better fitting

# Loop through the first three dimensions
for i, (dimension, thresholds) in enumerate(dimension_thresholds.items()):
    if dimension == "Risk":
        continue  # Skip the Risk dimension for this plot

    ax = axs[i]
    # Apply thresholds and classify the current dimension into categories
    merged_gdf[dimension + '_Category'] = merged_gdf[dimension].apply(lambda x: inform_class_thresholds(x, thresholds))

    # Get the unique categories present in the data
    present_categories = merged_gdf[dimension + '_Category'].unique()
    present_category_labels = [label for label in category_labels if label in present_categories]
    present_category_colors = [category_colors[category_labels.index(label)] for label in present_category_labels]

    # Normalize based only on the present categories
    norm = plt.Normalize(vmin=0, vmax=len(present_category_labels) - 1)
    merged_gdf['Color'] = merged_gdf[dimension + '_Category'].apply(lambda x: norm(present_category_labels.index(x)))

    # Create a new colormap based on present categories
    colormap = plt.cm.colors.ListedColormap(present_category_colors)

    # Plot using the 'Color' column to assign colors based on categories
    merged_gdf.plot(column='Color', cmap=colormap, legend=False, ax=ax)

    # Set title and remove axis
    ax.set_title(dimension_translation[dimension], fontsize=12)  # Title in French
    ax.set_axis_off()

# Create a custom legend for the first plot
legend_labels = [mpatches.Patch(color=color, label=map_legend_labels(label)) for color, label in zip(category_colors, category_labels)]
fig.legend(handles=legend_labels, title='Niveau de risque', loc='upper right')

# Show the plot with three dimensions
plt.tight_layout()
plt.show()

# 2. Create a plot for the Risk dimension
fig, ax = plt.subplots(figsize=(10, 8))  # Create a single figure for the Risk map

# Apply thresholds and classify the Risk dimension into categories
merged_gdf['Risk_Category'] = merged_gdf['Risk'].apply(lambda x: inform_class_thresholds(x, dimension_thresholds['Risk']))

# Get the unique categories present in the Risk dimension data
present_categories = merged_gdf['Risk_Category'].unique()
present_category_labels = [label for label in category_labels if label in present_categories]
present_category_colors = [category_colors[category_labels.index(label)] for label in present_category_labels]

# Normalize based only on the present categories
norm = plt.Normalize(vmin=0, vmax=len(present_category_labels) - 1)
merged_gdf['Color'] = merged_gdf['Risk_Category'].apply(lambda x: norm(present_category_labels.index(x)))

# Create a new colormap based on present categories
colormap = plt.cm.colors.ListedColormap(present_category_colors)

# Plot using the 'Color' column to assign colors based on categories
merged_gdf.plot(column='Color', cmap=colormap, legend=False, ax=ax)

# Set title and remove axis
ax.set_title('Cartes de zones à risque', fontsize=15)  # Title in French
ax.set_axis_off()

# Create a custom legend for the Risk plot
legend_labels = [mpatches.Patch(color=color, label=map_legend_labels(label)) for color, label in zip(present_category_colors, present_category_labels)]
fig.legend(handles=legend_labels, title='Niveau de risque', loc='upper right')

# Show the Risk map plot
plt.tight_layout()
plt.show()

# 3. Prepare Data for Excel Export
# Remove 'Color' column
final_df = merged_gdf.drop(columns=['Color'])

# Optionally, save the result to an Excel file
final_df.to_excel('INFORM_risk_score_category_df.xlsx', index=True)

# Print the final DataFrame
print(final_df)


# Create scatter plot - INFORM method
for region in master_df[common_column].unique():
    region_incidence = merged_time_gdf[merged_time_gdf[common_column] == region]['Attack Rate']
    region_risk = merged_gdf[merged_gdf[common_column] == region]['Risk']
    plt.scatter(region_risk, region_incidence, label=region, s=50)

plt.xlabel('Score de Risque')
plt.ylabel(f"Taux d'Attaque de Choléra")
plt.title(f"Corrélation entre le score de risque INFORM et le taux d'attaque de choléra")
plt.show()

# ################  Weighted Index ################
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
        # 4. Create a distance matrix using feature importances (Euclidean distance)
        distance_matrix = pdist(model_importance.reshape(-1, 1), metric='euclidean')

        # 5. Perform hierarchical clustering using 'linkage' method
        linkage_matrix = linkage(distance_matrix, method='ward')

        # 6. Plot the dendrogram to visualize feature similarity
        feature_names = X.columns.tolist()
        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix, labels=feature_names, leaf_rotation=90)
        plt.title("Feature Importance Dendrogram (Random Forest)")
        plt.xlabel("Feature")
        plt.ylabel("Euclidean Distance")
        plt.show()
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
plt.title('Importance des variables moyenne pour tous les modèles')
plt.xlabel('Importance relative')
plt.show()

### Create risk scores with model resulting coefficients
lr_model = LinearRegression().fit(X_train, y_train)
print("Regression Coefficients:", lr_model.coef_)
# Multiply the correlation coefficients with the indicator values
weighted_indicators = [coef * time_aggregated_df[col] for coef, col in zip(lr_model.coef_, columns_to_agg)]
# Combine the weighted indicators into a DataFrame
weighted_risk_scores_df = pd.DataFrame(weighted_indicators).transpose()
# Define the dimensions
dimensions = {
    'Hazard and Exposure': ['total_precipitation', 'Hazards', 'Pop_Density', 'Water_Bodies', 'Prison', 'Connected_Zones', 'Border'],
    'Vulnerability': ['Poverty', 'Conflicts', 'Avg_HH_size'],
    'Lack of Coping Capacity': ['Lack_of_PH_Facilities', 'Poor_Access', 'Insufficient_WASH'],
    'Risk': ['total_precipitation', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Conflicts', 'Avg_HH_size', 'Lack_of_PH_Facilities', 'Prison', 'Border', 'Connected_Zones', 'Poor_Access']
}
# Loop through every dimension to calculate the mean of all the relevant columns
risk_aggregated_df = pd.DataFrame()
for dimension, columns_to_agg in dimensions.items():
    # Group by 'Category' and calculate the mean for all columns
    risk_aggregated_df[dimension] = weighted_risk_scores_df[columns_to_agg].mean(axis=1)
risk_aggregated_df.to_csv('weighted_risk_score_df_'+temporal+'.csv', index=True)
print("Created risk scores")

##################### Draw the weighted risk maps ########################
# Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
weighted_merged_gdf = admin_boundaries.merge(risk_aggregated_df, on=common_column)
weighted_merged_time_gdf = admin_boundaries.merge(time_aggregated_df, on=common_column)
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
    weighted_merged_gdf[dimension + '_Category'] = pd.qcut(weighted_merged_gdf[dimension], q=5, labels=category_labels)
    # Use the 'dimension_Category' values to map to colors using the colormap
    norm = plt.Normalize(vmin=0, vmax=len(category_labels) - 1)
    weighted_merged_gdf['Color'] = weighted_merged_gdf[dimension + '_Category'].apply(lambda x: norm(category_labels.index(x)))
    # Plot using the 'Color' column to assign colors based on categories
    weighted_merged_gdf.plot(column='Color', cmap=colormap, legend=False, ax=ax)
    ax.set_title(risk_dimensions(dimension))
    ax.set_axis_off()
printable_df = weighted_merged_gdf[['NOM_REGION', common_column, 'Risk_Category']]
printable_df.to_csv('weighted_risk_score_df_' + temporal + '.csv', index=True)
# # Label the map with region names
# for idx, row in merged_gdf.iterrows():
#     label = row[common_column]
#     for ax in axs:
#         ax.annotate(text=label, xy=row['geometry'].centroid.coords[0], horizontalalignment='center', fontsize=8, color='black')
# Create a custom legend
legend_labels = [mpatches.Patch(color=color, label=map_legend_labels(label)) for color, label in zip(category_colors, category_labels)]
fig.legend(handles=legend_labels, title='Niveau de Risque', loc='upper right', bbox_to_anchor=(1, 1))
# Show the maps
plt.show()

######## Draw regression coefficients plot #######
# Create a DataFrame with the coefficients and corresponding indicators
coefficients_df = pd.DataFrame({
    'Indicator': X_train.columns,
    'Coefficient': lr_model.coef_
})
# Sort the coefficients in descending order for better visualization
coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)
# Plot the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Indicator', data=coefficients_df, palette="coolwarm")
# Add plot title and labels
plt.title('Coefficients de Régression Linéaire par Indicateur')
plt.xlabel('Coefficient')
plt.ylabel('Indicateur')
# Display the plot
plt.tight_layout()
plt.show()

############# Plot risk index accuracy - risk score vs incidence rate ##################
# Create scatter plot - weighted method
for region in master_df[common_column].unique():
    region_incidence = weighted_merged_time_gdf[weighted_merged_time_gdf[common_column] == region]['Attack Rate']
    region_risk = weighted_merged_gdf[weighted_merged_gdf[common_column] == region]['Risk']
    plt.scatter(region_risk, region_incidence, label=region, s=50)
plt.xlabel('Score de Risque')
plt.ylabel(f"Taux d'Attaque de Choléra")
plt.title(f"Corrélation entre le score de risque pondéré et le taux d'attaque de choléra")
plt.show()

# Create a box and whisker plot
boxplot_incidence = []
boxplot_risk_category = []
for region in master_df[common_column].unique():
    region_incidence = weighted_merged_time_gdf[weighted_merged_time_gdf[common_column] == region]['Attack Rate']
    boxplot_incidence.extend(region_incidence.tolist())
    region_risk_category = weighted_merged_gdf[weighted_merged_gdf[common_column] == region]['Risk_Category']
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
print("Showed risk map accuracy")
