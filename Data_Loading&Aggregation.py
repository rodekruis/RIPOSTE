import geopandas as gpd
import pandas as pd
import os
import ee
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.mask import mask
import numpy as np
from functools import reduce
from shapely.geometry import Point
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans


########################## Definitions ########################
# Log into Google Earth Engine
#ee.Authenticate()
#ee.Initialize()

# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')

# Administrative level SHP file
admin_shp_path = "Administrative Boundaries\cmr_admbnda_inc_20180104_SHP\cmr_admbnda_adm1_inc_20180104.shp"
admin_boundaries = gpd.read_file(admin_shp_path)
print("Loaded admin boundaries")

# Use the common column to merge the data spatially
common_column = "ADM1_FR"

# Create master dataframe to store all the datasets
index_columns = [common_column, 'start_date', 'end_date']
master_df = pd.DataFrame(columns=[common_column, 'start_date', 'end_date'])


### Functions
def add_dataframe_to_master(dataframe):
    # Add all the individual dataframes to the master dataframe
    global master_df
    master_df = pd.merge(master_df, dataframe, on=index_columns, how='outer')
def gee_extraction(imageset, band):
    image_collection = ee.ImageCollection(imageset)
    # Convert the shapefile to a GeoJSON FeatureCollection
    fc = ee.FeatureCollection(admin_boundaries.__geo_interface__)
    # Create an empyt list for all the results to be appended to inside the for loop
    results = []
    # Loop through spatial resolution dates
    for index, row in time_periods.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']
        print(start_date)
        # Filter the image collection by date range and the desired band (dataset) within the image collection
        filtered_collection = image_collection.select(band).filterDate(start_date, end_date)
        # Iterate through each polygon in the shapefile
        for polygon in fc.getInfo()['features']:
            # Extract the polygon geometry
            geometry = ee.Geometry(polygon['geometry'])
            # Calculate the mean value within the polygon
            mean_value = filtered_collection.filterBounds(geometry).mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1,
                bestEffort=True
            )
            # Append the result to the list
            results.append({
                'start_date': start_date,
                'end_date': end_date,
                'ADM1_FR': polygon['properties'][common_column],
                band: mean_value.getNumber(band).getInfo()
            })
    # Create a Pandas DataFrame from the results
    results_df = pd.DataFrame(results)
    #Save the dataframe to csv to reduce the need to run this script connecting to GEE
    results_df.to_csv((str(band)+".csv"), index=False)
    print(results_df)
    return(results_df)

def copy_temporal_resolution(df):
    # Match to temporal resolution
    temporal_dfs = []
    for index, row in time_periods.iterrows():
        # Duplicate the initial DataFrame
        df_copy = df.copy()
        # Update the start and end date columns
        df_copy['start_date'] = row['start_date']
        df_copy['end_date'] = row['end_date']
        # Append the duplicated DataFrame to the result list
        temporal_dfs.append(df_copy)
    # Combine the duplicated DataFrames into a single DataFrame
    final_df = pd.concat(temporal_dfs, ignore_index=True)
    return final_df
    print("Copied temporal resolution across df")

def normalize_minmax(df, column, min=None, max=None):
    normalized_df = df
    if min is None and max is None:
        normalized_df[column] = ((df[column] - df[column].min()) / (df[column].max() - df[column].min()))*10
        # print("both are none")
    elif min is None:
        normalized_df[column] = ((df[column] - df[column].min()) / (max - df[column].min()))*10
        # print("min is none")
    elif max is None:
        normalized_df[column] = ((df[column] - min) / (df[column].max() - min))*10
        # print("max is none")
    else:
        normalized_df[column] = ((df[column] - min) / (max - min))*10
        # print("both defined")
    return normalized_df

############################# Data Loading #################################
### Incidence data
# Load data
incidence_df = pd.read_csv('Regional_Incidence.csv')
# Create a dataframe of only the unqiue dates to set as the temporal resolution of other datasets
time_periods = pd.DataFrame(incidence_df[["start_date", "end_date"]]).drop_duplicates()
# Merge incidence data with spatial data (admin boundaries)
summary_incidence = admin_boundaries.merge(incidence_df, on=common_column, how="left")
incidence = summary_incidence[['start_date', 'end_date', 'ADM1_FR', 'cases', 'deaths']]
# Add to master dataframe
add_dataframe_to_master(incidence)
print("Collected incidence")

### Precipitation (GEE)
# Extract data from GEE
# precipitation_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'total_precipitation_sum')
# After first GEE extraction, run this to only use the csv and no longer connect to GEE
precipitation_df = pd.read_csv('total_precipitation_sum.csv')
# Add to master dataframe
add_dataframe_to_master(precipitation_df)
print("Collected precipitation")

### Temperature (GEE)
# Extract data from GEE
# temperature_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'skin_temperature')
# After first GEE extraction, run this to only use the csv and no longer connect to GEE
temperature_df = pd.read_csv('skin_temperature.csv')
# Add to master dataframe
add_dataframe_to_master(temperature_df)
print("Collected surface temperature")

### Poverty(CSV)
# Load data
wealth_df = pd.read_csv('Datasets_Directory/Relative_wealth_index.csv')
# Define point coordinates as geometry
geometry = [Point(xy) for xy in zip(wealth_df.longitude, wealth_df.latitude)]
# Format to geodataframe
wealth_gdf = gpd.GeoDataFrame(wealth_df, geometry=geometry)
wealth_gdf.crs = "EPSG:4326"
# Complete spaital join with admin boundaries shapefile
merged_wealth = gpd.sjoin(wealth_gdf, admin_boundaries, how="left", op="within")
# Group by admin boundaries name column
mean_wealth = merged_wealth.groupby(common_column)["rwi"].mean().reset_index()
# Inverse wealth to refer to poverty, the higher the rwi the greater the wealth
poverty = mean_wealth
poverty['rwi'] = (-1*mean_wealth['rwi'])
poverty.rename(columns={'rwi': 'Poverty'}, inplace=True)
# Align to desired temporal resolution
poverty_df = copy_temporal_resolution(poverty)
# Add to master dataframe
add_dataframe_to_master(poverty_df)
print("Collected poverty")

# Hazards (CSV)
disasters_df = pd.read_csv('Datasets_Directory/Climate-relatedDisasters.csv')
merged_disasters = admin_boundaries.merge(disasters_df, on=common_column, how="left")
merged_disasters.rename(columns={'Average': 'Hazards'}, inplace=True)
only_hazards = merged_disasters[['ADM1_FR', 'Hazards']]
# Align to desired temporal resolution
hazard_df = copy_temporal_resolution(only_hazards)
# Add to master dataframe
add_dataframe_to_master(hazard_df)
print("Collected hazards")

#Conflicts (CSV)
input_conflicts_df = pd.read_csv('Datasets_Directory/Conflicts.csv')
merged_conflicts = admin_boundaries.merge(input_conflicts_df, on=common_column, how="left")
#Define only the 2 columns needed in dataframe
conflicts = merged_conflicts[['ADM1_FR', 'Conflicts']]
# Align to desired temporal resolution
conflicts_df = copy_temporal_resolution(conflicts)
# Add to master dataframe
add_dataframe_to_master(conflicts_df)
print("Collected conflicts")

# WASH (CSV)
WASH_df = pd.read_csv('Datasets_Directory/WASH.csv')
merged_WASH = admin_boundaries.merge(WASH_df, on=common_column, how="left")
# Define only the 2 columns needed in dataframe
WASH = merged_WASH[['ADM1_FR', 'Insufficient_WASH']]
# Align to desired temporal resolution
wash_df = copy_temporal_resolution(WASH)
# Add to master dataframe
add_dataframe_to_master(wash_df)
print("Collected WASH")

# Health Care Facilities (CSV)
HCF_df = pd.read_csv('Datasets_Directory/HCF.csv')
merged_HCF = admin_boundaries.merge(HCF_df, on=common_column, how="left")
# Define only the 2 columns needed in dataframe
HCF = merged_HCF[['ADM1_FR', 'Pop_HCFs']]
# Align to desired temporal resolution
hcf_df = copy_temporal_resolution(HCF)
# Add to master dataframe
add_dataframe_to_master(hcf_df)
print("Collected HCF")

#################### Finished collecting datasets #######################
print(master_df)
print("Merged dataframes")

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

## Plots of indicators vs incidence
# Get the number of unique y datasets
unique_datasets = [col for col in master_df.columns if col not in index_columns]
# Calculate the number of rows and columns for subplots
num_rows = len(unique_datasets) // 3 + (len(unique_datasets) % 3 > 0)
num_cols = min(len(unique_datasets), 3)
# Create base plot
fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 12))
# Loop through datasets
for i, dataset in enumerate(unique_datasets):
    row, col = divmod(i, num_cols)
    ax = axs[row, col]
    ax.scatter(master_df['cases'], master_df[dataset], marker='o', label='Data')
    # Calculate the coefficients of the best-fit line
    coefficients = np.polyfit(master_df['cases'], master_df[dataset], 1)
    slope, intercept = coefficients
    # Create the best-fit line equation
    line_equation = f'Best-fit Line: y = {slope:.2f}x + {intercept:.2f}'
    # Plot the best-fit line
    ax.plot(master_df['cases'], slope * master_df['cases'] + intercept, color='red', linestyle='--',
            label=line_equation)
    ax.set_title(f'Scatter Plot for Cases vs {dataset}')
    ax.set_xlabel('cases')
    ax.set_ylabel(dataset)
    ax.legend(loc='upper left')
    ax.grid(True)
# Remove any empty subplots
for i in range(len(unique_datasets), num_rows * num_cols):
    fig.delaxes(axs.flatten()[i])
plt.tight_layout()
plt.show()

################### Outlier removal #####################
# NB: Initially removed 293 outliers so possibly too extreme for this work
# outliers_removed_count = 0
# for dataset in [col for col in master_df.columns if col not in index_columns]:
#   Q1 = master_df[dataset].quantile(0.25)
#   Q3 = master_df[dataset].quantile(0.75)
#   IQR = Q3 - Q1
#   threshold = 1.5 * IQR
#   # Create a boolean mask for identifying outliers
#   outlier_mask = (master_df[dataset] < (Q1 - threshold)) | (master_df[dataset] > (Q3 + threshold))
#   # Print the dataset name and the number of outliers removed
#   outliers_removed = outlier_mask.sum()
#   print(f'Dataset: {dataset}, Outliers Removed: {outliers_removed}')
#   # Print the index values of the removed outliers
#   removed_outlier_indices = master_df.index[outlier_mask].tolist()
#   print(f'Removed Outlier Indices: {removed_outlier_indices}')
#
#   # Remove outliers from the DataFrame
#   df_no_outliers = master_df[dataset][~outlier_mask]
#   outliers_removed_count += outliers_removed
# print(f"Total Outliers Removed: {outliers_removed_count}")
# print("Outliers removed")

################### Normalize df of indicators ######################
# NB: Currently not setting fixed min and max values for any dataset, this might need to be changed. Also nromalization might need to be done per region or time period.
normalized_df = master_df
for dataset in [col for col in master_df.columns if col not in index_columns]:
  normalized_df = normalize_minmax(master_df, dataset)
print(normalized_df)
print("Datasets normalized")

#################### Complete multi linear regression ##########################
normalized_df_no_id = normalized_df.drop(columns=index_columns)
reg = LinearRegression()
res = reg.fit(normalized_df_no_id,incidence['cases'])
print(f"Regression coefficients: {reg.coef_}")

est = sm.OLS(incidence['cases'], normalized_df_no_id)
est2 = est.fit()
print(est2.summary())

##################### Aggregation #########################
## Combine all the incidence data for each admin level
# Only find the mean of numeric columns
numeric_columns = normalized_df.select_dtypes(include=['number']).columns.tolist()
# Find the mean per admin level
time_aggregated_df = normalized_df.groupby(common_column)[numeric_columns].mean()
print(time_aggregated_df)
## Combine the datasets into values per dimension
# Define the dimensions
dimensions = {
    'Hazard and Exposure': ['total_precipitation_sum', 'skin_temperature', 'Hazards'],
    'Vulnerability': ['Poverty', 'Conflicts'],
    'Lack of Coping Capacity': ['Insufficient_WASH', 'Pop_HCFs'],
    'Risk': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Poverty', 'Conflicts', 'Insufficient_WASH', 'Pop_HCFs']
}
# Create empty dataframe for the dimension means
dimension_aggregated_df = pd.DataFrame()
# Loop through every dimension to calculate the mean of all the relevant columns
for dimension, columns_to_agg in dimensions.items():
    # Group by 'Category' and calculate the mean for other columns
    dimension_mean = time_aggregated_df[columns_to_agg].mean(axis=1)
    # Append the aggregated data to the result dataframe
    dimension_aggregated_df[dimension] = dimension_mean
# Print the aggregated dataframe
print(dimension_aggregated_df)

##################### Cluster Analysis - Unsure what Inform uses this for??  #########################
# # Define the number of clusters you want to create (for the Inform method this is 5)
# num_clusters = 5
# # Initialize a DataFrame to store cluster labels for each column
# cluster_labels_df = pd.DataFrame(columns=dimension_aggregated_df.columns)
# # Iterate through columns (excluding 'Country' or other non-numeric columns)
# for column in dimension_aggregated_df.columns:
#     # Perform hierarchical clustering using Ward's method for the current column
#     linkage_matrix = linkage(dimension_aggregated_df[[column]], method='ward', metric='euclidean')
#     # # Use the fcluster function to assign data points to clusters
#     # cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
#     # Add the cluster labels to the cluster_labels_df
#     cluster_labels_df[column] = cluster_labels
# # Add admin level column back to the cluster_labels_df
# cluster_labels_df.index = dimension_aggregated_df.index
# # Print the DataFrame with cluster labels for each column
# print(cluster_labels_df)

##################### Draw the risk maps ########################
# Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
merged_gdf = admin_boundaries.merge(dimension_aggregated_df, on=common_column)
# Define the risk categories)
num_categories = 5
category_labels = ["Very high", "High", "Medium", "Low", "Very low"]
category_colors = [(0.796078431372549, 0.09411764705882353, 0.11372549019607843, 1.0), (0.9921568627450981, 0.4196078431372549, 0.23529411764705882, 1.0), (1.0, 0.7098039215686275, 0.5490196078431373, 1.0), (1.0, 0.8784313725490196, 0.8235294117647058, 1.0), (1.0, 0.9607843137254902, 0.9411764705882353, 1.0)]
# Create a single figure with subplots for each dimension
num_dimensions = len(dimension_aggregated_df.columns)
fig, axs = plt.subplots(1, num_dimensions, figsize=(16, 8))
# Create map per dimension
for i, dimension in enumerate(dimension_aggregated_df.columns):
    ax = axs[i]
    # Create a color map with discrete colors
    colormap = plt.cm.colors.ListedColormap(category_colors)
    # Replace 'Value1' with the column you want to use for shading
    merged_gdf.plot(column=dimension, cmap=colormap, legend=False, ax=ax)
    # Customize plot settings as needed
    ax.set_title(dimension + ' Map')
    ax.set_axis_off()
# Label the map with polygon/region names
for idx, row in merged_gdf.iterrows():
    label = row[common_column]  # Replace 'RegionName' with the actual column name containing region names
    for ax in axs:
        ax.annotate(text=label, xy=row['geometry'].centroid.coords[0], horizontalalignment='center', fontsize=8,
                    color='black')
# Create a custom legend
legend_labels = [mpatches.Patch(color=color, label=label) for color, label in zip(category_colors, category_labels)]
fig.legend(handles=legend_labels, title='Categories', loc='upper right')
# Show the maps
plt.show()




