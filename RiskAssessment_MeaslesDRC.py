import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.mask import mask
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point

########################## Definitions ########################
# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Measles DRC/Data')

# Set spatial resolution and set the common column to merge the data spatially
common_column = "ADM2_FR"

# Set temporal resolution
temporal = "yearly"
temp_res = relativedelta(years=1)
study_start = datetime(2020, 1, 1)
study_end = datetime(2023, 10, 1)
start_dates = []
end_dates = []
current_date = study_start
while current_date <= study_end:
    start_dates.append(current_date)
    end_dates.append(current_date + temp_res)
    current_date += temp_res
data = {'start_date': start_dates, 'end_date': end_dates}
time_periods = pd.DataFrame(data)

# Administrative level SHP file
admin_shp_path = "Administrative Boundaries\cod_admbnda_adm2_rgc_20190911.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
admin_boundaries[common_column] = admin_boundaries[common_column].str.upper()
print("Loaded admin boundaries")

# Create master dataframe to store all the datasets
index_columns = ['start_date', 'end_date', common_column]
master_df = pd.DataFrame(columns=index_columns)

### Functions
def add_dataframe_to_master(dataframe):
    global master_df
    # # Convert date columns to datetime64[ns] if it's not already
    # dataframe['start_date'] = pd.to_datetime(dataframe['start_date'])
    # dataframe['end_date'] = pd.to_datetime(dataframe['end_date'])
    # Add all the individual dataframes to the master dataframe
    master_df = pd.merge(master_df, dataframe, on=index_columns, how='outer')

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

def normalize_by_pop(df, desired_column, column_to_normalize):
    demography_df = pd.read_csv('Datasets_Directory/cod_admpop_adm2_2020.csv')
    demography_df.rename(columns={'admin2Name_fr': common_column},inplace=True)  # Fix the naming of columns
    # Convert the values in the common_column to uppercase
    demography_df[common_column] = demography_df[common_column].str.upper()
    # Merge DataFrame with population data on common_column
    df = pd.merge(df, demography_df, on=common_column, how='left')
    print(df[[common_column, 'T_TL', column_to_normalize]])
    # Initialize the desired column with zeros
    df[desired_column] = 0.0
    # Set condition for non-zero values
    condition = df[column_to_normalize] != 0
    # Perform normalization using the corresponding population value for each region
    df.loc[condition, desired_column] = (df[column_to_normalize] / df['T_TL']) * 100
    return df

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

############################# Data Loading #################################
### Incidence data
# List of CSV files to load
csv_files = ["drc-2020-sem37.csv", "drc-2021-sem44.csv", "drc-2022_sem40.csv", "drc-2023_sem08.csv"] #Unreliable dataset with many incorrect dates and population data
# Columns that are of interest
selected_columns = ['MALADIE', 'ZS', 'TOTALCAS', 'TOTALDECES', 'DEBUTSEM']
# Create an empty DataFrame to store the filtered data
filtered_data = pd.DataFrame()
# Define the chunk size (adjust as needed)
chunk_size = 1000
# Create an empty list to store chunks of data
chunks = []
# Iterate through the CSV files and filter the data
for file in csv_files:
    print(file)
    for i,  chunk in enumerate(pd.read_csv(file, chunksize=chunk_size, usecols=selected_columns, low_memory=False)):
        filtered_chunk = chunk[chunk['MALADIE'] == "ROUGEOLE"]
        chunks.append(filtered_chunk)
# Concatenate the chunks into a single DataFrame
filtered_data = pd.concat(chunks, ignore_index=True)
filtered_data.rename(columns={'ZS': common_column, 'TOTALCAS':'cases', 'TOTALDECES':'deaths', 'DEBUTSEM':'start_date'}, inplace=True) # Fix the naming of columns
filtered_data['start_date'] = pd.to_datetime(filtered_data['start_date'], format="%d/%m/%Y")
# Filter rows in filtered_data that have common values in common_column with admin_boundaries
filtered_data = filtered_data[filtered_data[common_column].isin(admin_boundaries[common_column])]
# Merge admin_boundaries with filtered_data
summary_incidence = admin_boundaries.merge(filtered_data, on=common_column, how="left")
incidence = summary_incidence[['start_date', common_column, 'cases', 'deaths']]
# Aggregate the data temporally
temp_aggregated_data = []
for _, period in time_periods.iterrows():
    start_date = period['start_date']
    end_date = period['end_date']

    for region in incidence[common_column].unique():
        region_subset = incidence[incidence[common_column] == region]
        subset = region_subset[(pd.to_datetime(region_subset['start_date']) <= end_date) & (pd.to_datetime(region_subset['start_date']) >= start_date)] #No end dates in dataset
        if not subset.empty:
            total_cases = subset['cases'].sum()
            total_deaths = subset['deaths'].sum()
            temp_aggregated_data.append({'start_date': start_date, 'end_date': end_date, common_column: region,'cases': total_cases, 'deaths': total_deaths})
temp_aggregated_incidence = pd.DataFrame(temp_aggregated_data)
# Calculate attack rate = (Number of cases/population size)*100
temp_aggregated_incidence = normalize_by_pop(temp_aggregated_incidence, 'Attack Rate', 'cases') #only  150 out of 188 districts found in common with population data
print(temp_aggregated_incidence)
# Add to master dataframe
clean_incidence = temp_aggregated_incidence[['start_date', 'end_date', common_column, 'Attack Rate', 'cases', 'deaths']]
print(clean_incidence)
add_dataframe_to_master(clean_incidence)
print("Collected incidence")

### Population Density (TIFF) - using point data from WorldPop
# Load data
src = rasterio.open('Datasets_Directory/cod_pd_2020_1km.tif')
# Loop through admin areas and extract all values from the tif
total_pop_density = []
for index, row in admin_boundaries.iterrows():
    print(row[common_column])
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    # Exclude negative or invalid values
    valid_values = out_image[out_image >= 0]
    # Check if there are valid values
    if valid_values.size > 0:
        mean_value = np.nanmean(valid_values)
        total_pop_density.append({common_column: row[common_column], 'Pop_Density': mean_value})
    else:
        # If there are no valid values, consider it as NoData
        total_pop_density.append({common_column: row[common_column], 'Pop_Density': np.nan})
pop_density = pd.DataFrame(total_pop_density)
# Align to desired temporal resolution
pop_density_df = copy_temporal_resolution(pop_density)
print(pop_density_df)
# Add to master dataframe
add_dataframe_to_master(pop_density_df)
print("Collected population density")

### Population Movement (areas where they experience the conflict not necessarily the areas where they move to)
# Load data
movement_df = pd.read_csv('Datasets_Directory/rdc_mouvement_de_population_deplace_septembre_2023.csv')
movement_df.rename(columns={'admin2_label': common_column, 'person':'displaced_population'}, inplace=True) # Fix the naming of columns
movement_df[common_column] = movement_df[common_column].str.upper()
merged_movement = admin_boundaries.merge(movement_df, on=common_column, how="left")
print(movement_df[[common_column, 'displaced_population']])
print(merged_movement[[common_column, 'displaced_population']])
merged_movement = normalize_by_pop(merged_movement, 'Percentage_displaced_population', 'displaced_population')
clean_pop_move = merged_movement[[common_column, 'Percentage_displaced_population', 'displaced_population']]
# Align to desired temporal resolution
displaced_pop = copy_temporal_resolution(clean_pop_move)
print(displaced_pop)
# Add to master dataframe
add_dataframe_to_master(displaced_pop)
print("Collected population movement")

### Poverty(CSV)
# Load data
wealth_df = pd.read_csv('Datasets_Directory/cod_relative_wealth_index.csv')
# Define point coordinates as geometry
geometry = [Point(xy) for xy in zip(wealth_df.longitude, wealth_df.latitude)]
# Format to geodataframe
wealth_gdf = gpd.GeoDataFrame(wealth_df, geometry=geometry)
wealth_gdf.crs = "EPSG:4326"
# Complete spatial join with admin boundaries shapefile
merged_wealth = gpd.sjoin(wealth_gdf, admin_boundaries, how="left", op="within")
# Group by admin boundaries name column
mean_wealth = merged_wealth.groupby(common_column)["rwi"].mean().reset_index()
# Inverse wealth to refer to poverty, the higher the rwi the greater the wealth
poverty = mean_wealth
poverty['rwi'] = (-1*mean_wealth['rwi'])
poverty.rename(columns={'rwi': 'Poverty'}, inplace=True)
# Align to desired temporal resolution
poverty_df = copy_temporal_resolution(poverty)
print(poverty_df)
# Add to master dataframe
add_dataframe_to_master(poverty_df)
print("Collected poverty")

### Demography(CSV)
# Load data
demography_df = pd.read_csv('Datasets_Directory/cod_admpop_adm2_2020.csv')
demography_df.rename(columns={'admin2Name_fr': common_column}, inplace=True)
demography_df[common_column] = demography_df[common_column].str.upper()
# Group by admin level and calculate the sum of each group
extracted_demographies = demography_df.groupby(common_column).agg({
    'T_00_04': 'sum',
    'T_agee': 'sum', #Need to work out what the age range is for this, or maybe remove as adults aren't vulnerable only children and maybe teenagers
}).reset_index()
# extracted_demographies['Région'] = extracted_demographies['Région'].str.title() might still be need
# Merge on admin boundaries
merged_demographies = admin_boundaries.merge(extracted_demographies, on=common_column, how="left")
merged_demographies['Tot_Vulnerable_Population'] = (merged_demographies['T_00_04']+merged_demographies['T_agee'])
print(merged_demographies)
merged_demographies = normalize_by_pop(merged_demographies, 'Percentage_Vulnerable_Population', 'Tot_Vulnerable_Population')
target_demography = merged_demographies[[common_column, 'Percentage_Vulnerable_Population']]
# Align to desired temporal resolution
vulnerable_demographies_df = copy_temporal_resolution(target_demography)
print(vulnerable_demographies_df)
# Add to master dataframe
add_dataframe_to_master(vulnerable_demographies_df)
print("Collected demography")

### Health Care Facilities (CSV)
# Load data
HCF_df = pd.read_csv('Datasets_Directory/sub-saharan_health_facilities.csv')
# Define point coordinates as geometry
geometry = [Point(xy) for xy in zip(HCF_df.Long, HCF_df.Lat)]
# Format to geodataframe
HCF_gdf = gpd.GeoDataFrame(HCF_df, geometry=geometry)
HCF_gdf.crs = "EPSG:4326"
# Complete spatial join with admin boundaries shapefile
merged_HCF = gpd.sjoin(HCF_gdf, admin_boundaries, how="left", op="within")
# Group by admin boundaries name column and count number of HCFs
sum_HCF = merged_HCF.groupby(common_column).count().reset_index()
# Normalize
sum_HCF = normalize_by_pop(sum_HCF, 'HCF/pop', 'Country')
clean_HCF = sum_HCF[[common_column, 'HCF/pop', 'Country']]
# Align to desired temporal resolution
HCF_per_pop = copy_temporal_resolution(clean_HCF)
print(HCF_per_pop)
# Add to master dataframe
add_dataframe_to_master(HCF_per_pop)
print("Collected HCFs")

#################### Finished collecting datasets #######################
# Print the final dataframe to a CSV without an additional row for the admin level geometries
print(master_df)
# master_to_csv = master_df.drop(columns='geometry')
# master_df.to_csv('complete_dataset_df_'+temporal+'.csv', index=False)
print("Merged dataframes")

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

## Plots of indicators vs incidence
# Remove time periods that are missing values
master_df_clean = master_df.dropna(how='any')
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
    ax.scatter(master_df_clean['Attack Rate'], master_df_clean[dataset], marker='o', label='Data')
    # Calculate the coefficients of the best-fit line
    coefficients = np.polyfit(master_df_clean['Attack Rate'], master_df_clean[dataset], 1)
    slope, intercept = coefficients
    # Create the best-fit line equation
    line_equation = f'Best-fit Line: y = {slope:.2f}x + {intercept:.2f}'
    # Plot the best-fit line
    ax.plot(master_df_clean['Attack Rate'], slope * master_df_clean['Attack Rate'] + intercept, color='red', linestyle='--',
            label=line_equation)
    ax.set_title(f'Scatter Plot for Attack Rate vs {dataset}')
    ax.set_xlabel('Attack Rate')
    ax.set_ylabel(dataset)
    ax.legend(loc='upper left')
    ax.grid(True)
# Remove any empty subplots
for i in range(len(unique_datasets), num_rows * num_cols):
    fig.delaxes(axs.flatten()[i])
plt.tight_layout()
plt.show()

################### Outlier removal #####################
outliers_removed_count = 0
for dataset in [col for col in master_df.columns if col not in index_columns]:
  Q1 = master_df[dataset].quantile(0.25)
  Q3 = master_df[dataset].quantile(0.75)
  IQR = Q3 - Q1
  threshold = 1.5 * IQR
  # Create a boolean mask for identifying outliers
  outlier_mask = (master_df[dataset] < (Q1 - threshold)) | (master_df[dataset] > (Q3 + threshold))
  # Print the dataset name and the number of outliers removed
  outliers_removed = outlier_mask.sum()
  print(f'Dataset: {dataset}, Outliers Removed: {outliers_removed}')
  # Print the index values of the removed outliers
  removed_outlier_indices = master_df.index[outlier_mask].tolist()
  print(f'Removed Outlier Indices: {removed_outlier_indices}')

  # Remove outliers from the DataFrame
  master_df[dataset] = master_df[dataset][~outlier_mask]
  outliers_removed_count += outliers_removed
print(f"Total Outliers Removed: {outliers_removed_count}")
print("Outliers removed")

################### Normalize df of indicators ######################
# NB: Currently not setting fixed min and max values for any dataset, this might need to be changed. Also normalization might need to be done per region or time period.
normalized_df = master_df
for dataset in [col for col in master_df.columns if col not in ['start_date', 'end_date', common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]:
  normalized_df = normalize_minmax(master_df, dataset)
# Inverse datasets that were not initially indexes and are not yet with the negative influence being the highest value
normalized_df['HCF/pop'] = (10 - normalized_df['HCF/pop'])
normalized_df.rename(columns={'HCF/pop': 'Lack of HCF'}, inplace=True)
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
## Combine the datasets into values per dimension
# Define the dimensions
dimensions = {
    'Hazard and Exposure': ['Pop_Density'],
    'Vulnerability': ['Poverty', 'Percentage_Vulnerable_Population', 'Percentage_displaced_population'],
    'Lack of Coping Capacity': ['Lack of HCF'],
    'Risk': ['Pop_Density', 'Percentage_displaced_population', 'Poverty', 'Percentage_Vulnerable_Population', 'Lack of HCF']
}
# Create empty dataframe for the dimension means
dimension_aggregated_df = pd.DataFrame()
# Loop through every dimension to calculate the mean of all the relevant columns
for dimension, columns_to_agg in dimensions.items():
    # Group by 'Category' and calculate the mean for all columns
    dimension_mean = time_aggregated_df[columns_to_agg].mean(axis=1)
    # Append the aggregated data to the result dataframe
    dimension_aggregated_df[dimension] = dimension_mean
# Print the aggregated dataframe
# Create CSV of risk scores per indicator
risk_score_df = pd.concat([time_aggregated_df, dimension_aggregated_df],axis=1)
risk_score_df.to_csv('risk_score_df_'+temporal+'.csv', index=True)
print("Created risk scores")

##################### Draw the risk maps ########################
# Merge the shapefile GeoDataFrame with the cluster labels DataFrame based on the 'Country' column
merged_gdf = admin_boundaries.merge(dimension_aggregated_df, on=common_column)
merged_time_gdf = admin_boundaries.merge(time_aggregated_df, on=common_column)

# Define the risk categories
category_labels = ["Very low", "Low", "Medium", "High","Very high"]
category_colors = [(1.0, 0.9607843137254902, 0.9411764705882353, 1.0),
                   (1.0, 0.8784313725490196, 0.8235294117647058, 1.0),
                   (1.0, 0.7098039215686275, 0.5490196078431373, 1.0),
                   (0.9921568627450981, 0.4196078431372549, 0.23529411764705882, 1.0),
                   (0.796078431372549, 0.09411764705882353, 0.11372549019607843, 1.0)]
# Define different thresholds for each dimension (customize these values)
dimension_thresholds = {
    "Hazard and Exposure": [1.4, 2.6, 4.0, 6.0, 10.0],
    "Vulnerability": [1.9, 3.2, 4.7, 6.3, 10.0],
    "Lack of Coping Capacity": [3.1, 4.6, 5.9, 7.3, 10.0],
    "Risk": [1.9, 3.4, 4.9, 6.4, 10.0]
}
# Create a single figure with subplots for each dimension
num_dimensions = len(dimension_aggregated_df.columns)
fig, axs = plt.subplots(1, num_dimensions, figsize=(16, 8))
# Create map per dimension
for i, dimension in enumerate(dimension_aggregated_df.columns):
    ax = axs[i]
    colormap = plt.cm.colors.ListedColormap(category_colors)
    thresholds = dimension_thresholds.get(dimension, [0])
    merged_gdf[dimension + '_Category'] = merged_gdf[dimension].apply(lambda x: inform_class_thresholds(x, thresholds))
    # Use the 'dimension_Category' values to map to colors using the colormap
    norm = plt.Normalize(vmin=0, vmax=len(category_labels) - 1)
    merged_gdf['Color'] = merged_gdf[dimension + '_Category'].apply(lambda x: norm(category_labels.index(x)))
    # Plot using the 'Color' column to assign colors based on categories
    merged_gdf.plot(column='Color', cmap=colormap, legend=False, ax=ax)
    ax.set_title(dimension + ' Map')
    ax.set_axis_off()
# Label the map with region names (hide at admin 2 level as too many labels
# for idx, row in merged_gdf.iterrows():
#     label = row[common_column]
#     for ax in axs:
#         ax.annotate(text=label, xy=row['geometry'].centroid.coords[0], horizontalalignment='center', fontsize=8, color='black')
# Create a custom legend
legend_labels = [mpatches.Patch(color=color, label=label) for color, label in zip(category_colors, category_labels)]
fig.legend(handles=legend_labels, title='Risk Level', loc='upper right')
# Show the maps
plt.show()

# Create scatter plot between risk score and incidence rate
for region in master_df[common_column].unique():
    region_incidence = merged_time_gdf[merged_time_gdf[common_column] == region]['Attack Rate']
    region_risk = merged_gdf[merged_gdf[common_column] == region]['Risk']
    plt.scatter(region_risk, region_incidence, label=region, s=50)

plt.xlabel('INFORM Risk Level')
plt.ylabel('Measles Attack Rate')
plt.title('Correlation between risk level and measles incidence')
plt.legend()
plt.show()