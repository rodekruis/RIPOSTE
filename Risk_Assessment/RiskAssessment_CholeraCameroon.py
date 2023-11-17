import geopandas as gpd
import pandas as pd
import os
import ee
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.mask import mask
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point

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
temp_res = relativedelta(months=1)
study_start = datetime(2021, 10, 1)
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
admin_shp_path = "Administrative Boundaries\cmr_admbnda_inc_20180104_SHP\cmr_admbnda_adm1_inc_20180104.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
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

def gee_extraction(imageset, band):
    image_collection = ee.ImageCollection(imageset)
    # Convert the shapefile to a GeoJSON FeatureCollection
    fc = ee.FeatureCollection(admin_boundaries.__geo_interface__)
    # Create an empty list for all the results to be appended to inside the for loop
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
            print(polygon['properties'][common_column])
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
    results_df.to_csv((str(band)+"_"+temporal+".csv"), index=False, date_format='%Y-%m-%d')
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

def normalize_by_pop(df, desired_column, column_to_normalize):
    demography_df = pd.read_csv('Datasets_Directory/demography_2023.csv')
    extracted_demographies = demography_df.groupby('Région').agg({'Population 2023 estimée (Les deux sexes)': 'sum'}).reset_index()
    df[desired_column] = 0.0  # Initialize the column with zeros
    condition = df[column_to_normalize] != 0
    df.loc[condition, desired_column] = (df[column_to_normalize] / extracted_demographies['Population 2023 estimée (Les deux sexes)']) * 100

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
# Load data
incidence_df = pd.read_csv('Regional_Incidence.csv')
# Merge incidence data with spatial data (admin boundaries)
summary_incidence = admin_boundaries.merge(incidence_df, on=common_column, how="left")
incidence = summary_incidence[['start_date', 'end_date', common_column, 'cases', 'deaths']]
# Aggregate the data temporally
temp_aggregated_data = []
for _, period in time_periods.iterrows():
    start_date = period['start_date']
    end_date = period['end_date']
    for region in incidence[common_column].unique():
        region_subset = incidence[incidence[common_column] == region]
        subset = region_subset[(pd.to_datetime(region_subset['start_date']) <= end_date) & (pd.to_datetime(region_subset['end_date']) >= start_date)]
        if not subset.empty:
            total_cases = subset['cases'].sum()
            total_deaths = subset['deaths'].sum()
            temp_aggregated_data.append({'start_date': start_date, 'end_date': end_date, common_column: region,'cases': total_cases, 'deaths': total_deaths})
temp_aggregated_incidence = pd.DataFrame(temp_aggregated_data)
# Calculate attack rate = (Number of cases/population size)*100
normalize_by_pop(temp_aggregated_incidence, 'Attack Rate', 'cases')
# Add to master dataframe
add_dataframe_to_master(temp_aggregated_incidence)
print("Collected incidence")

### Precipitation (GEE)
# # Extract data from GEE
# precipitation_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'total_precipitation_sum')
# After first GEE extraction, run this to only use the csv and no longer connect to GEE
precipitation_df = pd.read_csv(('total_precipitation_sum_'+temporal+'.csv'), parse_dates=['start_date', 'end_date'])
# Add to master dataframe
add_dataframe_to_master(precipitation_df)
print("Collected precipitation")

# # Create a time series plot for each region
# regions = precipitation_df['ADM1_FR'].unique()
# for region in regions:
#     region_data = precipitation_df[precipitation_df['ADM1_FR'] == region]
#     plt.plot(region_data['start_date'], region_data['total_precipitation_sum'], label=region)
#
# plt.xlabel('Date')
# plt.ylabel('Precipitation')
# plt.title('Precipitation Time Series by Region')
# plt.legend()
# plt.show()

### Temperature (GEE)
# # Extract data from GEE
# temperature_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'skin_temperature')
# After first GEE extraction, run this to only use the csv and no longer connect to GEE
temperature_df = pd.read_csv('skin_temperature_'+temporal+'.csv', parse_dates=['start_date', 'end_date'])
# Add to master dataframe
add_dataframe_to_master(temperature_df)
print("Collected surface temperature")

# # Create a time series plot for each region
# regions = temperature_df['ADM1_FR'].unique()
# for region in regions:
#     region_data = temperature_df[temperature_df['ADM1_FR'] == region]
#     plt.plot(region_data['start_date'], region_data['skin_temperature'], label=region)
#
# plt.xlabel('Date')
# plt.ylabel('Temperature')
# plt.title('Temperature Time Series by Region')
# plt.legend()
# plt.show()

### Population Density (TIFF) - using point data from Meta (avg # of people per 30-meter grid tile in an amdin boundary); NB: could also use excel sheet from MinSanté which has all admin levels
# Load data
src = rasterio.open('Datasets_Directory/cmr_density_2020.tif')
# Loop through admin areas and extract all values from the tif
total_pop_density = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    mean_value = np.nanmean(out_image)
    total_pop_density.append({common_column: row[common_column], 'Pop_Density': mean_value})
    pop_density = pd.DataFrame(total_pop_density)
# Align to desired temporal resolution
pop_density_df = copy_temporal_resolution(pop_density)
# Add to master dataframe
add_dataframe_to_master(pop_density_df)
print("Collected population density")

### Proximity to water bodies (TIF)
# Load data
src = rasterio.open('Datasets_Directory/WaterBodies.tif')
# Loop through admin areas and extract all values from the tif
total_waterbodies = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    mean_value = np.nanmean(out_image)
    total_waterbodies.append({common_column: row['ADM1_FR'], 'Water_Bodies': mean_value})
    waterbodies = pd.DataFrame(total_waterbodies)
# Align to desired temporal resolution
waterbodies_df = copy_temporal_resolution(waterbodies)
# Add to master dataframe
add_dataframe_to_master(waterbodies_df)
print("Collected water bodies")

### Poverty(CSV)
# Load data
wealth_df = pd.read_csv('Datasets_Directory/Relative_wealth_index.csv')
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
# Add to master dataframe
add_dataframe_to_master(poverty_df)
print("Collected poverty")

### Demography(CSV)
# Load data
demography_df = pd.read_csv('Datasets_Directory/demography_2023.csv')
demography_df.rename(columns={'Région': common_column}, inplace=True)
# Group by admin level and calculate the sum of each group
extracted_demographies = demography_df.groupby(common_column).agg({
    'Population 2023 estimée (Les deux sexes)': 'sum',
    'enfants 0-59 mois': 'sum',
    'Femmes enceintes attendues': 'sum',
    '50 ans et plus Masculin': 'sum',
    '50 ans et plus Féminin': 'sum'
}).reset_index()
# extracted_demographies['Région'] = extracted_demographies['Région'].str.title() might still be need
# Merge on admin boundaries
merged_demographies = admin_boundaries.merge(extracted_demographies, on=common_column, how="left")
merged_demographies['Tot_Vulnerable_Population'] = (merged_demographies['enfants 0-59 mois']+merged_demographies['Femmes enceintes attendues']+merged_demographies['50 ans et plus Masculin']+merged_demographies['50 ans et plus Féminin'])
normalize_by_pop(merged_demographies, 'Percentage_Vulnerable_Population', 'Tot_Vulnerable_Population')
target_demography = merged_demographies[[common_column, 'Percentage_Vulnerable_Population']]
# extracted_demographies['Percentage_Children_under_5'] = (extracted_demographies['enfants 0-59 mois']/extracted_demographies['Population 2023 estimée (Les deux sexes)'])*100
# extracted_demographies['Percentage_Adults_over_50'] = ((extracted_demographies['50 ans et plus Masculin']+extracted_demographies['50 ans et plus Féminin'])/extracted_demographies['Population 2023 estimée (Les deux sexes)'])*100
# extracted_demographies['Percentage_Pregnant_Women'] = (extracted_demographies['Femmes enceintes attendues']/extracted_demographies['Population 2023 estimée (Les deux sexes)'])*100
# extracted_demographies['Région'] = extracted_demographies['Région'].str.title()
# # Merge on admin boundaries
# merged_demographies = admin_boundaries.merge(extracted_demographies, on=common_column, how="left")
# merged_demographies['Avg_Percentage_Vulnerable_Population'] = (merged_demographies['Percentage_Children_under_5']+merged_demographies['Percentage_Adults_over_50']+merged_demographies['Percentage_Pregnant_Women'])/3
# target_demography = merged_demographies[['ADM1_FR', 'Avg_Percentage_Vulnerable_Population']]
# Align to desired temporal resolution
vulnerable_demographies_df = copy_temporal_resolution(target_demography)
# Add to master dataframe
add_dataframe_to_master(vulnerable_demographies_df)
print("Collected demography")

### Household Size (CSV)
hh_size = pd.read_csv('Datasets_Directory/hh_size.csv')
merged_hh = admin_boundaries.merge(hh_size, on=common_column, how="left")
# Align to desired temporal resolution
hh_size_df = copy_temporal_resolution(merged_hh)
# Add to master dataframe
add_dataframe_to_master(hh_size_df)
print("Collected average household size")

### Hazards (CSV)
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

### Health Care Facilities (CSV)
HCF_df = pd.read_csv('Datasets_Directory/HCF.csv')
merged_HCF = admin_boundaries.merge(HCF_df, on=common_column, how="left")
# Define only the 2 columns needed in dataframe
HCF = merged_HCF[['ADM1_FR', 'Pop_HCFs']]
# Align to desired temporal resolution
hcf_df = copy_temporal_resolution(HCF)
# Add to master dataframe
add_dataframe_to_master(hcf_df)
print("Collected HCF")

### Public health training (CSV)
ph_training = pd.read_csv('Datasets_Directory/formations_sanitaires.csv')
merged_ph_training = admin_boundaries.merge(ph_training, on=common_column, how="left")
# Define only the 2 columns needed in dataframe
ph_training_clean = merged_ph_training[['ADM1_FR', 'FOSA/1000_hab.']]
# Align to desired temporal resolution
ph_training_df = copy_temporal_resolution(ph_training_clean)
# Add to master dataframe
add_dataframe_to_master(ph_training_df)
print("Collected public health training")


#################### Finished collecting datasets #######################
#Remove time periods that are missing values
master_df = master_df.dropna(how='any')

# Print the final dataframe to a CSV without an additional row for the admin level geometries
print(master_df[['start_date', 'end_date', common_column, 'cases', 'Avg_HH_size']])
master_to_csv = master_df.drop(columns='geometry')
master_to_csv.to_csv('complete_dataset_df_'+temporal+'.csv', index=False)
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
unique_datasets = [col for col in master_df.columns if col not in ['start_date', 'end_date', common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
# Calculate the number of rows and columns for subplots
num_rows = len(unique_datasets) // 3 + (len(unique_datasets) % 3 > 0)
num_cols = min(len(unique_datasets), 3)
# Create base plot
fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 12))
# Loop through datasets
for i, dataset in enumerate(unique_datasets):
    row, col = divmod(i, num_cols)
    ax = axs[row, col]
    ax.scatter(master_df['Attack Rate'], master_df[dataset], marker='o', label='Data')
    # Calculate the coefficients of the best-fit line
    coefficients = np.polyfit(master_df['Attack Rate'], master_df[dataset], 1)
    slope, intercept = coefficients
    # Create the best-fit line equation
    line_equation = f'Best-fit Line: y = {slope:.2f}x + {intercept:.2f}'
    # Plot the best-fit line
    ax.plot(master_df['Attack Rate'], slope * master_df['Attack Rate'] + intercept, color='red', linestyle='--',
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
## Combine the datasets into values per dimension
# Define the dimensions
dimensions = {
    'Hazard and Exposure': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies'],
    'Vulnerability': ['Poverty', 'ercentage_Vulnerable_Population', 'Conflicts', 'Avg_HH_size'],
    'Lack of Coping Capacity': ['Pop_HCFs', 'Lack_of_PH_Training'],
    'Risk': ['total_precipitation_sum', 'skin_temperature', 'Hazards', 'Insufficient_WASH', 'Pop_Density', 'Water_Bodies', 'Poverty', 'Percentage_Vulnerable_Population', 'Conflicts', 'Avg_HH_size', 'Pop_HCFs', 'Lack_of_PH_Training']
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
# Label the map with region names
for idx, row in merged_gdf.iterrows():
    label = row[common_column]
    for ax in axs:
        ax.annotate(text=label, xy=row['geometry'].centroid.coords[0], horizontalalignment='center', fontsize=8, color='black')
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
plt.ylabel('Cholera Attack Rate')
plt.title('Correlation between risk level and cholera incidence')
plt.legend()
plt.show()