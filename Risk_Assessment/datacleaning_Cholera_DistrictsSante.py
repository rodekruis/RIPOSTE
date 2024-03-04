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
import geopandas as gpd
from shapely.geometry import MultiPolygon

########################## Definitions ########################
# # Log into Google Earth Engine
ee.Authenticate()
ee.Initialize()

# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')

# Set spatial resolution and set the common column to merge the data spatially
common_column = "DISTRICT_S"

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
admin_shp_path = "Administrative Boundaries\Health Boundaries\District_sante_2022.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column, 'geometry']]
print("Loaded admin boundaries")
# Create master dataframe to store all the datasets
index_columns = ['start_date', 'end_date', common_column]
master_df = pd.DataFrame(columns=index_columns)

### Functions
def add_dataframe_to_master(dataframe):
    global master_df
    # Add all the individual dataframes to the master dataframe
    master_df = pd.merge(master_df, dataframe, on=index_columns, how='outer')


def gee_extraction(imageset, band):
    # simplify geometry to reduce computation size
    simplified_admin_boundaries = admin_boundaries
    simplified_admin_boundaries.geometry = simplified_admin_boundaries.simplify(tolerance=0.05, preserve_topology=True)
    # Collect imageset
    image_collection = ee.ImageCollection(imageset)
    # Convert the shapefile to a GeoJSON FeatureCollection
    fc = ee.FeatureCollection(simplified_admin_boundaries.__geo_interface__)
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
    results_df.to_csv((str(band)+"_"+temporal+"_ADM2.csv"), index=False, date_format='%Y-%m-%d')
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
    demography_df.rename(columns={'Région': common_column, 'Population 2023 estimée (Les deux sexes)': 'Population'}, inplace=True)
    extracted_demographies = demography_df.groupby(common_column).agg({'Population': 'sum'}).reset_index()
    # Merge DataFrame with population data on common_column
    df = pd.merge(df, extracted_demographies, on=common_column, how='left')
    # Initialize the desired column with zeros
    df[desired_column] = 0.0
    # Perform normalization using the corresponding population value for each region
    for index, row in df.iterrows():
        if row[column_to_normalize] > row['Population']:
            df.loc[index, desired_column] = np.nan
        elif row[column_to_normalize] != 0:
            df.loc[index, desired_column] = (row[column_to_normalize] / row['Population'])
        elif row[column_to_normalize] == 0:
            df.loc[index, desired_column] = 0.0
        else:
            df.loc[index, desired_column] = np.nan
    return df

############################# Data Loading #################################
### Precipitation (GEE)
# # Extract data from GEE
precipitation_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'total_precipitation_sum')
# After first GEE extraction, run this to only use the csv and no longer connect to GEE
# precipitation_df = pd.read_csv(('total_precipitation_sum_'+temporal+'_districts.csv'), parse_dates=['start_date', 'end_date'])
# Add to master dataframe
add_dataframe_to_master(precipitation_df)
print("Collected precipitation")

### Temperature (GEE)
# # Extract data from GEE
temperature_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'temperature_2m')
# After first GEE extraction, run this to only use the csv and no longer connect to GEE
# temperature_df = pd.read_csv('temperature_2m_'+temporal+'_districts.csv', parse_dates=['start_date', 'end_date'])
# Add to master dataframe
add_dataframe_to_master(temperature_df)
print("Collected surface temperature")

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
# Define a buffer distance around the admin boundaries to ensure the sea is counted to consider pixels around the boundary
buffer_distance = 0.1
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    # Create a buffered geometry to include neighboring pixels
    buffered_geom = geom.buffer(buffer_distance)
    out_image, out_transform = mask(src, [buffered_geom], crop=True)
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

# ### Demography(CSV) - currently no clear vulnerable groups based on literature
# # Load data
# demography_df = pd.read_csv('Datasets_Directory/demography_2023.csv')
# demography_df.rename(columns={'Région': common_column}, inplace=True)
# # Group by admin level and calculate the sum of each group
# extracted_demographies = demography_df.groupby(common_column).agg({
#     'Population 2023 estimée (Les deux sexes)': 'sum',
#     'enfants 0-59 mois': 'sum',
#     'Femmes enceintes attendues': 'sum',
#     '50 ans et plus Masculin': 'sum',
#     '50 ans et plus Féminin': 'sum'
# }).reset_index()
# # Merge on admin boundaries
# merged_demographies = admin_boundaries.merge(extracted_demographies, on=common_column, how="left")
# merged_demographies['Tot_Vulnerable_Population'] = (merged_demographies['enfants 0-59 mois']+merged_demographies['Femmes enceintes attendues']+merged_demographies['50 ans et plus Masculin']+merged_demographies['50 ans et plus Féminin'])
# merged_demographies = normalize_by_pop(merged_demographies, 'Fraction_Vulnerable_Population', 'Tot_Vulnerable_Population')
# target_demography = merged_demographies[[common_column, 'Fraction_Vulnerable_Population']]
# # Align to desired temporal resolution
# vulnerable_demographies_df = copy_temporal_resolution(target_demography)
# # Add to master dataframe
# add_dataframe_to_master(vulnerable_demographies_df)
# print("Collected demography")

### Household Size (CSV)
hh_size = pd.read_csv('Datasets_Directory/hh_size.csv')
merged_hh = admin_boundaries.merge(hh_size, on='NOM_REGION', how="left")
# Define only the 2 columns needed in dataframe
only_hh_size = merged_hh[[common_column, 'Avg_HH_size']]
# Align to desired temporal resolution
hh_size_df = copy_temporal_resolution(only_hh_size)
# Add to master dataframe
add_dataframe_to_master(hh_size_df)
print("Collected average household size")

### Hazards (CSV)
disasters_df = pd.read_csv('Datasets_Directory/Climate-relatedDisasters.csv')
merged_disasters = admin_boundaries.merge(disasters_df, on='NOM_REGION', how="left")
merged_disasters.rename(columns={'Average': 'Hazards'}, inplace=True)
#Define only the 2 columns needed in dataframe
only_hazards = merged_disasters[[common_column, 'Hazards']]
# Align to desired temporal resolution
hazard_df = copy_temporal_resolution(only_hazards)
# Add to master dataframe
add_dataframe_to_master(hazard_df)
print("Collected hazards")

#Conflicts (CSV)
input_conflicts_df = pd.read_csv('Datasets_Directory/Conflicts.csv')
merged_conflicts = admin_boundaries.merge(input_conflicts_df, on='NOM_REGION', how="left")
#Define only the 2 columns needed in dataframe
only_conflicts = merged_conflicts[[common_column, 'Conflicts']]
# Align to desired temporal resolution
conflicts_df = copy_temporal_resolution(only_conflicts)
# Add to master dataframe
add_dataframe_to_master(conflicts_df)
print("Collected conflicts")

# WASH (CSV)
WASH_df = pd.read_csv('Datasets_Directory/WASH.csv')
merged_WASH = admin_boundaries.merge(WASH_df, on='NOM_REGION', how="left")
# Define only the 2 columns needed in dataframe
WASH = merged_WASH[[common_column, 'Insufficient_WASH']]
# Align to desired temporal resolution
wash_df = copy_temporal_resolution(WASH)
# Add to master dataframe
add_dataframe_to_master(wash_df)
print("Collected WASH")

# ### Health Care Facilities (CSV) - Equivalent to FOSAs and less reliable data source
# pop_HCF_df = pd.read_csv('Datasets_Directory/HCF.csv')
# merged_pop_HCF = admin_boundaries.merge(pop_HCF_df, on=common_column, how="left")
# # Define only the 2 columns needed in dataframe
# pop_HCF = merged_pop_HCF[['ADM1_FR', 'Pop_HCFs']]
# # Align to desired temporal resolution
# pop_HCF_temp = copy_temporal_resolution(pop_HCF)
# # Add to master dataframe
# add_dataframe_to_master(pop_HCF_temp)
# print("Collected HCF")

### Public health facilities (CSV)
ph_facility = pd.read_csv('Datasets_Directory/formations_sanitaires.csv')
merged_ph_facility = admin_boundaries.merge(ph_facility, on='NOM_REGION', how="left")
# Define only the 2 columns needed in dataframe
ph_facility_clean = merged_ph_facility[[common_column, 'FOSA/1000_hab.']]
# Align to desired temporal resolution
ph_facility_df = copy_temporal_resolution(ph_facility_clean)
# Add to master dataframe
add_dataframe_to_master(ph_facility_df)
print("Collected public health facility")

#################### Finished collecting datasets #######################
# Print the final dataframe to a CSV without an additional row for the admin level geometries
print(master_df)
master_df.to_csv('complete_dataset_df_'+temporal+'_districts.csv', index=False)
print("Merged dataframes")