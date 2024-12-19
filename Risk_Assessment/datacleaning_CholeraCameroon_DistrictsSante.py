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
import cdsapi
from rasterio.windows import from_bounds
from rasterstats import zonal_stats

########################## Definitions ########################
# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/510/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')

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
admin_boundaries = full_admin_boundaries[['NOM_REGION', common_column, 'geometry', 'POPULATION']]
print("Loaded admin boundaries")
# Create master dataframe to store all the datasets
index_columns = ['start_date', 'end_date', common_column, 'NOM_REGION']
master_df = pd.DataFrame(columns=index_columns)

### Functions
def add_dataframe_to_master(dataframe):
    global master_df
    # Add all the individual dataframes to the master dataframe
    master_df = pd.merge(master_df, dataframe, on=index_columns, how='outer')

def CDS_extraction(imageset, band):
    client = cdsapi.Client()
    results = []
    # simplify geometry to reduce computation size
    simplified_admin_boundaries = admin_boundaries.copy()
    simplified_admin_boundaries['geometry'] = simplified_admin_boundaries['geometry'].simplify(tolerance=0.05, preserve_topology=True)

    for index, row in time_periods.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']
        print(f"Processing period from {start_date} to {end_date}")
        request = {
            "variable": [band],
            "year": [start_date.year],
            "month": [f'{start_date.month:02d}'],
            "time": ["13:00"],
            "area": [13.5, 8, 1, 17],
            "format": "grib",
            "download_format": "unarchived"
        }
        # Define the path for the GRIB file
        grib_foldername = os.path.join(os.getcwd(), "CDS_grib_files")
        grib_filename = os.path.join(grib_foldername, f"{band}_{start_date.year}_{start_date.month:02d}.grib")

        # Retrieve data
        client.retrieve(imageset, request).download(grib_filename)
        print("Made request")
        # Process raster data
        with rasterio.open(grib_filename) as src:
            transform = src.transform
            simplified_admin_boundaries = simplified_admin_boundaries.to_crs(src.crs)

            # Perform zonal statistics for the admin districts
            stats = zonal_stats(
                simplified_admin_boundaries,  # District geometries
                src.read(1),  # Raster data (1st band of the GRIB file)
                affine=transform,  # Affine transform of the raster
                stats=["mean"],  # Statistics to calculate the mean of precipitation
                all_touched=True,  # Consider all pixels touched by the geometry
                nodata=0.0  # Set nodata value for raster
            )

            # Append the results for each district
            for i, stat in enumerate(stats):
                district_name = simplified_admin_boundaries.iloc[i][common_column]
                region_name = simplified_admin_boundaries.iloc[i]['NOM_REGION']
                results.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'DISTRICT_S': district_name,
                    'NOM_REGION': region_name,
                    band: stat['mean']  # Use sum as the aggregation type
                })

    # Save the dataframe to csv in every time period loop (append mode)
    results_df = pd.DataFrame(results)
    results_df.to_csv((str(band) + "_" + temporal + "_DISTRICT.csv"), mode='a', header=not os.path.exists((str(band) + "_" + temporal + "_DISTRICT.csv")), index=False, date_format='%Y-%m-%d', float_format='%.8f')
    print(results_df)
    return results_df

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

############################# Data Loading #################################
### Incidence data
# Load data
incidence_df = pd.read_csv('District_Incidence.csv')
# Merge incidence data with spatial data (admin boundaries)
summary_incidence = admin_boundaries.merge(incidence_df, on=[common_column, "NOM_REGION"], how="left")
incidence = summary_incidence[['start_date', 'end_date', 'NOM_REGION', common_column, 'cases', 'deaths','POPULATION']]
# Aggregate the data temporally
temp_aggregated_data = []
for _, period in time_periods.iterrows():
    start_date = period['start_date']
    end_date = period['end_date']
    for district in incidence[common_column].unique():
        district_subset = incidence[incidence[common_column] == district]
        incidence_start_date = pd.to_datetime(district_subset['start_date'], format='%d/%m/%Y')
        incidence_end_date = pd.to_datetime(district_subset['end_date'], format='%d/%m/%Y')
        subset = district_subset[(incidence_start_date <= end_date) & (incidence_end_date >= start_date)]
        if not subset.empty:
            total_cases = subset['cases'].sum()
            total_deaths = subset['deaths'].sum()
            region_name = district_subset['NOM_REGION'].iloc[0]
            population_size = district_subset['POPULATION'].iloc[0]
            temp_aggregated_data.append({'start_date': start_date, 'end_date': end_date, 'NOM_REGION': region_name, common_column: district,'cases': total_cases, 'deaths': total_deaths, 'POPULATION': population_size})
temp_aggregated_incidence = pd.DataFrame(temp_aggregated_data)

## Calculate attack rate = (Number of cases/population size)*100
# Initialize the desired column with zeros
temp_aggregated_incidence['Attack Rate'] = 0.0
# Perform normalization using the corresponding population value for each region
for index, row in temp_aggregated_incidence.iterrows():
    if row['cases'] > row['POPULATION']:
        temp_aggregated_incidence.loc[index, 'Attack Rate'] = np.nan
    elif row['cases'] != 0:
        temp_aggregated_incidence.loc[index, 'Attack Rate'] = (row['cases'] / row['POPULATION'])
    elif row['cases'] == 0:
        temp_aggregated_incidence.loc[index, 'Attack Rate'] = 0.0
    else:
        temp_aggregated_incidence.loc[index, 'Attack Rate'] = np.nan


# clean out all unnecessary columns
clean_incidence = temp_aggregated_incidence[['start_date', 'end_date', 'NOM_REGION', common_column, 'Attack Rate', 'cases', 'deaths']]
print(clean_incidence)
# Add to master dataframe
add_dataframe_to_master(clean_incidence)
print("Collected incidence")

### Precipitation (CDS) - unit meters
# Extract data from CDS
# precipitation_df = CDS_extraction("reanalysis-era5-single-levels-monthly-means", 'total_precipitation')
# After first CDS extraction, run this to only use the csv and no longer connect to CDS
precipitation_df = pd.read_csv(('total_precipitation_'+temporal+'_DISTRICT.csv'), parse_dates=['start_date', 'end_date'])
# Add to master dataframe
print(precipitation_df)
add_dataframe_to_master(precipitation_df)
print("Collected precipitation")

### Population Density (TIFF) - using point data from Meta (avg # of people per 30-meter grid tile in an amdin boundary); NB: could also use excel sheet from MinSanté which has all admin levels
# Load data
src = rasterio.open('Datasets_Directory/cmr_density_2020.tif')
# Loop through admin areas and extract all values from the tif
total_pop_density = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    mean_value = np.nanmean(out_image)
    total_pop_density.append({'NOM_REGION': row['NOM_REGION'], common_column: row[common_column], 'Pop_Density': mean_value})
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
    sum_value = np.nansum(out_image)
    total_waterbodies.append({'NOM_REGION': row['NOM_REGION'], common_column: row[common_column], 'Water_Bodies': sum_value})
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
# Keep all columns from admin boundaries
mean_wealth_all_columns = admin_boundaries.merge(mean_wealth, on=common_column, how='left')
# Define only the 3 columns needed in dataframe
only_mean_wealth = mean_wealth_all_columns[[common_column, 'NOM_REGION', 'rwi']]
# Inverse wealth to refer to poverty, the higher the rwi the greater the wealth
poverty = only_mean_wealth
poverty['rwi'] = (-1*only_mean_wealth['rwi'])
poverty.rename(columns={'rwi': 'Poverty'}, inplace=True)
# Align to desired temporal resolution
poverty_df = copy_temporal_resolution(poverty)
# Add to master dataframe
add_dataframe_to_master(poverty_df)
print("Collected poverty")

### Household Size (CSV)
hh_size = pd.read_csv('Datasets_Directory/hh_size.csv')
merged_hh = admin_boundaries.merge(hh_size, on='NOM_REGION', how="left")
# Define only the 2 columns needed in dataframe
only_hh_size = merged_hh[['NOM_REGION', common_column, 'Avg_HH_size']]
# Align to desired temporal resolution
hh_size_df = copy_temporal_resolution(only_hh_size)
# Add to master dataframe
print(hh_size_df)
add_dataframe_to_master(hh_size_df)
print("Collected average household size")

### Hazards (CSV)
disasters_df = pd.read_csv('Datasets_Directory/Climate-relatedDisasters.csv')
merged_disasters = admin_boundaries.merge(disasters_df, on='NOM_REGION', how="left")
merged_disasters.rename(columns={'Average': 'Hazards'}, inplace=True)
#Define only the 2 columns needed in dataframe
only_hazards = merged_disasters[['NOM_REGION', common_column, 'Hazards']]
# Align to desired temporal resolution
hazard_df = copy_temporal_resolution(only_hazards)
# Add to master dataframe
add_dataframe_to_master(hazard_df)
print("Collected hazards")

#Conflicts (CSV)
input_conflicts_df = pd.read_csv('Datasets_Directory/Conflicts.csv')
merged_conflicts = admin_boundaries.merge(input_conflicts_df, on='NOM_REGION', how="left")
#Define only the 2 columns needed in dataframe
only_conflicts = merged_conflicts[['NOM_REGION', common_column, 'Conflicts']]
# Align to desired temporal resolution
conflicts_df = copy_temporal_resolution(only_conflicts)
# Add to master dataframe
add_dataframe_to_master(conflicts_df)
print("Collected conflicts")

# WASH (CSV)
WASH_df = pd.read_csv('Datasets_Directory/WASH.csv')
merged_WASH = admin_boundaries.merge(WASH_df, on='NOM_REGION', how="left")
# Define only the 2 columns needed in dataframe
WASH = merged_WASH[['NOM_REGION', common_column, 'Insufficient_WASH']]
# Align to desired temporal resolution
wash_df = copy_temporal_resolution(WASH)
# Add to master dataframe
add_dataframe_to_master(wash_df)
print("Collected WASH")

# Public health facilities (CSV)
ph_facility = pd.read_csv('Datasets_Directory/formations_sanitaires.csv')
merged_ph_facility = admin_boundaries.merge(ph_facility, on='NOM_REGION', how="left")
# Define only the 2 columns needed in dataframe
ph_facility_clean = merged_ph_facility[['NOM_REGION', common_column, 'FOSA/1000_hab.']]
# Align to desired temporal resolution
ph_facility_df = copy_temporal_resolution(ph_facility_clean)
# Add to master dataframe
add_dataframe_to_master(ph_facility_df)
print("Collected public health facility")

# Prisons (CSV) - made boolean although the districts of Tchollire, Buea and Bangangte have 2 prisons
prisons = pd.read_csv('Datasets_Directory/prisons.csv')
merged_prisons = admin_boundaries.merge(prisons, on=common_column, how="left")
# Replace missing values with 0 (for districts without prisons in the CSV)
merged_prisons['Prison'].fillna(0, inplace=True)
# Define only the 2 columns needed in dataframe
prisons_clean = merged_prisons[['NOM_REGION', common_column, 'Prison']]
# Align to desired temporal resolution
prisons_df = copy_temporal_resolution(prisons_clean)
# Add to master dataframe
add_dataframe_to_master(prisons_df)
print("Collected prisons")

# Border region risk (csv) - boolean for districts with a border with other countries
borders = pd.read_csv('Datasets_Directory/borders.csv')
merged_borders = admin_boundaries.merge(borders, on=common_column, how="left")
# Replace missing values with 0 (for districts without border risk in the CSV)
merged_borders['Border'].fillna(0, inplace=True)
# Define only the 2 columns needed in dataframe
borders_clean = merged_borders[['NOM_REGION', common_column, 'Border']]
# Align to desired temporal resolution
borders_df = copy_temporal_resolution(borders_clean)
# Add to master dataframe
add_dataframe_to_master(borders_df)
print("Collected borders")

# Roads/Accessibility - area with main roads are more connected and so have higher risk and areas with no roads have low accessibility also have higher risk
# Read the vector line file for roads
roads = gpd.read_file('Datasets_Directory/hotosm_cmr_roads_lines.shp')
# Reproject the roads to a projected CRS (UTM zone 33N - EPSG:32633 for Cameroon) for accurate length calculation
roads_projected = roads.to_crs('EPSG:32633')
# Add a new column 'road_length' to the roads GeoDataFrame with the length of each road segment in meters
roads['road_length'] = roads.geometry.length
# Perform a spatial join to associate each road with a district in admin_boundaries
roads_in_districts = gpd.sjoin(roads, admin_boundaries, how='left', op='intersects')
# Sum the length of road segments within each district
road_length_per_district = roads_in_districts.groupby(common_column)['road_length'].sum().reset_index(name='Roads')
# Merge the road lengths with the admin boundaries to ensure all districts are included
merged_roads = admin_boundaries.merge(road_length_per_district, on=common_column, how='left')
# Define only the columns needed in the final dataframe
roads_clean = merged_roads[['NOM_REGION', common_column, 'Roads']]
# Align to desired temporal resolution
roads_df = copy_temporal_resolution(roads_clean)
# Add the dataframe to the master dataframe
add_dataframe_to_master(roads_df)
print("Collected and normalized roads data")

# PAMI (csv)
pami = pd.read_csv('Datasets_Directory/PAMI.csv')
merged_pami = admin_boundaries.merge(pami, on=['NOM_REGION', common_column], how="left")
# Define only the 2 columns needed in dataframe
pami_clean = merged_pami[['NOM_REGION', common_column, 'PAMI']]
# Align to desired temporal resolution
pami_df = copy_temporal_resolution(pami_clean)
# Add to master dataframe
add_dataframe_to_master(pami_df)
print("Collected borders")

#################### Finished collecting datasets #######################
# Print the final dataframe to a CSV without an additional row for the admin level geometries
print(master_df)
master_df.to_csv('complete_dataset_df_'+temporal+'_districts.csv', index=False, float_format='%.8f')
print("Merged dataframes")