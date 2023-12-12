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
# Make all admin names upper case to match other dataset inputs
admin_boundaries[common_column] = admin_boundaries[common_column].str.upper()
print("Loaded admin boundaries")

# Create master dataframe to store all the datasets
index_columns = ['start_date', 'end_date', common_column]
master_df = pd.DataFrame(columns=index_columns)

### Functions
def add_dataframe_to_master(dataframe):
    global master_df
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
    # Initialize the desired column with zeros
    df[desired_column] = 0.0
    # Perform normalization using the corresponding population value for each region
    for index, row in df.iterrows():
        if row[column_to_normalize] > row['T_TL']:
            df.loc[index, desired_column] = np.nan
        elif row[column_to_normalize] != 0:
            df.loc[index, desired_column] = (row[column_to_normalize] / row['T_TL'])
        elif row[column_to_normalize] == 0:
            df.loc[index, desired_column] = 0.0
        else:
            df.loc[index, desired_column] = np.nan
    return df

############################# Data Loading #################################
### Incidence data (multipe files and files to large to read in one go so need to split the csv files into chunks to read)
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
# clean out all unnecessary columns
clean_incidence = temp_aggregated_incidence[['start_date', 'end_date', common_column, 'Attack Rate', 'cases', 'deaths']]
# Add to master dataframe
add_dataframe_to_master(clean_incidence)
print("Collected incidence")

### Population Density (TIFF) - using point data from WorldPop
# Load data
src = rasterio.open('Datasets_Directory/cod_pd_2020_1km.tif')
# Loop through admin areas and extract all values from the tif
total_pop_density = []
for index, row in admin_boundaries.iterrows():
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
# Add to master dataframe
add_dataframe_to_master(pop_density_df)
print("Collected population density")

### Population Movement (areas where they experience the conflict not necessarily the areas where they move to)
# Load data
movement_df = pd.read_csv('Datasets_Directory/rdc_mouvement_de_population_deplace_septembre_2023.csv')
movement_df.rename(columns={'admin2_label': common_column, 'person':'displaced_population'}, inplace=True) # Fix the naming of columns
movement_df[common_column] = movement_df[common_column].str.upper()
# Group by common_column and sum the displaced_population
grouped_movement = movement_df.groupby(common_column)['displaced_population'].sum().reset_index()
merged_movement = admin_boundaries.merge(grouped_movement, on=common_column, how="left")
# Fill NaN values with 0 in the 'displaced_population' column
merged_movement['displaced_population'] = merged_movement['displaced_population'].fillna(0)
merged_movement = normalize_by_pop(merged_movement, 'Fraction_displaced_population', 'displaced_population')
clean_pop_move = merged_movement[[common_column, 'Fraction_displaced_population']]
# Align to desired temporal resolution
displaced_pop = copy_temporal_resolution(clean_pop_move)
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
# Add to master dataframe
add_dataframe_to_master(poverty_df)
print("Collected poverty")

### Demography(CSV)
# Load data
demography_df = pd.read_csv('Datasets_Directory/cod_admpop_adm2_2020.csv')
demography_df.rename(columns={'admin2Name_fr': common_column}, inplace=True)
demography_df[common_column] = demography_df[common_column].str.upper()
# Group by admin level and calculate the sum of each group
extracted_demographies = demography_df.groupby(common_column).agg({ # if the excel has a blank cell the sum shouldn't make them 0s
    'T_00_04': lambda x: x.sum(skipna=True) if any(x.notna()) else np.nan,
    'T_agee': lambda x: x.sum(skipna=True) if any(x.notna()) else np.nan, #Need to work out what the age range is for this, or maybe remove as adults aren't vulnerable only children and maybe teenagers
}).reset_index()
# Merge on admin boundaries
merged_demographies = admin_boundaries.merge(extracted_demographies, on=common_column, how="left")
merged_demographies['Tot_Vulnerable_Population'] = (merged_demographies['T_00_04']+merged_demographies['T_agee'])
print(merged_demographies['Tot_Vulnerable_Population'])
merged_demographies = normalize_by_pop(merged_demographies, 'Fraction_Vulnerable_Population', 'Tot_Vulnerable_Population')
target_demography = merged_demographies[[common_column, 'Fraction_Vulnerable_Population']]
# Align to desired temporal resolution
vulnerable_demographies_df = copy_temporal_resolution(target_demography)
# Add to master dataframe
add_dataframe_to_master(vulnerable_demographies_df)
print(vulnerable_demographies_df)
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
clean_HCF = sum_HCF[[common_column, 'HCF/pop']]
clean_HCF.rename(columns={'Country':'HCFs'}, inplace=True) # Fix the naming of columns
# Align to desired temporal resolution
HCF_per_pop = copy_temporal_resolution(clean_HCF)
# Add to master dataframe
add_dataframe_to_master(HCF_per_pop)
print("Collected HCFs")

### Vaccination coverage (TIFF)
# Load data
src = rasterio.open('Datasets_Directory/Vaccination_coverage/DRC_MCV1_MEAN.tif')
# Loop through admin areas and extract all values from the tif
proportion_vaccinated_children = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    # Exclude negative or invalid values
    valid_values = out_image[out_image >= 0]
    # Check if there are valid values
    if valid_values.size > 0:
        mean_value = np.nanmean(valid_values)
        proportion_vaccinated_children.append({common_column: row[common_column], 'Proportion_Vaccinated_Children': mean_value})
    else:
        # If there are no valid values, consider it as NoData
        proportion_vaccinated_children.append({common_column: row[common_column], 'Proportion_Vaccinated_Children': np.nan})
vaccination_coverage = pd.DataFrame(proportion_vaccinated_children)
# Inverse vaccination to refer to unvaccinated
unvaccinated = vaccination_coverage
unvaccinated['Proportion_Vaccinated_Children'] = (1-vaccination_coverage['Proportion_Vaccinated_Children'])
unvaccinated.rename(columns={'Proportion_Vaccinated_Children': 'Proportion_Unvaccinated_Children'}, inplace=True)
# Align to desired temporal resolution
unvaccinated_df = copy_temporal_resolution(unvaccinated)
# Add to master dataframe
add_dataframe_to_master(unvaccinated_df)
print("Collected vaccination coverage")

#################### Finished collecting datasets #######################
# Remove time periods that are missing incidence dat
master_df = master_df.dropna(subset=['cases'])
# Print the final dataframe to a CSV
print(master_df)
master_df.to_csv('complete_dataset_df_'+temporal+'.csv', index=False)
print("Merged dataframes")
# Remove time periods that are missing incidence data
master_df_clean = master_df.dropna(subset=['Attack Rate'])
