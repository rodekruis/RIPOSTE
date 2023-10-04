import geopandas as gpd
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import rasterio
from rasterio.mask import mask
import numpy as np
from functools import reduce
from shapely.geometry import Point
import statsmodels.api as sm
import ee

#Log into Google Earth Engine
ee.Authenticate()
ee.Initialize()

#Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')

# Administrative level SHP file
admin_shp_path = "Administrative Boundaries\cmr_admbnda_inc_20180104_SHP\cmr_admbnda_adm1_inc_20180104.shp"
admin_boundaries = gpd.read_file(admin_shp_path)
print("Loaded admin boundaries")

# Use the common column to merge the data spatially
common_column = "ADM1_FR"

# Initialize empty lists to store merged datasets
hazard_indicators = []
vulnerability_indicators = []
capacity_indicators = []
indicators = []

### Incidence data
#Load data
incidence_df = pd.read_csv('Regional_Incidence.csv')
#Create a dataframe of only the unqiue dates to set as the temporal resolution of other datasets
time_periods = pd.DataFrame(incidence_df[["start_date", "end_date"]]).drop_duplicates()
#Merge incidence data with spatial data (admin boundaries)
summary_incidence = admin_boundaries.merge(incidence_df, on=common_column, how="left")
#Normalize the data
normalized_incidence = summary_incidence
normalized_incidence['cases'] = (summary_incidence['cases'] - summary_incidence['cases'].min()) / (summary_incidence['cases'].max() - summary_incidence['cases'].min())
normalized_incidence['deaths'] = (summary_incidence['deaths'] - summary_incidence['deaths'].min()) / (summary_incidence['deaths'].max() - summary_incidence['deaths'].min())
cases = normalized_incidence['cases']
print("Collected incidence")

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
    #Normalize data
    normalized_results = results_df
    normalized_results[band] = (results_df[band] - results_df[band].min()) / (results_df[band].max() - results_df[band].min())
    #Save the dataframe to csv to reduce the need to run this script connecting to GEE
    normalized_results.to_csv((str(band)+".csv"), index=False)
    print(normalized_results)
    return(normalized_results)

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

###Precipitation (GEE)
#Extract data from GEE
precipitation_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'total_precipitation_sum')
#After first GEE extraction, run this to only use the csv and no longer connect to GEE
#precipitation_df = pd.read_csv('total_precipitation_sum.csv')
#Add to indicator data bucket
hazard_indicators.append(precipitation_df)
print("Collected precipitation")

###Temperature (GEE)
#Extract data from GEE
temperature_df = gee_extraction("ECMWF/ERA5_LAND/DAILY_AGGR", 'skin_temperature')
#After first GEE extraction, run this to only use the csv and no longer connect to GEE
#temperature_df = pd.read_csv('skin_temperature.csv')
#Add to indicator data bucket
hazard_indicators.append(temperature_df)
print("Collected surface temperature")

###Poverty(CSV)
#Load data
wealth_df = pd.read_csv('Datasets_Directory/Relative_wealth_index.csv')
#Define point coordinates as geometry
geometry = [Point(xy) for xy in zip(wealth_df.longitude, wealth_df.latitude)]
#Format to geodataframe
wealth_gdf = gpd.GeoDataFrame(wealth_df, geometry=geometry)
wealth_gdf.crs = "EPSG:4326"
#Complete spaital join with admin boundaries shapefile
merged_wealth = gpd.sjoin(wealth_gdf, admin_boundaries, how="left", op="within")
#Group by admin boundaries name column
mean_wealth = merged_wealth.groupby(common_column)["rwi"].mean().reset_index()
#Normalize (the higher the rwi the greater the wealth, so multiply by -1 to convert to poverty)
scaled_poverty = mean_wealth
scaled_poverty['rwi'] = (((-1*mean_wealth['rwi']) + 1) / 2)
scaled_poverty.rename(columns={'rwi': 'Poverty'}, inplace=True)
poverty_df = copy_temporal_resolution(scaled_poverty)
vulnerability_indicators.append(poverty_df)
print("Collected poverty")

#Hazards (CSV)
disasters_df = pd.read_csv('Datasets_Directory/Climate-relatedDisasters.csv')
merged_disasters = admin_boundaries.merge(disasters_df, on=common_column, how="left")
normalized_disasters = merged_disasters[['ADM1_FR', 'Average']]
normalized_disasters['Average'] = (merged_disasters['Average'] - merged_disasters['Average'].min()) / (merged_disasters['Average'].max() - merged_disasters['Average'].min())
normalized_disasters.rename(columns={'Average': 'Hazards'}, inplace=True)
hazard_df = copy_temporal_resolution(normalized_disasters)
hazard_indicators.append(hazard_df)
print("Collected hazards")

#Conflicts (CSV)
input_conflicts_df = pd.read_csv('Datasets_Directory/Conflicts.csv')
merged_conflicts = admin_boundaries.merge(input_conflicts_df, on=common_column, how="left")
normalized_conflicts = merged_conflicts[['ADM1_FR', 'Conflicts']]
normalized_conflicts['Conflicts'] = (merged_conflicts['Conflicts'] - merged_conflicts['Conflicts'].min()) / (merged_conflicts['Conflicts'].max() - merged_conflicts['Conflicts'].min())
conflicts_df = copy_temporal_resolution(normalized_conflicts)
vulnerability_indicators.append(conflicts_df)
print("Collected conflicts")

#WASH (CSV)
WASH_df = pd.read_csv('Datasets_Directory/WASH.csv')
merged_WASH = admin_boundaries.merge(WASH_df, on=common_column, how="left")
normalized_WASH = merged_WASH[['ADM1_FR', 'Insufficient_WASH']]
normalized_WASH['Insufficient_WASH'] = (merged_WASH['Insufficient_WASH'] - merged_WASH['Insufficient_WASH'].min()) / (merged_WASH['Insufficient_WASH'].max() - merged_WASH['Insufficient_WASH'].min())
wash_df = copy_temporal_resolution(normalized_WASH)
capacity_indicators.append(wash_df)
print("Collected WASH")

#Health Care Facilities (CSV)
HCF_df = pd.read_csv('Datasets_Directory/HCF.csv')
merged_HCF = admin_boundaries.merge(HCF_df, on=common_column, how="left")
normalized_HCF = merged_HCF[['ADM1_FR', 'Pop_HCFs']]
normalized_HCF['Pop_HCFs'] = (merged_HCF['Pop_HCFs'] - merged_HCF['Pop_HCFs'].min()) / (merged_HCF['Pop_HCFs'].max() - merged_HCF['Pop_HCFs'].min())
hcf_df = copy_temporal_resolution(normalized_HCF)
capacity_indicators.append(hcf_df)
print("Collected HCF")

####################
#Merge lists of indicators
indicators = indicators + hazard_indicators + vulnerability_indicators + capacity_indicators
merged_df = reduce(lambda left, right: pd.merge(left, right, on=[common_column, 'start_date', 'end_date'], how='inner'), indicators)
merged_df_no_id = merged_df.drop(columns=[common_column, 'start_date', 'end_date'])
print(merged_df_no_id)
reg = LinearRegression()
print(cases)
res = reg.fit(merged_df_no_id,cases)
print(f"Regression coefficients: {reg.coef_}")

est = sm.OLS(cases, merged_df_no_id)
est2 = est.fit()
print(est2.summary())



