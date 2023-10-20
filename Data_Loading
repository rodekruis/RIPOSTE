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

os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')

# Administrative SHP file
adminshapefile_path = "Administrative Boundaries\cmr_admbnda_inc_20180104_SHP\cmr_admbnda_adm1_inc_20180104.shp"
admin_boundaries = gpd.read_file(adminshapefile_path)
print("Loaded admin boundaries")

# Use the common column to merge the data
common_column = "ADM1_FR"

# Initialize empty lists to store merged datasets
hazard_indicators = []
vulnerability_indicators = []
capacity_indicators = []
indicators = []

# Incidence data
incidence_df = pd.read_csv('Regional_Incidence.csv')
#incidence_df['start_date'] = pd.to_datetime(incidence_df['start_date'])
#incidence_df['end_date'] = pd.to_datetime(incidence_df['end_date'])
merged_incidence = admin_boundaries.merge(incidence_df, on=common_column, how="left")
summary_incidence = merged_incidence.groupby(common_column).agg({
    'cases': 'sum',
    'deaths': 'sum',
}).reset_index()
normalized_incidence = summary_incidence
normalized_incidence['cases'] = (summary_incidence['cases'] - summary_incidence['cases'].min()) / (summary_incidence['cases'].max() - summary_incidence['cases'].min())
normalized_incidence['deaths'] = (summary_incidence['deaths'] - summary_incidence['deaths'].min()) / (summary_incidence['deaths'].max() - summary_incidence['deaths'].min())
cases = normalized_incidence['cases']
print("Collect incidence")


#Hazards (CSV)
disasters_df = pd.read_csv('Datasets_Directory/Climate-relatedDisasters.csv')
merged_disasters = admin_boundaries.merge(disasters_df, on=common_column, how="left")
normalized_disasters = merged_disasters[['ADM1_FR', 'Average']]
normalized_disasters['Average'] = (merged_disasters['Average'] - merged_disasters['Average'].min()) / (merged_disasters['Average'].max() - merged_disasters['Average'].min())
normalized_disasters.rename(columns={'Average': 'Hazards'}, inplace=True)
hazard_indicators.append(normalized_disasters)
print("Collected hazards")

#Precipitation (TIFF)
src = rasterio.open('Datasets_Directory/Precipitation.tif')
mean_precipitation = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    mean_value = np.nanmean(out_image)
    mean_precipitation.append({common_column: row['ADM1_FR'], 'Mean': mean_value})
    mean_precipitation_df = pd.DataFrame(mean_precipitation)
normalized_precipitation = mean_precipitation_df
normalized_precipitation['Mean'] = (mean_precipitation_df['Mean'] - mean_precipitation_df['Mean'].min()) / (mean_precipitation_df['Mean'].max() - mean_precipitation_df['Mean'].min())
normalized_precipitation.rename(columns={'Mean': 'Precipitation'}, inplace=True)
hazard_indicators.append(normalized_precipitation)
print("Collected precipitation")

#Temperature (TIFF)
src = rasterio.open('Datasets_Directory/Temperature.tif')
mean_temperature = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    mean_value = np.nanmean(out_image)
    mean_temperature.append({common_column: row['ADM1_FR'], 'Mean': mean_value})
    mean_temperature_df = pd.DataFrame(mean_temperature)
normalized_temperature = mean_temperature_df
normalized_temperature['Mean'] = (mean_temperature_df['Mean'] - mean_temperature_df['Mean'].min()) / (mean_temperature_df['Mean'].max() - mean_temperature_df['Mean'].min())
normalized_temperature.rename(columns={'Mean': 'Temperature'}, inplace=True)
hazard_indicators.append(normalized_temperature)
print("Collected temperature")

#Poverty(CSV)
wealth_df = pd.read_csv('Datasets_Directory/Relative_wealth_index.csv')
geometry = [Point(xy) for xy in zip(wealth_df.longitude, wealth_df.latitude)]
wealth_gdf = gpd.GeoDataFrame(wealth_df, geometry=geometry)
wealth_gdf.crs = "EPSG:4326"
merged_wealth = gpd.sjoin(wealth_gdf, admin_boundaries, how="left", op="within")
mean_wealth = merged_wealth.groupby(common_column)["rwi"].mean().reset_index()
scaled_poverty = mean_wealth
scaled_poverty['rwi'] = -1*((mean_wealth['rwi'] + 1) / 2)
scaled_poverty.rename(columns={'rwi': 'Poverty'}, inplace=True)
vulnerability_indicators.append(scaled_poverty)
print("Collected poverty")

#Conflicts (CSV)
conflicts_df = pd.read_csv('Datasets_Directory/Conflicts.csv')
merged_conflicts = admin_boundaries.merge(conflicts_df, on=common_column, how="left")
normalized_conflicts = merged_conflicts[['ADM1_FR', 'Conflicts']]
normalized_conflicts['Conflicts'] = (merged_conflicts['Conflicts'] - merged_conflicts['Conflicts'].min()) / (merged_conflicts['Conflicts'].max() - merged_conflicts['Conflicts'].min())
vulnerability_indicators.append(normalized_conflicts)
print("Collected conflicts")

#WASH (CSV)
WASH_df = pd.read_csv('Datasets_Directory/WASH.csv')
merged_WASH = admin_boundaries.merge(WASH_df, on=common_column, how="left")
normalized_WASH = merged_WASH[['ADM1_FR', 'Insufficient_WASH']]
normalized_WASH['Insufficient_WASH'] = (merged_WASH['Insufficient_WASH'] - merged_WASH['Insufficient_WASH'].min()) / (merged_WASH['Insufficient_WASH'].max() - merged_WASH['Insufficient_WASH'].min())
capacity_indicators.append(normalized_WASH)
print("Collected WASH")

#Health Care Facilities (CSV)
HCF_df = pd.read_csv('Datasets_Directory/HCF.csv')
merged_HCF = admin_boundaries.merge(HCF_df, on=common_column, how="left")
normalized_HCF = merged_HCF[['ADM1_FR', 'Pop_HCFs']]
normalized_HCF['Pop_HCFs'] = (merged_HCF['Pop_HCFs'] - merged_HCF['Pop_HCFs'].min()) / (merged_HCF['Pop_HCFs'].max() - merged_HCF['Pop_HCFs'].min())
capacity_indicators.append(normalized_HCF)
print("Collected HCF")



####################
#Merge lists of indicators
indicators = indicators + hazard_indicators + vulnerability_indicators + capacity_indicators
merged_df = reduce(lambda left, right: pd.merge(left, right, on=common_column, how='inner'), indicators)
merged_df_no_id = merged_df.drop(columns=[common_column])
print(merged_df_no_id)
reg = LinearRegression()
print(cases)
res = reg.fit(merged_df_no_id,cases)
print(f"Regression coefficients: {reg.coef_}")

est = sm.OLS(cases, merged_df_no_id)
est2 = est.fit()
print(est2.summary())



