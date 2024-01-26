import geopandas as gpd
import pandas as pd
import os
import numpy as np
import rasterio
from rasterio.mask import mask
import distancerasters as dr

########################## Definitions ########################
# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data')
# Set spatial resolution and set the common column to merge the data spatially
common_column = "ADM1_FR"
# Set temporal resolution
temporal = "monthly"
# Load admin boundaries
# Administrative level SHP file
admin_shp_path = "Administrative Boundaries\cmr_admbnda_inc_20180104_SHP\cmr_admbnda_adm1_inc_20180104.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
#Load data
master_df = pd.read_csv('complete_dataset_df_'+temporal+'.csv')
index_columns = ['start_date', 'end_date', common_column]
# Accessibility to HCFs
hcfs = gpd.read_file('Datasets_Directory/hcfs_noconflict.shp')
roads = gpd.read_file('Datasets_Directory/hotosm_cmr_roads_lines.shp')
pop_density = 'Datasets_Directory/cmr_density_2020.tif'
print("Loaded data")

# ### Create straight-line distance raster to the closest HCF
# # resolution (in units matching projection) at which vector data will be rasterized
# pixel_size = 0.01
#
# # rasterize vector data and output to geotiff
# rv_array, affine = dr.rasterize(hcfs, pixel_size=pixel_size, bounds=admin_boundaries.total_bounds, output="C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data/hcf_rasterized.tif")
#
# def raster_conditional(rarray):
#     return (rarray == 1)
#
# # generate distance array and output to geotiff
# my_dr = dr.DistanceRaster(rv_array, affine=affine, output_path="C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Cholera Cameroon/Data/distance_raster.tif", conditional=raster_conditional)

### Create reporting probability raster
# decay in the probability of reporting a disease case (larger distance, larger g): g(d) = exp(14.77 + 9.17d)
# Open the input raster
# with rasterio.open('distance_raster.tif') as src:
#     # Read the raster data and divide the meter distance values by 100000 to get values per 100km
#     data = (src.read(1))/1000
#     # Apply decay function (**** Causes error because exponential creates too big numbers)
#     # data = np.exp(14.77 + (9.17 * data))
#     # Get metadata from the source raster
#     meta = src.meta
#
# # Update metadata to reflect the changes
# meta.update({'count': 1})
#
# # Create a new raster with the modified data
# with rasterio.open('decay_raster.tif', 'w', **meta) as dst:
#     # Write the modified data to the new raster
#     dst.write(data, 1)

### Get average decay for every region
src = rasterio.open('decay_raster.tif')
average_decay = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    mean_value = np.nanmean(out_image)
    average_decay.append({common_column: row['ADM1_FR'], 'Average_Decay': mean_value})
    decay = pd.DataFrame(average_decay)

### Normalize to between 0.95 and 0.05
# Get the minimum and maximum values in the raster
min_value = decay['Average_Decay'].min()
max_value = decay['Average_Decay'].max()
# Normalize the raster values to the range [0.05, 0.95]
decay['Probability_underreporting'] = 0.05 + 0.9 * ((decay['Average_Decay'] - min_value) / (max_value - min_value))
print(decay)

### Correct regional incidence data
master_df_underreporting = master_df.merge(decay, on=common_column, how="left")
print(master_df_underreporting)
print(master_df[[common_column, 'cases', 'deaths', 'Attack Rate']])
master_df_underreporting['cases'] = (master_df_underreporting['cases']/(1-master_df_underreporting['Probability_underreporting']))*1
master_df_underreporting['deaths'] = (master_df_underreporting['deaths']/(1-master_df_underreporting['Probability_underreporting']))*1
master_df_underreporting['Attack Rate'] = (master_df_underreporting['Attack Rate']/(1-master_df_underreporting['Probability_underreporting']))*1
print(master_df_underreporting[[common_column, 'cases', 'deaths', 'Attack Rate']])

master_df_underreporting.to_csv('underreporting_complete_dataset_df_'+temporal+'.csv', index=False)