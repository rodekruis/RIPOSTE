import geopandas as gpd
import pandas as pd
import os
import numpy as np
import rasterio
from rasterio.mask import mask
import distancerasters as dr

########################## Definitions ########################
# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/510/Anticipatory Action/RIPOSTE/Measles DRC/Data')
# Set spatial resolution and set the common column to merge the data spatially
zone = "Zone"
province = "Province"
# Set temporal resolution
temporal = "static"
# Load admin boundaries
# Administrative level SHP file
admin_shp_path = "Els/RDC_Zones de santé.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
full_admin_boundaries.rename(columns={'Nom': zone, 'PROVINCE': province},inplace=True)  # Fix the naming of columns
admin_boundaries = full_admin_boundaries[[zone, province, 'geometry']]
admin_boundaries[zone] = admin_boundaries[zone].str.upper()
admin_boundaries[province] = admin_boundaries[province].str.upper()
#Load data
master_df = pd.read_csv('complete_dataset_df_'+temporal+'.csv')
index_columns = ['start_date', 'end_date', zone, province]
# Accessibility to HCFs
hcfs = gpd.read_file('DRC_hcf.shp')
pop_density = 'Datasets_Directory/cod_pd_2020_1km.tif'
print("Loaded data")


## Create straight-line distance raster to the closest HCF
# resolution (in units matching projection) at which vector data will be rasterized
pixel_size = 0.01

# rasterize vector data and output to geotiff
rv_array, affine = dr.rasterize(hcfs, pixel_size=pixel_size, bounds=admin_boundaries.total_bounds, output="C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/510/Anticipatory Action/RIPOSTE/Measles DRC/Data/hcf_rasterized.tif")

def raster_conditional(rarray):
    return (rarray == 1)

# generate distance array and output to geotiff
my_dr = dr.DistanceRaster(rv_array, affine=affine, output_path="C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/510/Anticipatory Action/RIPOSTE/Measles DRC/Data/distance_raster.tif", conditional=raster_conditional)

## Create reporting probability raster
# decay in the probability of reporting a disease case (larger distance, larger g): g(d) = exp(14.77 + 9.17d)
# Open the input raster
with rasterio.open('distance_raster.tif') as src:
    # Read the raster data and divide the meter distance values by 100000 to get values per 100km
    data = (src.read(1))/1000
    # Get metadata from the source raster
    meta = src.meta

# Update metadata to reflect the changes
meta.update({'count': 1})

# Create a new raster with the modified data
with rasterio.open('decay_raster.tif', 'w', **meta) as dst:
    # Write the modified data to the new raster
    dst.write(data, 1)

### Get average decay for every region
src = rasterio.open('decay_raster.tif')
average_decay = []
for index, row in admin_boundaries.iterrows():
    geom = row['geometry']
    out_image, out_transform = mask(src, [geom], crop=True)
    mean_value = np.nanmean(out_image)
    average_decay.append({zone: row[zone], province: row[province], 'Average_Decay': mean_value})
    decay = pd.DataFrame(average_decay)

### Normalize to between 0.95 and 0.05 - might need to be adjjsted as distances are not that big so undereporting might not be as severe and differences between areas is definitely not as severe
# Get the minimum and maximum values in the raster
min_value = decay['Average_Decay'].min()
max_value = decay['Average_Decay'].max()
# Normalize the raster values to the range [0.05, 0.95]
decay['Probability_underreporting'] = 0.05 + 0.9 * ((decay['Average_Decay'] - min_value) / (max_value - min_value))
print(decay)

### Correct regional incidence data
master_df_underreporting = master_df.merge(decay, on=[zone, province], how="left")
print(master_df_underreporting)
print(master_df[[zone, province, 'cases', 'deaths', 'Attack Rate']])
master_df_underreporting['cases'] = (master_df_underreporting['cases']/(1-master_df_underreporting['Probability_underreporting']))*1
master_df_underreporting['deaths'] = (master_df_underreporting['deaths']/(1-master_df_underreporting['Probability_underreporting']))*1
master_df_underreporting['Attack Rate'] = (master_df_underreporting['Attack Rate']/(1-master_df_underreporting['Probability_underreporting']))*1
print(master_df_underreporting[[zone, province, 'cases', 'deaths', 'Attack Rate']])

master_df_underreporting.to_csv('underreporting_complete_dataset_df_'+temporal+'.csv', index=False)