import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio import features
from rasterio.plot import show
from rasterio.mask import mask
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point
import seaborn as sns
import statsmodels.api as sm
import momepy
from momepy.datasets import get_path
import geopandas as gpd
import networkx as nx
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, mapping
from shapely.ops import cascaded_union
from shapely.ops import nearest_points
import time
import multiprocessing
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

########################## Definitions ########################
#set wd
os.chdir('C:/Users/mdroogleverfortuyn/OneDrive - Rode Kruis/Documenten/Anticipatory Action/RIPOSTE/Measles DRC/Data')

#set variables
# Set spatial resolution and set the common column to merge the data spatially
common_column = "ADM2_FR"

# Create master dataframe to store all the datasets
index_columns = ['start_date', 'end_date', common_column]

# Set temporal resolution
temporal = "yearly"

#load admin boundaries
# Administrative level SHP file
admin_shp_path = "Administrative Boundaries\cod_admbnda_adm2_rgc_20190911.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column, 'Shape_Leng', 'Shape_Area', 'geometry']]
# Make all admin names upper case to match other dataset inputs
admin_boundaries[common_column] = admin_boundaries[common_column].str.upper()
print("Loaded admin boundaries")

#Load data
master_df = pd.read_csv('complete_dataset_df_'+temporal+'.csv')

################################ Under reporting #############################
### Load data
hcfs = pd.read_csv('Datasets_Directory/sub-saharan_health_facilities.csv')
road_network = gpd.read_file('edges_small.shp')
road_network.crs = "EPSG:32734"
road_network = road_network.to_crs(epsg="4326")
pop_density = rasterio.open('Datasets_Directory/cod_pd_2020_1km.tif')
print("Loaded data")

# Structure data
# Define point coordinates as geometry
geometry = [Point(xy) for xy in zip(hcfs.Long, hcfs.Lat)]
# Format to geodataframe
hcf_gdf = gpd.GeoDataFrame(hcfs, geometry=geometry, crs="EPSG:4326")
# Complete spatial join with admin boundaries shapefile
merged_HCF = hcf_gdf.clip(admin_boundaries)
merged_HCF = gpd.sjoin(merged_HCF, admin_boundaries, how="left", op="within")
merged_HCF.to_file("DRC_hcf.shp")
print("Structured data")

# ### Connect HCFs to network graph
# shply_line = cascaded_union(road_network.geometry)
# snapped_hcf = merged_HCF.copy()
# print(snapped_hcf.crs)
# count = 0
# for i in snapped_hcf.index:
#     count = count + 1
#     print("Snapping "+str(count))
#     point = snapped_hcf.loc[i, 'geometry']
#     snapped_hcf.at[i, 'geometry'] = shply_line.interpolate(shply_line.project(point))
# snapped_hcf.to_file("snapped_hcf.shp")
# print(snapped_hcf)

### Create impedance raster
# # Specify pixel size and raster extent based on admin_boundaries
# geom = [shapes for shapes in road_network.geometry]
# rasterized = features.rasterize(geom,
#                                 out_shape = pop_density.shape,
#                                 fill = 0.9*255,
#                                 out = None,
#                                 transform = pop_density.transform,
#                                 all_touched = True,
#                                 default_value = 0.1*255,
#                                 dtype = None)
# # Plot raster
# fig, ax = plt.subplots(1, figsize = (10, 10))
# show(rasterized, ax = ax)
# plt.gca().invert_yaxis()
# plt.show()
#
# print(rasterized.dtype)
#
# with rasterio.open(
#         "impedance_raster.tif", "w",
#         driver = "GTiff",
#         crs = pop_density.crs,
#         transform = pop_density.transform,
#         dtype = rasterio.uint8,
#         count = 1,
#         width = pop_density.width,
#         height = pop_density.height) as dst:
#     dst.write(rasterized, indexes = 1)

### Create shortest path impedance raster
import numpy as np
from scipy.ndimage import distance_transform_cdt
import rasterio
from rasterio.transform import from_origin

# Load the impedance raster
impedance_raster_path = "impedance_raster.tif"
with rasterio.open(impedance_raster_path) as impedance_raster:
    impedance_data = impedance_raster.read(1)
    impedance_transform = impedance_raster.transform
print("Loaded impedance raster")

# Load the health care facility points
hcf_rasterized = features.rasterize(
    [(Point(xy[0], xy[1]), facility_id) for facility_id, xy in enumerate(zip(merged_HCF.geometry.x, merged_HCF.geometry.y), start=1)],
    out_shape=impedance_data.shape,
    transform=impedance_transform,
    fill=0,
    dtype=rasterio.uint16
)
print("Loaded HCF points")
with rasterio.open("rasterized_HCFs.tif", "w", driver="GTiff", crs=impedance_raster.crs,
                   transform=impedance_transform, dtype=rasterio.float32, count=1,
                   width=impedance_data.shape[1], height=impedance_data.shape[0]) as output_raster:
    output_raster.write(hcf_rasterized, indexes = 1)

# Create an array to store the least cost distances to the nearest HCF
least_cost_distance = np.zeros_like(impedance_data, dtype=np.float32)

# Iterate through each health care facility point
for facility_id in np.unique(hcf_rasterized[hcf_rasterized > 0]):
    print(facility_id)
    # Create a binary mask for the current health care facility
    hcf_mask = (hcf_rasterized == facility_id).astype(np.uint8)

    # Calculate the distance transform with cumulative cost
    distance_to_hcf = distance_transform_cdt(hcf_mask, metric="taxicab")

    # Update the least cost distance array with the minimum distances
    least_cost_distance = np.minimum(least_cost_distance, distance_to_hcf)

print("Saving...")
# Save the least cost raster for the entire region
with rasterio.open("least_cost_raster.tif", "w", driver="GTiff", crs=impedance_raster.crs,
                   transform=impedance_transform, dtype=rasterio.float32, count=1,
                   width=impedance_data.shape[1], height=impedance_data.shape[0]) as output_raster:
    output_raster.write(least_cost_distance, 1)


# ### Create reporting probability raster
# # decay in the probability of reporting a disease case (larger distance, larger g): g(d) = exp(14.77 + 9.17d)
# # Open the input raster
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
#
# ### Get average decay for every region
# src = rasterio.open('decay_raster.tif')
# average_decay = []
# for index, row in admin_boundaries.iterrows():
#     geom = row['geometry']
#     out_image, out_transform = mask(src, [geom], crop=True)
#     mean_value = np.nanmean(out_image)
#     average_decay.append({common_column: row['ADM1_FR'], 'Average_Decay': mean_value})
#     decay = pd.DataFrame(average_decay)
#
# ### Normalize to between 0.95 and 0.05
# # Get the minimum and maximum values in the raster
# min_value = decay['Average_Decay'].min()
# max_value = decay['Average_Decay'].max()
# # Normalize the raster values to the range [0.05, 0.95]
# decay['Probability_underreporting'] = 0.05 + 0.9 * ((decay['Average_Decay'] - min_value) / (max_value - min_value))
# print(decay)