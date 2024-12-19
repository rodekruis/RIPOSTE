import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import cdsapi
from rasterstats import zonal_stats
import rasterio
import geopandas as gpd

def CDS_extraction(imageset, band):
    client = cdsapi.Client()
    results = []

    # Simplify geometry to reduce computation size
    simplified_admin_boundaries = admin_boundaries.copy()
    simplified_admin_boundaries['geometry'] = simplified_admin_boundaries['geometry'].simplify(tolerance=0.05, preserve_topology=True)

    for index, row in time_periods.iterrows():
        start_date = row['start_date']
        end_date = row['end_date']
        print(f"Processing period from {start_date} to {end_date}")

        # Loop through each day within the month
        current_date = start_date
        while current_date <= end_date:
            # # Request data for the current day
            # request = {
            #     "variable": [band],
            #     "year": [current_date.year],
            #     "month": [f'{current_date.month:02d}'],
            #     "day": [f'{current_date.day:02d}'],
            #     "time": [f'{hour:02d}:00' for hour in range(24)],  # Request all 24 hours of the day
            #     "area": [13.5, 8, 1, 17],  # Bounding box for Cameroon
            #     "format": "grib"
            # }
            #
            # # Define the path for the GRIB file
            grib_foldername = os.path.join(os.getcwd(), "CDS_grib_files_daily")
            # if not os.path.exists(grib_foldername):
            #     os.makedirs(grib_foldername)
            grib_filename = os.path.join(grib_foldername, f"{band}_{current_date.year}_{current_date.month:02d}_{current_date.day:02d}.grib")
            #
            # # Retrieve data
            # client.retrieve(imageset, request).download(grib_filename)
            # print(f"Downloaded GRIB file for {current_date}")

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
                    results.append({
                        'date': current_date,
                        'DISTRICT_S': district_name,
                        band: stat['mean']  # Use mean as the aggregation type
                    })

            # Move to the next day
            current_date += timedelta(days=1)

    # Save the dataframe to CSV after completing all daily extractions
    results_df = pd.DataFrame(results)
    results_df.to_excel(f"{band}_{temporal}_DISTRICT.xlsx", index=False, float_format='%.8f')
    print(results_df)
    return results_df

########################## Definitions ########################
# Set working directory
os.chdir('C:/Users/mdroogleverfortuyn/Rode Kruis/510 - Anticipatory Action - Documents/[PRJ] RIPOSTE/French Red Cross AFD RIPOSTE/Cholera Cameroon/Data/Python Data/Precipitation')

# Set common column to merge the data spatially and temporally
common_column = "DISTRICT_S"

# Set temporal resolution and study period
temporal = "daily"
study_start = datetime(2021, 10, 1)
study_end = datetime(2023, 12, 31)

# Create time periods for every day between the start and end date
start_dates = pd.date_range(study_start, study_end, freq='D')
data = {'start_date': start_dates, 'end_date': start_dates}  # For daily processing, start_date and end_date are the same
time_periods = pd.DataFrame(data)

# Administrative level SHP file
admin_shp_path = "District_sante_2022.shp"
full_admin_boundaries = gpd.read_file(admin_shp_path)
admin_boundaries = full_admin_boundaries[[common_column, 'geometry', 'POPULATION']]
print("Loaded admin boundaries")

### Precipitation (CDS)
# Extract daily data from CDS
precipitation_df = CDS_extraction("reanalysis-era5-land", 'total_precipitation')
print("Printed precipitation data")
