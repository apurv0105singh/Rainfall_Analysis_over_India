# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 13:59:29 2025

@author: DELL
"""

# ==============================
# Rainfall Analysis over India (Beginner-friendly)
# ==============================

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pymannkendall as mk   # for trend test

# ------------------------------
# Step 1: Load rainfall data
# ------------------------------
files = []
for year in range(2010, 2021):
    file_path = f"C:/Users/DELL/Desktop/PROJECTS/Rainfall_Analysis_over_India/lab5/RF25_ind{year}_rfp25.nc"
    files.append(file_path)

# Open all files together
ds = xr.open_mfdataset(files, combine="by_coords")
print("✅ Rainfall data loaded successfully!")

rain = ds["RAINFALL"]

# ------------------------------
# Step 2: Choose grid point
# ------------------------------
lat, lon = 25, 80.5
rain_point = rain.sel(LATITUDE=lat, LONGITUDE=lon, method="nearest")

# ------------------------------
# Step 3: Monthly climatology (average per month)
# ------------------------------
monthly_mean = rain_point.groupby("TIME.month").mean().compute()

plt.figure(figsize=(8,5))
monthly_mean.plot(marker="o")
plt.title("Average Monthly Rainfall at Grid Point")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.show()

# ------------------------------
# Step 4: Annual totals (sum of daily values each year)
# ------------------------------
annual_totals = rain_point.groupby("TIME.year").sum().compute()
annual_totals = annual_totals.values  # convert to numpy array

years = list(range(2010, 2021))

plt.figure(figsize=(8,5))
plt.plot(years, annual_totals, marker="o")
plt.title("Annual Rainfall at Grid Point")
plt.xlabel("Year")
plt.ylabel("Rainfall (mm)")
plt.show()

# ------------------------------
# Step 5: Simple extreme indices
# ------------------------------
def rainfall_indices(data, threshold=1.0):
    rainy_days = np.sum(data > threshold)  # days > 1 mm
    sdii = data.sum() / rainy_days if rainy_days > 0 else np.nan
    r95p = np.sum(data[data > np.percentile(data, 95)])  # very wet days
    max_5day = pd.Series(data).rolling(5).sum().max()   # max 5-day rainfall
    return rainy_days, sdii, r95p, max_5day

indices = {}
for year in years:
    data = rain_point.sel(TIME=str(year)).values
    indices[year] = rainfall_indices(data)

indices_df = pd.DataFrame.from_dict(indices, orient="index",
                                    columns=["RainyDays","SDII","R95p","Max5day"])
print("✅ Extreme rainfall indices calculated")
print(indices_df)

# ------------------------------
# Step 6: Trend analysis (Mann-Kendall)
# ------------------------------
def check_trend(series, label):
    result = mk.original_test(series)
    print(f"{label}: trend = {result.trend}, slope = {result.slope:.2f}, p = {result.p:.3f}")

check_trend(indices_df["RainyDays"], "Rainy Days")
check_trend(indices_df["SDII"], "SDII")
check_trend(indices_df["R95p"], "R95p")
check_trend(indices_df["Max5day"], "Max 5-day rainfall")

# ------------------------------
# Step 7: Heatmap (rainfall by year and month)
# ------------------------------
rain_df = rain_point.to_dataframe(name="rain").reset_index()
rain_df["year"] = rain_df["TIME"].dt.year
rain_df["month"] = rain_df["TIME"].dt.month

rain_matrix = rain_df.groupby(["year", "month"])["rain"].sum().unstack()

plt.figure(figsize=(10,6))
sns.heatmap(rain_matrix, cmap="Blues", cbar_kws={"label": "Rainfall (mm)"})
plt.title("Monthly Rainfall Heatmap")
plt.xlabel("Month")
plt.ylabel("Year")
plt.show()

# ------------------------------
# Step 8: Regional analysis (India states)
# ------------------------------
shapefile = "C:/Users/DELL/Desktop/PROJECTS/Rainfall_Analysis_over_India/india_st.shp"
india_states = gpd.read_file(shapefile)

# Match CRS (coordinate system)
india_states = india_states.set_crs("EPSG:4326", allow_override=True)

# Mean rainfall for each grid cell
rain_mean = rain.groupby("TIME.year").sum("TIME").mean("year").compute()
rain_mean_df = rain_mean.to_dataframe(name="rain").reset_index()

gdf = gpd.GeoDataFrame(rain_mean_df,
                       geometry=gpd.points_from_xy(rain_mean_df.LONGITUDE, rain_mean_df.LATITUDE),
                       crs="EPSG:4326")

# Join grid rainfall with states
joined = gpd.sjoin(gdf, india_states, how="inner", predicate="within")

print("Shapefile columns:", india_states.columns)

# Suppose state column is 'STATE'
regional_avg = joined.groupby("STATE")["rain"].mean().sort_values(ascending=False)

print("\n✅ Top 10 wettest states:\n", regional_avg.head(10))

regional_avg.plot(kind="bar", figsize=(12,5))
plt.title("Mean Annual Rainfall by State (2010-2020)")
plt.ylabel("Rainfall (mm)")
plt.show()
