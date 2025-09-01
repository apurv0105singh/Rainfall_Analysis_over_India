import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ==============================
# Step 1: Load rainfall data files
# ==============================
files = [
    f"C:/Users/DELL/Desktop/PROJECTS/Rainfall_Analysis_over_India/lab5/RF25_ind{year}_rfp25.nc"
    for year in range(2010, 2021)
]

# Open all files concatenating along TIME
# This handles leap years and variable TIME lengths

ds_all = xr.open_mfdataset(
    files,
    combine="by_coords",
    parallel=True,
    chunks={"TIME": 100}
)
print("✅ Files loaded successfully!")

# ==============================
# Step 2: Extract rainfall at a grid point
# ==============================
lat_value, lon_value = 25, 80.5
rain_point = ds_all["RAINFALL"].sel(LATITUDE=lat_value, LONGITUDE=lon_value, method="nearest")

# ==============================
# Step 3: Calculate Annual, Monsoon, Annual Max Daily Rainfall (Time Series)
# ==============================
years = list(range(2010, 2021))
annual_rainfall = [rain_point.sel(TIME=str(year)).sum().compute().item() for year in years]
monsoon_rainfall = [
    rain_point.sel(TIME=slice(f"{year}-06-01", f"{year}-09-30")).sum().compute().item()
    for year in years
]
annual_max_rainfall = [rain_point.sel(TIME=str(year)).max().compute().item() for year in years]

# ==============================
# Step 4: Plot Time Series
# ==============================
fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
plots = [
    (annual_rainfall, "Annual Rainfall (mm)", "Annual Rainfall Time Series", "b"),
    (monsoon_rainfall, "Monsoon Rainfall (mm)", "Monsoon (JJAS) Rainfall Time Series", "g"),
    (annual_max_rainfall, "Max Daily Rainfall (mm)", "Annual Maximum Daily Rainfall", "r"),
]
for ax, (data, ylabel, title, color) in zip(axes, plots):
    ax.plot(years, data, marker='o', linestyle='-', color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.show()

# ==============================
# Step 5: Statistics (Mean, Std, CV)
# ==============================
def compute_stats(data, label):
    mean = np.mean(data)
    std = np.std(data)
    cv = (std / mean) * 100 if mean != 0 else np.nan
    print(f"Results for {label}:")
    print(f"  Mean = {mean:.2f}")
    print(f"  Std Dev = {std:.2f}")
    print(f"  CV = {cv:.2f}%\n")

compute_stats(annual_rainfall, "Annual Rainfall")
compute_stats(monsoon_rainfall, "Monsoon Rainfall")
compute_stats(annual_max_rainfall, "Max Daily Rainfall")

# ==============================
# Step 6: Trend Analysis
# ==============================
def trend_analysis(data, years, label, color):
    slope, intercept, r_value, p_value, std_err = linregress(years, data)
    print(f"Trend in {label}: {slope:.2f} units/year (p={p_value:.3f})")

    plt.figure(figsize=(10, 5))
    plt.plot(years, data, marker='o', linestyle='-', color=color, label=label)
    plt.plot(years, intercept + slope*np.array(years), '--', color="black", label="Trend line")
    plt.xlabel("Year")
    plt.ylabel(label)
    plt.title(f"Trend in {label} (slope={slope:.2f}, p={p_value:.3f})")
    plt.legend()
    plt.show()

trend_analysis(annual_rainfall, years, "Annual Rainfall (mm)", "b")
trend_analysis(monsoon_rainfall, years, "Monsoon Rainfall (mm)", "g")
trend_analysis(annual_max_rainfall, years, "Max Daily Rainfall (mm)", "r")

# ==============================
# Step 7: Anomaly Detection
# ==============================
def anomaly_plot(data, years, label):
    long_term_mean = np.mean(data)
    anomalies = [val - long_term_mean for val in data]
    plt.figure(figsize=(10, 5))
    plt.bar(years, anomalies, color=["red" if a < 0 else "blue" for a in anomalies])
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Anomaly (mm)")
    plt.title(f"{label} Anomalies (relative to mean {long_term_mean:.2f})")
    plt.show()

anomaly_plot(annual_rainfall, years, "Annual Rainfall")
anomaly_plot(monsoon_rainfall, years, "Monsoon Rainfall")
anomaly_plot(annual_max_rainfall, years, "Max Daily Rainfall")

# ==============================
# Step 8: Spatial Analysis (Mean, Std, CV Maps)
# ==============================
annual = ds_all["RAINFALL"].groupby("TIME.year").sum(dim="TIME")
monsoon = ds_all["RAINFALL"].sel(TIME=ds_all.TIME.dt.month.isin([6,7,8,9])).groupby("TIME.year").sum(dim="TIME")
daily_max = ds_all["RAINFALL"].groupby("TIME.year").max(dim="TIME")

annual_mean, annual_std = annual.mean("year"), annual.std("year")
monsoon_mean, monsoon_std = monsoon.mean("year"), monsoon.std("year")
daily_max_mean, daily_max_std = daily_max.mean("year"), daily_max.std("year")

annual_cv  = (annual_std / annual_mean).where(annual_mean > 0)
monsoon_cv = (monsoon_std / monsoon_mean).where(monsoon_mean > 0)
daily_max_cv = (daily_max_std / daily_max_mean).where(daily_max_mean > 0)

# Load India shapefile
india_shapefile = gpd.read_file("C:/Users/DELL/Desktop/PROJECTS/Rainfall_Analysis_over_India/india_st.shp")

def plot_field(ax, data, title):
    pcm = ax.pcolormesh(data.LONGITUDE, data.LATITUDE, data.compute(), cmap="terrain", shading="auto")
    india_shapefile.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(pcm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, label="mm")

# 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
plot_field(axes[0,0], annual_mean, "Annual Rainfall Mean")
plot_field(axes[0,1], annual_std,  "Annual Rainfall Std Dev")
plot_field(axes[0,2], annual_cv,   "Annual Rainfall CV")
plot_field(axes[1,0], monsoon_mean, "Monsoon Rainfall Mean")
plot_field(axes[1,1], monsoon_std,  "Monsoon Rainfall Std Dev")
plot_field(axes[1,2], monsoon_cv,   "Monsoon Rainfall CV")
plot_field(axes[2,0], daily_max_mean, "Daily Annual Max Rainfall Mean")
plot_field(axes[2,1], daily_max_std,  "Daily Annual Max Rainfall Std Dev")
plot_field(axes[2,2], daily_max_cv,   "Daily Annual Max Rainfall CV")

plt.tight_layout()
plt.savefig("Rainfall_Analysis_3x3.png", dpi=300, bbox_inches="tight")
plt.show()