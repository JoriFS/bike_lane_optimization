import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the pivot table (make sure you have pyarrow or fastparquet installed)
df_2d = pd.read_parquet("data/processed/travelMatrix_sum.parquet")  
df_distance_2d = pd.read_parquet("data/processed/travelMatrix_distanceMeters.parquet")

# Plot the matrix as a heatmap
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(df_2d.values, cmap="viridis", vmax=1)

# Add colorbar
# plt.colorbar(cax)

# Set axis labels (optional - slow if too many labels)
#ax.set_xticks(range(len(df_2d.columns)))
#ax.set_yticks(range(len(df_2d.index)))
# ax.set_xticklabels(df_2d.columns, rotation=90, fontsize=6)
# ax.set_yticklabels(df_2d.index, fontsize=6)

ax.set_title("Travel Frequency Matrix (on_stop_id → off_stop_id), yellow means more than one trip, dark mean zero trips", pad=20)
plt.tight_layout()
plt.show()

# PLOT HISTOGRAM OF NUMBER OF TRIPS PER STATION
row_sums = df_2d.sum(axis=1).to_numpy()

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(row_sums, bins=60, color='skyblue', edgecolor='black', range=(0, 30000))

plt.title("Histogram of station usage")
plt.xlabel("# of trips ending or starting in a station")
plt.ylabel("# of stations")
plt.grid(True)
plt.tight_layout()
plt.show()


# PLOT SCATTERPLOT OF TRIP LENGTH VS # OF TRIPS
# Flatten both pivot tables into 1D arrays
x = df_distance_2d.values.flatten()
y = df_2d.values.flatten()

# Optional: remove NaNs or invalid (e.g., zero-trip) pairs
mask = ~np.isnan(x) & ~np.isnan(y) & (y > 0)
x_clean = x[mask]
y_clean = y[mask]

# Plot scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(x_clean, y_clean, alpha=0.6)

plt.xlabel("Distance (meters)")
plt.ylabel("Number of Trips")
plt.title("Trips vs. Distance Scatterplot")
plt.grid(True)
plt.tight_layout()
plt.show()


# PLOT HEATMAP DISTANCE VS TRIP FREQUENCY
# Remove NaNs and zero/negative values (log scale can't handle them)
mask = (
    ~np.isnan(x) & ~np.isnan(y) & 
    (x > 0) & (x <= 2000) & 
    (y > 0) & (y <= 2000)
)

x_clean = x[mask]
y_clean = y[mask]

# Plot 2D histogram
plt.figure(figsize=(8, 6))
hist = plt.hist2d(
    x_clean, y_clean,
    bins=50,
    range=[[0, 2000], [0, 2000]],
    cmap="viridis",
    norm='log'  # <- log scale
)

plt.colorbar(label="log-scaled count")
plt.xlabel("Distance (meters)")
plt.ylabel("Number of Trips")
plt.title("2D Histogram (Log Scale): Trips vs. Distance")
plt.tight_layout()
plt.show()