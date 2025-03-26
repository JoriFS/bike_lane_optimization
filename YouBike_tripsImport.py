# import geopandas as gpd
# import re
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from shapely.geometry import LineString, Point
import json
import pandas as pd
import numpy as np
# from tqdm import tqdm


# Load your JSON data
# with open("data/YouBike raw/youBike_trips_202307_weekdays.json", "r", encoding="utf-8") as f:
#     ubike_data = json.load(f)


# * Load Processed ubike data
ubike_df = pd.read_excel("data/processed/ubike_stats_df.xlsx")
print(ubike_df)


# df_2d = ubike_df.pivot_table(
#     index="on_stop_id",
#     columns="off_stop_id",
#     values="sum_of_txn_times",
#     aggfunc="sum",     # or another aggregator if needed
#     fill_value=0       # fill missing pairs with 0 instead of NaN
# )

# # ignore trips from station x to itself (set to zero)
# common_ids = df_2d.index.intersection(df_2d.columns)
# df_2d.loc[common_ids, common_ids] = df_2d.loc[common_ids, common_ids].mask(np.eye(len(common_ids), dtype=bool), 0)

# # Remove all-zero rows
# df_2d = df_2d[(df_2d != 0).any(axis=1)]
# # Remove all-zero columns
# df_2d = df_2d.loc[:, (df_2d != 0).any(axis=0)]

# # save data in two formats (parquet fast and efficient to reimport later)
# df_2d.to_parquet("data/processed/travelMatrix_raw.parquet")
# df_2d.to_excel("data/processed/travelMatrix_raw.xlsx")

# print(df_2d)


# # Calculate the sum of trips between stations x_ij and x_ji and put it in both diagonals of the matrix
# # Create a copy to avoid modifying during iteration
# df_copy = df_2d.copy()

# # Iterate through all on-off pairs
# for on in df_2d.index:
#     for off in df_2d.columns:
#         # Check if reverse direction exists
#         if off in df_2d.index and on in df_2d.columns:
#             x_ij = df_2d.at[on, off]
#             x_ji = df_2d.at[off, on]
#             # Combine into x_ij and zero x_ji
#             df_copy.at[on, off] = x_ij + x_ji
#             df_copy.at[off, on] = x_ij + x_ji

# df_2d = df_copy
# print(df_2d)

# # save data in two formats (parquet fast and efficient to reimport later)
# df_2d.to_parquet("data/processed/travelMatrix_sum.parquet")
# df_2d.to_excel("data/processed/travelMatrix_sum.xlsx")


# Use 'first' if each pair occurs once, or you just want the first occurrence
pivot_origin_x = ubike_df.pivot_table(index="on_stop_id", columns="off_stop_id", values="origin_x", aggfunc='first')
pivot_origin_y = ubike_df.pivot_table(index="on_stop_id", columns="off_stop_id", values="origin_y", aggfunc='first')
pivot_dest_x   = ubike_df.pivot_table(index="on_stop_id", columns="off_stop_id", values="destination_x", aggfunc='first')
pivot_dest_y   = ubike_df.pivot_table(index="on_stop_id", columns="off_stop_id", values="destination_y", aggfunc='first')

pivot_dest_x.to_parquet("data/processed/travelMatrix_destinationX.parquet")
pivot_dest_y.to_parquet("data/processed/travelMatrix_destinationY.parquet")
pivot_origin_x.to_parquet("data/processed/travelMatrix_originX.parquet")
pivot_origin_y.to_parquet("data/processed/travelMatrix_originY.parquet")

# calculate distance between start and end point
# Extract values as NumPy arrays
lat1 = np.radians(pivot_origin_y.values)
lon1 = np.radians(pivot_origin_x.values)
lat2 = np.radians(pivot_dest_y.values)
lon2 = np.radians(pivot_dest_x.values)

# Haversine formula (vectorized)
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
c = 2 * np.arcsin(np.sqrt(a))
R = 6371000  # Earth radius in meters

distance_matrix = R * c

# Convert back to a DataFrame with the same index and columns
pivot_distance_m = pd.DataFrame(distance_matrix,
                                index=pivot_origin_x.index,
                                columns=pivot_origin_x.columns)

pivot_distance_m.to_parquet("data/processed/travelMatrix_distanceMeters.parquet")


demands_origin = ubike_df.groupby('on_stop').agg({"origin_x": "mean", "origin_y": "mean", "sum_of_txn_times": "sum"}).reset_index().rename(columns = {"on_stop": "station", "sum_of_txn_times": "demand_origin"})
demands_destination = ubike_df.groupby('off_stop').agg({"destination_x": "mean", "destination_y": "mean", "sum_of_txn_times": "sum"}).reset_index().rename(columns = {"off_stop": "station", "sum_of_txn_times": "demand_destination"})
demands = pd.merge(demands_origin, demands_destination, on = 'station', how = 'inner')
demands = demands.rename(columns = {"origin_x": "x", "origin_y": "y"}).drop(["destination_x", "destination_y"],axis = 1)
demands['idx'] = demands.index + 1

print(demands)