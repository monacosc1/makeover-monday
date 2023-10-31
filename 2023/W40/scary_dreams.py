import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd

# Load the dataset
file_path = './data/Scary-Dreams.csv'
df_scary = pd.read_csv(file_path)

# Show basic information about the dataset and the first few rows
data_info = df_scary.info()
first_rows = df_scary.head()

data_info, first_rows

# Rename the 'score' column to 'test-score'
df_scary.rename(columns={'scary dreams: (United States)': 'scary_dreams_score'}, inplace=True)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df_scary['Month'], df_scary['scary_dreams_score'], marker='o', color='b', linestyle='-')

# Adding title and labels
plt.title('Google Search of "Scary Dreams" Over Time', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Scary Dreams', fontsize=14)

# Adjusting the x-ticks to display only January of each year
months = df_scary['Month'].tolist()
labels = [month if '-01' in month else '' for month in months]
plt.xticks(months, labels, rotation=45)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save plot
plt.savefig('./output/scary_dreams_over_time.png', bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()

# Load the dataset
file_path = './data/geoMap.csv'
df_geo = pd.read_csv(file_path, skiprows=2)

# Show basic information about the dataset and the first few rows
data_info = df_geo.info()
first_rows = df_geo.head()

data_info, first_rows

# Rename columns by assigning a new list of column names
df_geo.columns = ['State', 'scary_dreams_score']

# # Load world map data and filter for the US
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# us_map = world[world.name == "United States of America"].explode()

# Since this dataset doesn't have individual US states, let's use a different source for US states
us_states = gpd.read_file('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')

# Merge map data with the scores
merged = us_states.set_index('name').join(df_geo.set_index('State'))

# Filter out Alaska
merged = merged[merged.index != "Alaska"]

# Fill NaN values with a specific number (e.g., -1) to denote them
merged['scary_dreams_score'].fillna(-1, inplace=True)

# Create a custom colormap where -1 corresponds to lightgray
cmap = plt.cm.colors.ListedColormap(['lightgray', *plt.cm.Blues(np.linspace(0, 1, 256))])
norm = plt.cm.colors.Normalize(vmin=-1, vmax=100)

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

# Plot the map with outlines for all states
merged.plot(column='scary_dreams_score', cmap=cmap, norm=norm, linewidth=0.8, edgecolor='black', ax=ax, legend=True)

# Set the title
ax.set_title('Scary Dreams Scores by State', fontdict={'fontsize': '25', 'fontweight' : '3'})

# Remove the x and y axis
ax.axis('off')

# Save plot
plt.savefig('./output/geo_map_scary_dreams.png', bbox_inches='tight')

plt.show()