import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd

# Load the dataset
file_path = './data/Hyperlocal_Temperature_Monitoring.csv'
df = pd.read_csv(file_path)

# Show basic information about the dataset and the first few rows
data_info = df.info()
first_rows = df.head()

data_info, first_rows

# # save into a smaller file
# # Take a sample of 50 rows without replacement
# sampled_df = df.sample(n=50, replace=False)

# # Save the sample to a new CSV file
# sampled_df.to_csv('./data/sampled_file.csv', index=False)


# Geospatial Analysis:
# Cluster analysis to determine if there are geographically distinct clusters of temperature readings.

from sklearn.cluster import KMeans

# Preprocess the data: drop rows with missing values in 'Latitude', 'Longitude', or 'AirTemp'
df_clean = df.dropna(subset=['Latitude', 'Longitude', 'AirTemp'])

# Selecting features for clustering
X = df_clean[['Latitude', 'Longitude', 'AirTemp']]

# Using the Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the results onto a line graph to observe 'The elbow'
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares

# Save plot
plt.savefig('./output/elbow_rule.png', bbox_inches='tight')

plt.show()

# Applying k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_clean['Cluster'] = kmeans.fit_predict(X)

# Visualizing the clusters
plt.figure(figsize=(10, 8))

# Scatter plot for clusters
plt.scatter(df_clean[df_clean['Cluster'] == 0]['Longitude'], df_clean[df_clean['Cluster'] == 0]['Latitude'],
            s=100, c='red', label ='Cluster 1')
plt.scatter(df_clean[df_clean['Cluster'] == 1]['Longitude'], df_clean[df_clean['Cluster'] == 1]['Latitude'],
            s=100, c='blue', label ='Cluster 2')
plt.scatter(df_clean[df_clean['Cluster'] == 2]['Longitude'], df_clean[df_clean['Cluster'] == 2]['Latitude'],
            s=100, c='green', label ='Cluster 3')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=300, c='yellow', label = 'Centroids')

plt.title('Clusters of sensors')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# Save plot
plt.savefig('./output/kmeans_cluster.png', bbox_inches='tight')

plt.show()

# Temperature variation by borough. 
# Bar plots to compare average temperatures across boroughs
# Statistical tests to determine if the differences in temperatures are statistically different

# Calculate the average temperature for each borough
borough_avg_temp = df_clean.groupby('Borough')['AirTemp'].mean().reset_index()

# Create a bar plot to compare average temperatures across boroughs
plt.figure(figsize=(12, 6))
plt.bar(borough_avg_temp['Borough'], borough_avg_temp['AirTemp'], color='skyblue')
plt.title('Average Temperature by Borough')
plt.xlabel('Borough')
plt.ylabel('Average Temperature (°F)')
plt.xticks(rotation=45)  # Rotate the x-axis labels to show them more clearly

# Save plot
plt.savefig('./output/comparison_avg_temperatures.png', bbox_inches='tight')

plt.show()

# Display the average temperatures for reference
borough_avg_temp

from scipy.stats import shapiro, levene

# Shapiro-Wilk Test for normality for each borough
normality_results = {borough: shapiro(df[df['Borough'] == borough]['AirTemp']).pvalue for borough in df['Borough'].unique()}

# Levene's test for equal variances
levene_stat, levene_p = levene(*[df[df['Borough'] == borough]['AirTemp'] for borough in df['Borough'].unique()])

normality_results, (levene_stat, levene_p)

# Since all the p-values from the Shapiro-Wilk test are greater than 0.05, we fail to
# reject the null hypothesis of normality, indicating that the temperature data for 
# each borough does not deviate significantly from a normal distribution.

from scipy.stats import f_oneway

# ANOVA test
anova_stat, anova_p = f_oneway(
    df[df['Borough'] == 'Bronx']['AirTemp'],
    df[df['Borough'] == 'Brooklyn']['AirTemp'],
    df[df['Borough'] == 'Manhattan']['AirTemp'],
    df[df['Borough'] == 'Queens']['AirTemp']
)

anova_stat, anova_p

# With a p-value much higher than the typical significance level of 0.05, there is no 
# statistical evidence to reject the null hypothesis. This indicates that the mean 
# temperatures across the different boroughs do not differ significantly in our sample.

# Thus, according to this ANOVA test on the sampled data, it would appear that the 
# average temperatures are statistically similar across the boroughs included in 
# this dataset. 

# Comparing temperature data year over year to identify any trends or changes using 
# line plots or area plots.

# Convert the 'Day' column to datetime
df['Day'] = pd.to_datetime(df['Day'])

# Extract year and month from the 'Day' column
df['Year'] = df['Day'].dt.year
df['Month'] = df['Day'].dt.month

# Calculate the average temperature for each month-year combination
monthly_avg_temp = df.groupby(['Year', 'Month'])['AirTemp'].mean().reset_index()

# Pivot the data to have years as columns and months as rows for plotting
monthly_avg_temp_pivot = monthly_avg_temp.pivot(index='Month', columns='Year', values='AirTemp')

# Plotting the data
plt.figure(figsize=(14, 7))
for year in monthly_avg_temp_pivot.columns:
    plt.plot(monthly_avg_temp_pivot.index, monthly_avg_temp_pivot[year], marker='o', label=year)
    
plt.title('Average Monthly Temperature Year over Year')
plt.xlabel('Month')
plt.ylabel('Average Temperature (°F)')
plt.legend(title='Year')
plt.grid(True)

# Save plot
plt.savefig('./output/yearly_changes.png', bbox_inches='tight')

plt.show()

# mean and median temperatures at different hours to identify the hottest
# and coolest parts of the day

# Group the data by the 'Hour' column and calculate mean and median temperatures for each hour
hourly_temp_stats = df.groupby('Hour')['AirTemp'].agg(['mean', 'median']).reset_index()

# Plotting the results
plt.figure(figsize=(14, 7))

# Plot mean temperatures
plt.plot(hourly_temp_stats['Hour'], hourly_temp_stats['mean'], label='Mean Temperature', marker='o')

# Plot median temperatures
plt.plot(hourly_temp_stats['Hour'], hourly_temp_stats['median'], label='Median Temperature', marker='x')

plt.title('Mean and Median Temperatures by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Temperature (°F)')
plt.xticks(hourly_temp_stats['Hour'])  # Ensure all hours are represented on the x-axis
plt.legend()
plt.grid(True)

# Save plot
plt.savefig('./output/hourly_changes.png', bbox_inches='tight')

plt.show()

# Display the statistics for reference
hourly_temp_stats

# geographic mapping

import geopandas as gpd
import folium
from folium.plugins import HeatMap

# Load a GeoDataFrame with New York's borough boundaries
# Make sure to have the corresponding shapefile (.shp) for New York's boroughs
gdf = gpd.read_file('./data/geo_export_7458d700-7510-4eee-a6a3-49b1c273aea4.shp')

# Create a base map centered around New York
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Plot each sensor as a point on the map
for idx, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        popup=f"Temperature: {row['AirTemp']}°F",
        color='blue' if row['AirTemp'] < 60 else 'red',
        fill=True
    ).add_to(m)

# Optionally, you can create a heat map layer instead
# heat_data = [[row['Latitude'], row['Longitude']] for idx, row in df.iterrows()]
# HeatMap(heat_data).add_to(m)

# Overlay the GeoDataFrame
# Make sure the GeoDataFrame is in the same projection as the folium map (EPSG:4326)
folium.GeoJson(gdf).add_to(m)

# Save to an HTML file
m.save('./output/ny_temperature_map.html')

# To display in a Jupyter notebook environment
# m

