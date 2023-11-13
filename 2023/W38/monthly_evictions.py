import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
file_path = './data/Monthly_Eviction_Filings_by_Location.csv'
df = pd.read_csv(file_path)

# Show basic information about the dataset and the first few rows
data_info = df.info()
first_rows = df.head()

data_info, first_rows

# # save into a smaller file
# # Take a sample of 50 rows without replacement
# sampled_df = df.sample(n=500, replace=False)

# # Save the sample to a new CSV file
# sampled_df.to_csv('./data/sampled_file.csv', index=False)

# 1. Distribution of average filings by city. Top 10
# import matplotlib.pyplot as plt

# Average filings by city, focusing on top 10 cities by count
top_cities = df['city'].value_counts().head(10).index
avg_filings_by_city = df[df['city'].isin(top_cities)].groupby('city')['filings'].mean().sort_values()

# Bar chart for average filings by top 10 cities
plt.figure(figsize=(12, 7))
avg_filings_by_city.plot(kind='barh')
plt.title('Average Filings by City (Top 10 Cities)')
plt.xlabel('Average Filings')
plt.ylabel('City')

# Save plot
plt.savefig('./output/avg_filings_by_city.png', bbox_inches='tight')

plt.show()

# 2. Distribution of average filings by facial majority
# Average filings by racial majority
avg_filings_by_racial_majority = df.groupby('racial_majority')['filings'].mean().sort_values()

# Bar chart for average filings by racial majority
plt.figure(figsize=(10, 6))
avg_filings_by_racial_majority.plot(kind='bar')
plt.title('Average Filings by Racial Majority')
plt.xlabel('Racial Majority')
plt.ylabel('Average Filings')
plt.xticks(rotation=45)

# Save plot
plt.savefig('./output/avg_filings_by_racial_majority.png', bbox_inches='tight')

plt.show()

# 3. Time Series Analysis
# Aggregate filings on a monthly basis with the reloaded and correctly interpreted data
monthly_filings_reloaded = df.groupby('month')['filings'].sum()

# Apply seasonal decomposition with a 12-month period
decomposition_reloaded = seasonal_decompose(monthly_filings_reloaded, model='additive', period=12)

# Visualization of the seasonal decomposition with a 12-month period
plt.figure(figsize=(14, 8))

# Trend
plt.subplot(311)
plt.plot(decomposition_reloaded.trend)
plt.title('Trend - 12-Month Period')

# Seasonal
plt.subplot(312)
plt.plot(decomposition_reloaded.seasonal)
plt.title('Seasonality - 12-Month Period')

# Residual
plt.subplot(313)
plt.plot(decomposition_reloaded.resid)
plt.title('Residuals - 12-Month Period')

plt.tight_layout()
# Save plot
plt.savefig('./output/seasonality.png', bbox_inches='tight')

plt.show()


 