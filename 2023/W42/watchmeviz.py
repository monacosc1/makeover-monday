import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np

# Load the dataset
file_path = './data/WatchMeViz.csv'
data = pd.read_csv(file_path)

# Show basic information about the dataset and the first few rows
data_info = data.info()
first_rows = data.head()

data_info, first_rows

# 1. Convert 'Video publish date' to datetime format
data['Video publish date'] = pd.to_datetime(data['Video publish date'], format='%d/%m/%Y')

# 2. Convert 'Average view duration' to a numeric format (in minutes)
def convert_duration_to_minutes(duration_str):
    """Converts a time duration string in the format 'HH:MM:SS' to a total number of minutes."""
    time_obj = datetime.datetime.strptime(duration_str, "%H:%M:%S")
    total_minutes = time_obj.hour * 60 + time_obj.minute + time_obj.second / 60
    return total_minutes

# Applying the conversion function to 'Average view duration'
data['Average view duration (minutes)'] = data['Average view duration'].apply(convert_duration_to_minutes)

# 3. Check for any missing or inconsistent data
missing_data = data.isnull().sum()
inconsistent_data = data.describe(include='all')

missing_data, inconsistent_data

# Setting the plot style to 'fivethirtyeight'
plt.style.use('fivethirtyeight')

# 1. Trend Analysis: Views & Likes
# Creating a figure and axis object
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plotting 'Views' on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Video Publish Date')
ax1.set_ylabel('Views', color=color)
ax1.plot(data['Video publish date'], data['Views'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Removing gridlines for the primary y-axis
ax1.grid(False)

# Creating a secondary y-axis to plot 'Likes'
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Likes', color=color)  
ax2.plot(data['Video publish date'], data['Likes'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Removing gridlines for the secondary y-axis
ax2.grid(False)

# Adding title
plt.title('Views and Likes Over Time', fontsize=16)

# Save plot
plt.savefig('./output/Views_and_Likes_Over_Time.png', bbox_inches='tight')

# Showing the plot
plt.show()

# 2. Relationship between variables: average view duration vs. average percentage viewed
# Creating a scatter plot for 'Average View Duration vs. Average Percentage Viewed'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Average view duration (minutes)', y='Average percentage viewed (%)')

# Adding title and labels
plt.title('Average View Duration vs. Average Percentage Viewed', fontsize=16)
plt.xlabel('Average View Duration (minutes)', fontsize=12)
plt.ylabel('Average Percentage Viewed (%)', fontsize=12)

# Save plot
plt.savefig('./output/avg_view_duration_by_avg_percent_viewed.png', bbox_inches='tight')

# Show the plot
plt.show()

from scipy.stats import linregress

# Performing linear regression
slope, intercept, r_value, p_value, std_err = linregress(data['Average view duration (minutes)'], data['Average percentage viewed (%)'])

# Showing the results
slope, intercept, r_value, p_value, std_err

# Interpretation:
# The positive slope indicates a positive relationship between average view duration and average percentage viewed.
# The correlation coefficient of 0.3611 suggests a moderate positive linear relationship.
# Most importantly, the p-value of 0.0001 is much less than 0.05, indicating that the relationship between average view duration and average percentage viewed is statistically significant.

# 3. Comparative analysis: average views, likes and watch time by year
# Creating a bar plot for 'Average Views by Year'
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='MM Year', y='Views', estimator=np.mean, ci=None, palette='mako')

# Adding title and labels
plt.title('Average Views by Year', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Number of Views', fontsize=12)

# Save plot
plt.savefig('./output/avg_stats_by_year.png', bbox_inches='tight')

# Show the plot
plt.show()

# 4. Top videos based on views, likes and watch time
# Identifying the top 3 performing videos based on Views, Likes, and Watch Time
top_3_views = data.nlargest(3, 'Views')[['Video title', 'Views']]
top_3_likes = data.nlargest(3, 'Likes')[['Video title', 'Likes']]
top_3_watch_time = data.nlargest(3, 'Watch time (hours)')[['Video title', 'Watch time (hours)']]

top_3_views, top_3_likes, top_3_watch_time

from IPython.display import display, HTML
import dataframe_image as dfi

# Displaying the tables in a more formatted way
display(HTML('<h3>Top 3 Videos Based on Views</h3>'))
display(top_3_views.style.set_properties(**{'text-align': 'left'}))

display(HTML('<h3>Top 3 Videos Based on Likes</h3>'))
display(top_3_likes.style.set_properties(**{'text-align': 'left'}))

display(HTML('<h3>Top 3 Videos Based on Watch Time (Hours)</h3>'))
display(top_3_watch_time.style.set_properties(**{'text-align': 'left'}))

# Save the tables as images
dfi.export(top_3_views, './output/Top_3_Views.png')
dfi.export(top_3_likes, './output/Top_3_Likes.png')
dfi.export(top_3_watch_time, './output/Top_3_Watch_Time.png')

# 5. Correlation Analysis. Correlation matrix of numeric variables
# Selecting only the numeric columns from the dataset
numeric_data = data.select_dtypes(include=[np.number])

# Calculating the correlation matrix
correlation_matrix = numeric_data.corr()

# Creating a heatmap for the correlation matrix
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
heatmap.set_title('Correlation Matrix', fontdict={'fontsize':18}, pad=16)

# Save plot
plt.savefig('./output/correlation_matrix.png', bbox_inches='tight')

plt.show()

# Observations:
# Views and Likes: There is a moderate positive correlation (0.60) between views and likes, indicating that videos with more views tend to have more likes.
# Views and Watch Time: Views and watch time are highly positively correlated (0.92), which makes sense as more views would generally lead to more total watch time.
# Likes and Watch Time: Likes and watch time also have a positive correlation (0.67), though it's not as strong as the correlation between views and watch time.
# Average View Duration and Average Percentage Viewed: There is a moderate positive correlation (0.35) between average view duration and average percentage viewed, which we have analyzed earlier.