import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "data/First Time Home Buyers.xlsx"
df = pd.read_excel(file_path)
# file_path = './data/Monthly_Eviction_Filings_by_Location.csv'
# df = pd.read_csv(file_path)

# import glob

# # Get a list of all Excel files in the 'data' directory
# excel_files = glob.glob("data/*.xlsx")

# # Read the first Excel file from the list
# if excel_files:
#     df2 = pd.read_excel(excel_files[0])
# else:
#     print("No Excel files found in the directory.")

# Show basic information about the dataset and the first few rows
data_info = df.info()
first_rows = df.head()

data_info, first_rows

# # save into a smaller file
# # Take a sample of 50 rows without replacement
# sampled_df = df.sample(n=500, replace=False)

# # Save the sample to a new CSV file
# sampled_df.to_csv('./data/sampled_file.csv', index=False)

# 1. Trend Analysis Over Time
# Cleaning the data by removing rows with missing values in the numeric columns
home_buyers_data_cleaned = df.dropna(subset=['Avg house price', 'Avg money borrowed', 'Avg buyer income'])

# Manually computing the annual averages for 'Avg house price', 'Avg money borrowed', and 'Avg buyer income'
annual_averages = {
    'Year': [],
    'Avg house price': [],
    'Avg money borrowed': [],
    'Avg buyer income': []
}

for year in home_buyers_data_cleaned['Year'].unique():
    annual_averages['Year'].append(year)
    annual_averages['Avg house price'].append(
        home_buyers_data_cleaned.loc[home_buyers_data_cleaned['Year'] == year, 'Avg house price'].mean()
    )
    annual_averages['Avg money borrowed'].append(
        home_buyers_data_cleaned.loc[home_buyers_data_cleaned['Year'] == year, 'Avg money borrowed'].mean()
    )
    annual_averages['Avg buyer income'].append(
        home_buyers_data_cleaned.loc[home_buyers_data_cleaned['Year'] == year, 'Avg buyer income'].mean()
    )

# Converting the dictionary of lists into a DataFrame for plotting
annual_averages_df = pd.DataFrame(annual_averages)

# Plotting the trends over time for each of the three metrics
plt.figure(figsize=(15, 15))

# Plotting Average House Price
plt.subplot(3, 1, 1)
plt.plot(annual_averages_df['Year'], annual_averages_df['Avg house price'], marker='o', linestyle='-', color='blue')
plt.title('Average House Price Over Time (Yearly)')
plt.ylabel('Average House Price ($)')
plt.grid(True)

# Plotting Average Money Borrowed
plt.subplot(3, 1, 2)
plt.plot(annual_averages_df['Year'], annual_averages_df['Avg money borrowed'], marker='o', linestyle='-', color='green')
plt.title('Average Money Borrowed Over Time (Yearly)')
plt.ylabel('Average Money Borrowed ($)')
plt.grid(True)

# Plotting Average Buyer Income
plt.subplot(3, 1, 3)
plt.plot(annual_averages_df['Year'], annual_averages_df['Avg buyer income'], marker='o', linestyle='-', color='red')
plt.title('Average Buyer Income Over Time (Yearly)')
plt.ylabel('Average Buyer Income ($)')
plt.xlabel('Year')
plt.grid(True)

plt.tight_layout() # Adjust layout
plt.savefig('./output/trend_analysis.png', bbox_inches='tight')

plt.show()


# 2. Correlation Analysis
# Using the manually computed annual averages for correlation analysis
# Convert annual_averages dictionary to a DataFrame
annual_averages_df = pd.DataFrame.from_dict(annual_averages)

# Calculate the correlation matrix
correlation_matrix = annual_averages_df[['Avg house price', 'Avg money borrowed', 'Avg buyer income']].corr()

# Plotting the heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Housing Metrics')
plt.savefig('./output/correlation_analysis.png', bbox_inches='tight')
plt.show()

# 3. LTV & Home Price Multiple of Income Analysis
# Calculating LTV (Loan to Value) as 'Avg money borrowed' / 'Avg house price'
annual_averages_df['LTV'] = annual_averages_df['Avg money borrowed'] / annual_averages_df['Avg house price']

# Calculating House Price Multiple of Income as 'Avg house price' / 'Avg buyer income'
annual_averages_df['House Price Multiple of Income'] = annual_averages_df['Avg house price'] / annual_averages_df['Avg buyer income']

# Plotting LTV and House Price Multiple of Income over the years
plt.figure(figsize=(15, 7))

# LTV Plot
plt.plot(annual_averages_df['Year'], annual_averages_df['LTV'], label='LTV (Loan to Value)', marker='o')

# House Price Multiple of Income Plot
plt.plot(annual_averages_df['Year'], annual_averages_df['House Price Multiple of Income'], 
         label='House Price Multiple of Income', marker='x')

plt.title('Yearly LTV vs. House Price Multiples of Income')
plt.xlabel('Year')
plt.ylabel('Ratio')
plt.legend()
plt.grid(True)
plt.savefig('./output/ltv_multiple_analysis.png', bbox_inches='tight')
plt.show()


# 4. Regression Analysis
# Prepare the features (independent variables) and target (dependent variable)
X = annual_averages_df[['Avg money borrowed', 'Avg buyer income']]
y = annual_averages_df['Avg house price']

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict the 'Avg house price' using the model
y_pred = model.predict(X)

# Calculate the performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

mse, r2, coefficients, intercept


