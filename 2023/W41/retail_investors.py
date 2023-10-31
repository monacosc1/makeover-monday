import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset
file_path = './data/Retail_Investors_Focus.xlsx'
data = pd.read_excel(file_path)

# Show basic information about the dataset and the first few rows
data_info = data.info()
first_rows = data.head()

data_info, first_rows

# Sort data by Percent in descending order
data['Investment Strategy'] = [x for _, x in sorted(zip(data['Percent of Respondents'], data['Investment Strategy']), reverse=True)]
data['Percent of Respondents'] = sorted(data['Percent of Respondents'], reverse=True)

# Set up the plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))

# Draw the bar chart
sns.barplot(y=data['Investment Strategy'], x=data['Percent of Respondents'], orient='h', palette="rocket")

# Customize the plot
plt.title('What are Retail Investors\nInterested in Buying in 2023?', fontsize=16, fontweight='bold')
plt.xlabel('Percent of Respondents', fontsize=14)
plt.ylabel('')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add percentages on the bars
for index, value in enumerate(data['Percent of Respondents']):
    plt.text(value, index, f' {value}%', va='center', fontsize=12, color='white', fontweight='bold')

# Save plot
plt.savefig('./output/percent_retail_buying.png', bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()