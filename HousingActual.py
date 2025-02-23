import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import numpy as np
import seaborn as sns
import os

# Read CSV file
#df = pd.read_csv("Raw Data/Resale_House_Price.csv", parse_dates=['date'])
file_path = "Raw Data/Resale_House_Price.csv"
df = pd.read_csv(file_path, parse_dates=['date'])

# Check 'date' is in YYYY-MM format
df['date'] = df['date'].dt.strftime('%Y-%m')

# Create sub-folder 'plots' in 'Raw Data' folder and save plots
save_dir = os.path.dirname(file_path)                
save_plots_dir = os.path.join (save_dir, 'plots')

# Create folder if it doesn't exist
os.makedirs(save_plots_dir, exist_ok=True)

# Group by 'YYYY-MM' and calculate max, min, median prices
monthly_stats = df.groupby('date')['resale_price'].agg(['max', 'min', 'median']).reset_index()

# Create Line Plot
plt.figure(figsize = (10, 6))
plt.plot(monthly_stats['date'], monthly_stats['max'], color='green', linestyle='-', label='Max Price')
plt.plot(monthly_stats['date'], monthly_stats['min'], color='red', linestyle='-', label='Min Price')
plt.plot(monthly_stats['date'], monthly_stats['median'], color='blue', linestyle='-', label='Median Price')

# Set custom y-axis ticks at 100k intervals
min_value = df['resale_price'].min()
max_value = df['resale_price'].max()

# Create tick locations at every 100k
tick_locations = np.arange(min_value, max_value + 100000, 100000)

plt.yticks(tick_locations)

# Format y-axis labels to display comma (e.g. 200,000)
def format_ticks(value, tick_number):
    if value >= 1e6:
            return f"{value / 1e6:.1f}M"
    elif value >= 1e3:
          return f"{int(value / 1e3)}K"
    else:
        return int(value)

formatter = ticker.FuncFormatter(format_ticks)
#formatter.set_scientific(False) # Avoid scientific notation such as x 10^6
#formatter.set_useOffset(False)

current_axis = plt.gca()  #Get current axis
current_axis.yaxis.set_major_formatter(formatter)

# Format x-axis to show alternate 'date' entry
current_axis.xaxis.set_major_locator(ticker.IndexLocator(base=2, offset=0))

# Customize Line Plot
plt.xlabel('Date (YYYY-MM)')
plt.ylabel('Resale Price')
plt.title('Monthly Resale Price Statistics')
plt.legend()
plt.xticks(rotation=90)
plt.grid(True)

#plt.tight_layout()

# Save Line Plot
line_plot_path = os.path.join(save_plots_dir, 'LinePlot.png')
plt.savefig(line_plot_path, dpi=300, bbox_inches='tight')
plt.close()


# Create Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(x='date', y='resale_price', data=df)

formatter = ticker.FuncFormatter(format_ticks)
#formatter.set_scientific(False) # Avoid scientific notation such as x 10^6
#formatter.set_useOffset(False)

current_axis = plt.gca()  #Get current axis
current_axis.yaxis.set_major_formatter(formatter)

# Format x-axis to show alternate 'date' entry
current_axis.xaxis.set_major_locator(ticker.IndexLocator(base=2, offset=0))

# Customize Boxplot
plt.title('Monthly Resale Price Distribution')
plt.xlabel('Date (YYYY-MM)')
plt.ylabel('Resale Price')
plt.xticks(rotation=90)
plt.grid(True)

# Save Boxplot
box_plot_path = os.path.join(save_plots_dir, 'BoxPlot.png')
plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plots saves at: {save_plots_dir}")
