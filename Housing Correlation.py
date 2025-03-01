# Assume actual_median_mean.csv & predicted_price.csv are 2 databases
# Practise SQL by merging data from 2 database

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from matplotlib.ticker import FormatStrFormatter
#import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
import os

try:
    os.remove('correlation.csv')
except FileNotFoundError:
    pass

actual_df = pd.read_csv('actual_median_mean.csv')
predicted_df = pd.read_csv('predicted_price.csv')

# SQLite Method
# Only appends date matching data from both csv files.
'''
conn = sqlite3.connect(':memory:')
actual_df.to_sql('actual_prices', conn, index=False)
predicted_df.to_sql('predicted_prices', conn, index=False)

query = """
SELECT 
    a.date, 
    a.median_actual_prices, 
    a.mean_actual_prices, 
    p.predicted_price
FROM actual_prices a
JOIN predicted_prices p ON a.date = p.date;
"""
merged_df = pd.read_sql_query(query, conn)
conn.close()

print(merged_df)

# Save merged data to csv for checking purposes
os.remove('correlation.csv)
merged_df.to_csv('correlation.csv')

# Correlation
correlation_mean = np.corrcoef(merged_df['predicted_price'], merged_df['mean_actual_prices'])[0,1]
print(f"Predicted vs Actual Mean Correlation: {correlation_mean:.5f}")

correlation_median = np.corrcoef(merged_df['predicted_price'], merged_df['median_actual_prices'])[0,1]
print(f"Predicted vs Actual Median Correlation: {correlation_median:.5f}")
'''

# Python Method
# Using outer join on 'date' column to include all dates.
# Predicted has more data than actual, outer join will let data frame recognise additional rows of data.

merged_df = pd.merge(actual_df, predicted_df, on='date', how='outer')
try:
    existing_df = pd.read_csv('correlation.csv')
    merged_df.to_csv('correlation.csv', mode='a', header=False, index=False)
except FileNotFoundError:
    merged_df.to_csv('correlation.csv', index=False)

# Correlation with Python
clean_df = merged_df.dropna()   # Exclude empty rows to prevent correlation error
if not clean_df.empty:
    correlation_mean = np.corrcoef(clean_df['predicted_price'],
clean_df['mean_actual_prices'])[0,1]
    print(f"Predicted vs Actual Mean Correlation:{correlation_mean:.5f}")

    correlation_median = np.corrcoef(clean_df['predicted_price'],
clean_df['median_actual_prices'])[0,1]
    print(f"Predicted vs Actual Median Correlation:{correlation_median:.5f}")
else:
    print("No valid data to calculate correlation")


# Create Plot with Outliers
df_corr_check = pd.read_csv("correlation.csv", parse_dates=['date'])
df_outliers = pd.read_csv("outliers.csv", parse_dates=['date'])

'''
# Drop rows with missing values, but this will also truncate Predicted Prices.
df_corr_check = df_corr_check.dropna(subset=[
    'mean_actual_prices',
    'median_actual_prices'
])
'''

# Create 2 Plots Side By Side
# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 6))
fig = make_subplots(rows=1, cols=2, subplot_titles=('Outlier View', 'Zoom View'))
                                                    

# Subplot with Outliers
fig.add_trace(go.Scatter(
    x=df_corr_check['date'],
    y=df_corr_check['mean_actual_prices'],
    mode='lines',
    name='Mean',
    hovertemplate='Date: %{x}<br>Mean Price: %{y:,.0f}',
    line=dict(color='red'),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_corr_check['date'],
    y=df_corr_check['median_actual_prices'],
    mode='lines',
    name='Median',
    hovertemplate='Date: %{x}<br>Median Price: %{y:,.0f}',
    line=dict(color='green'),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_corr_check['date'],
    y=df_corr_check['predicted_price'],
    mode='markers',
    name='Predicted',
    hovertemplate='Date: %{x}<br>Model Predicted Price: %{y:,.0f}',
    marker=dict(color='blue', symbol='circle'),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df_outliers['date'],
    y=df_outliers['resale_price'],
    mode='markers',
    name='Outliers',
    hovertemplate='Date: %{x}<br>Outlier: %{y:,.0f}',
    marker=dict(color='gray', symbol='circle-open'), 
), row=1, col=1)


# Subplot without Outliers
fig.add_trace(go.Scatter(
    x=df_corr_check['date'],
    y=df_corr_check['mean_actual_prices'],
    mode='lines',
    name='Mean',
    hovertemplate='Date: %{x}<br>Mean Price: %{y:,.0f}',
    line=dict(color='red'),
    showlegend=False
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=df_corr_check['date'],
    y=df_corr_check['median_actual_prices'],
    mode='lines',
    name='Median',
    hovertemplate='Date: %{x}<br>Median Price: %{y:,.0f}',
    line=dict(color='green'),
    showlegend=False  
), row=1, col=2)

fig.add_trace(go.Scatter(
    x=df_corr_check['date'],
    y=df_corr_check['predicted_price'],
    mode='markers',
    name='Predicted',
    hovertemplate='Date: %{x}<br>Model Predicted Price: %{y:,.0f}',
    marker=dict(color='blue', symbol='circle'),
    showlegend=False
), row=1, col=2)

# Update Layout
fig.update_layout(
    title= dict(text="Housing Price Prediction - Random Forest Regressor", font=dict(size=24), automargin=True, yref='paper'),
    title_text = "Housing Price Prediction - Random Forest Regressor",
    title_font_size = 24,
    showlegend=True,
    legend=dict(x=0.80, y=0.25),
    legend_font_size=16,
    
    annotations=[
        dict(text=f"Correlation<br>Predicted vs Mean= {correlation_mean:.5f} <br>Predicted vs Median = {correlation_median:.4f} ", x=0.65, y=0.85, xref="paper", yref="paper", align='left', showarrow=False),
        dict(text="Zoom View", x=0.75, y=1),
    ],
    
    autosize=True,
    hovermode='x unified',  # Show hover information 
)
#fig.show()

fig.write_html("Raw Data/plots/CorrelationCheck.html")
