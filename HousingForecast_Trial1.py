# Source Data: https://data.gov.sg/datasets?topics=housing&page=1&resultId=189
# Tutorial: https://www.youtube.com/watch?v=vV12dGe_Fho
# Tutorial uses XGBoost for Time Series forecasting
# Model Selection: https://www.researchgate.net/publication/383112591_A_Comparative_Study_of_Regression_Models_for_Housing_Price_Prediction
# https://www.ibm.com/think/topics/random-forest
# We will use Prophet and RandomForestRegressor for forecasting instead.
# Reason 1: Handles classification & regression problems.
# Reason 2: Predicts better results, particularly when individual data are uncorrelated.
# Reason 3: Decision Tree Regression more sensitive to noise and outliers.
# Justification: Resale house are scattered and random.  Even if they are in the same area, there are other factors affecting the resale prices.
# Assumptions:  We assume fiscal and economic stability in this modelling, and does not take into account the risk of black swan events.


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from datetime import datetime, timedelta

# Load dataset
df = pd.read_csv("Raw Data/Resale_House_Price.csv")

# Convert date column to datetime format, extract YYYY & MM
df['date'] = pd.to_datetime(df['date'])

# Create features from date into dataframe
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Because csv dataset does not have separate YYYY & MM columns, this line will use dataframe to take care about this requirement
train_mask = df['date'] < pd.to_datetime('2024-11')

# Split data into training and test sets
# Data up to 2024 used for training
# Data up to 2025-02 used for validation

X_train = df[train_mask][['year', 'month']]
y_train = df[train_mask]['resale_price']


test_mask = (df['date'] >= pd.to_datetime('2024-11')) & (df['date'] < pd.to_datetime('2026-01'))
X_test = df[test_mask][['year', 'month']]
y_test = df[test_mask]['resale_price']


# Initialize  & train Random Forest Regressor
    # Question to Study: Why n_estimators = 100 because it's a default setting in Sklearn
        # Higher number means more trees build before returning max voting or average of predictions
        # >100 limited by processor, also affects speed of code.
    # Question to Study why state = 42?
    # oob_score needs to turn on, else oob_score will return false even though model will compute and store in model.oob_score_
    # oob measures prediction error of random forests
model = RandomForestRegressor(n_estimators=100,random_state=42, oob_score=True)
model.fit(X_train, y_train)


# Evaluate model performance on test data using Mean Squared Error (mse) & R-squared (R2)
    # Question to Study"  What is **2?
test_predictions = model.predict(X_test)
#print("Model RMSE on 2025 data:", np.sqrt(np.mean((test_predictions - y_test)**2)))

oob_score = model.oob_score_
print(f"Out-of-Bag Score: {oob_score}")

mse = mean_squared_error(y_test, test_predictions)
print(f"Mean Square Error: {mse}")

r2 = r2_score(y_test, test_predictions)
print(f"R-squared: {r2}")


# Prepare future dates for prediction (2017-01 to 2026-12)
future_dates = pd.date_range(start='2017-01', end='2026-12', freq='MS')
future_df = pd.DataFrame({'date': future_dates, 'year': future_dates.year, 'month': future_dates.month})

# Predict prices for future dates
future_predictions = model.predict(future_df[['year', 'month']])
future_df['predicted_price'] = future_predictions


# Combine historical and predicted data for plotting
    # Notna, Not NA, is Panda's function that checks for non-missing values in DataFrame / Series.
    # Returns a boolean mask:  True = value not missing.  False = Value missing
combined_data = pd.concat([df[['date', 'resale_price']], future_df[['date', 'predicted_price']]], ignore_index=True)
combined_data['type'] = ['Historical' if pd.notna(x) else 'Predicted' for x in combined_data['resale_price']]


# Create interactive chart
fig = px.line(
    combined_data, 
    x='date', 
    y=['resale_price', 'predicted_price'], 
    labels={'variable': 'Data Type'},
    line_shape='spline', 
    markers=True,
    color_discrete_map={
        'Historical':'grey', 
        'Predicted': 'blue'
    }
)


# 22 Feb 2025 - Original Chart using Traces
# Question to Study:  What is <extra></extra>
fig.update_traces(hovertemplate='%{x}<br>Price:%{y:,.0f}$<extra></extra>')
fig.update_layout(
    title='Housing Price Prediction', 
    xaxis_title='Date',
    yaxis_title='Resale Price',
    hovermode='x unified')

# 23 Feb 2025 - Update Chart again
# Add actual vs predicted values to chart
#fig.add_scatter(x=X_train.index, y=y_train.values, mode = 'markers', name = 'Actual (Train)')
#fig.add_scatter(x=X_test.index, y=test_predictions, mode='lines', name='Predicted (Test)')


# Add OOB Score, MSE and R-Squared to Chart
fig.update_layout(
    annotations=[
        dict(text=f"OOB Score: {oob_score:.3f}", x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False),
        dict(text=f"MSE: {mse:.2f}", x=0.05, y=0.90, xref="paper", yref="paper", showarrow=False),
        dict(text=f"R-Squared: {r2:.3f}", x=0.05, y=0.85, xref="paper", yref="paper", showarrow=False)
    ]
)


# Display Chart
fig.show()