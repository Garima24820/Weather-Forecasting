#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Pragya priya
#
# Created:     16-06-2024
# Copyright:   (c) Pragya priya 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Sample data for demonstration (dates and temperatures)
dates = np.array(['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06'])
dates = np.array([datetime.strptime(date, "%Y-%m-%d") for date in dates])
temperatures = np.array([22.5, 23.1, 24.0, 25.3, 26.5, 27.8])

# Reshape dates for sklearn input (as days since the first date)
days_since = np.array([(date - dates[0]).days for date in dates]).reshape(-1, 1)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model using the dates (days since) and temperatures
model.fit(days_since, temperatures)

# Predicting future temperatures for the next 5 days
future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 6)]
future_days_since = np.array([(date - dates[0]).days for date in future_dates]).reshape(-1, 1)
future_temperatures = model.predict(future_days_since)

# Printing the predicted temperatures for future dates
for i, date in enumerate(future_dates):
    print(f"Predicted temperature on {date.strftime('%Y-%m-%d')}: {future_temperatures[i]:.2f} °C")

# Plotting the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(dates, temperatures, color='blue', label='Actual temperatures')
plt.plot(dates, model.predict(days_since), color='red', linestyle='-', linewidth=2, label='Linear regression')
plt.scatter(future_dates, future_temperatures, color='green', label='Predicted temperatures')
plt.title('Temperature Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
