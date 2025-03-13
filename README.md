# 📊 Gold Price Forecasting 📈

This project analyzes and forecasts gold prices using time series techniques such as **SARIMA** and **Box-Cox transformation**. It includes visualization, stationarity checks, and model selection.

## 📂 Project Overview

- **Data Source:** XAU/USD daily price data (`XAU_1d_data.csv`)
- **Tools Used:** Python, Pandas, Matplotlib, Statsmodels, Scipy, Seaborn
- **Forecasting Model:** SARIMA with Grid Search Optimization

## 🛠 Key Features

### 1️⃣ **Data Processing**
- Loads gold price data and converts `Date` column to **datetime format**.
- Resamples data to **monthly averages**.
- Handles missing values and extracts numerical features.

### 2️⃣ **Visualization**
- Plots **monthly gold price trends**.
- Uses **seasonal decomposition** to analyze trends & cycles.

### 3️⃣ **Stationarity Test**
- Uses **Dickey-Fuller test** to check for stationarity.
- Applies **Box-Cox transformation** and **differencing** to stabilize variance.

### 4️⃣ **SARIMA Model Selection**
- Uses **Grid Search** to find the best SARIMA parameters.
- Selects the model with the lowest **AIC (Akaike Information Criterion)**.

### 5️⃣ **Forecasting**
- Predicts future gold prices for the next **24 months**.
- Visualizes actual vs. forecasted prices.

## 📊 Visualizations

### **Gold Price Trends**
![Gold Price Trend](https://your-image-link.png)

### **SARIMA Forecast**
![Forecast](https://your-forecast-image-link.png)

## 📝 Code Snippet

```python
# Train SARIMA model with optimal parameters
model = sm.tsa.statespace.SARIMAX(df_month['Close_box'].dropna(),
                                  order=(1, 1, 1),
                                  seasonal_order=(1, 1, 1, 12)).fit()
                                  
# Forecast next 24 months
df_month2['forecast'] = invboxcox(model.predict(start=len(df_month), end=len(df_month2)-1), lmbda)
