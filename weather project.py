!pip install pandas matplotlib seaborn plotly statsmodels pmdarima

# Step 2: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from google.colab import files
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Step 3: Upload CSV in Google Colab
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Step 4: Read file
df = pd.read_csv(file_name, encoding="latin1")

# Step 5: Clean & prepare data
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

pollutant_cols = ["so2", "no2", "rspm", "spm", "pm2_5"]
df = df.dropna(subset=pollutant_cols, how='all')

# Step 6: Summary table - Average pollutants per state
avg_pollutants = df.groupby("state")[pollutant_cols].mean(numeric_only=True).reset_index()
avg_pollutants = avg_pollutants.sort_values(by="pm2_5", ascending=False)
print("📊 Average Pollutant Levels by State:")
print(avg_pollutants)

# Step 7: Bar Chart - Average PM2.5 per state
plt.figure(figsize=(10,6))
sns.barplot(data=avg_pollutants, x="pm2_5", y="state", palette="viridis")
plt.title("Average PM2.5 Levels by State")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("State")
plt.show()

# Step 8: Time Series Plot - PM2.5
if "pm2_5" in df.columns and "date" in df.columns:
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x="date", y="pm2_5", color="red")
    plt.title("PM2.5 Levels Over Time")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.show()

# Step 9: Correlation Plot - SO2 vs NO2
if "so2" in df.columns and "no2" in df.columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="so2", y="no2", color="purple", alpha=0.6)
    sns.regplot(data=df, x="so2", y="no2", scatter=False, color="black")
    plt.title("Correlation: SO₂ vs NO₂")
    plt.xlabel("SO₂ (µg/m³)")
    plt.ylabel("NO₂ (µg/m³)")
    plt.show()

# Step 10: PM2.5 Forecasting
if "pm2_5" in df.columns and "date" in df.columns:
    ts_data = df.dropna(subset=["pm2_5"]).sort_values("date")
    ts_data["month"] = ts_data["date"].dt.to_period("M").dt.to_timestamp()
    monthly_pm25 = ts_data.groupby("month")["pm2_5"].mean()

    # Fit ARIMA model
    model = pm.auto_arima(monthly_pm25, seasonal=True, m=12, stepwise=True, suppress_warnings=True)

    # Forecast next 12 months
    forecast_values = model.predict(n_periods=12)
    forecast_index = pd.date_range(start=monthly_pm25.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    forecast_series = pd.Series(forecast_values, index=forecast_index)

    # Plot forecast
    plt.figure(figsize=(12,6))
    plt.plot(monthly_pm25, label="Historical PM2.5")
    plt.plot(forecast_series, label="Forecast PM2.5", linestyle="--")
    plt.title("Forecast of PM2.5 Levels for Next 12 Months")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.show()
