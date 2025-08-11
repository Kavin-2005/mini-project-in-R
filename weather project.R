# ===== AIR QUALITY REPORT, ANALYSIS & FORECASTING =====

# Step 1: Install packages (run once)
install.packages(c("tidyverse", "lubridate", "forecast", "tseries"))

# Step 2: Load libraries
library(tidyverse)
library(lubridate)
library(forecast)
library(tseries)

# Step 3: Upload CSV (in RStudio or Colab-like R env, choose file)
file_name <- file.choose()

# Step 4: Read file
df <- read.csv(file_name, fileEncoding = "latin1", stringsAsFactors = FALSE)

# Step 5: Clean & prepare data
if ("date" %in% names(df)) {
  df$date <- as.Date(df$date, format = "%Y-%m-%d")
}

pollutant_cols <- c("so2", "no2", "rspm", "spm", "pm2_5")

# Remove rows with all pollutant columns missing
df <- df %>%
  filter(rowSums(is.na(select(., all_of(pollutant_cols)))) < length(pollutant_cols))

# Step 6: Summary table - Average pollutants per state
avg_pollutants <- df %>%
  group_by(state) %>%
  summarise(across(all_of(pollutant_cols), mean, na.rm = TRUE)) %>%
  arrange(desc(pm2_5))

cat("📊 Average Pollutant Levels by State:\n")
print(avg_pollutants)

# Step 7: Bar Chart - Average PM2.5 per state
ggplot(avg_pollutants, aes(x = pm2_5, y = reorder(state, pm2_5))) +
  geom_col(fill = "darkgreen") +
  labs(title = "Average PM2.5 Levels by State",
       x = "PM2.5 (µg/m³)",
       y = "State") +
  theme_minimal()

# Step 8: Time Series Plot - PM2.5
if ("pm2_5" %in% names(df) && "date" %in% names(df)) {
  ggplot(df, aes(x = date, y = pm2_5)) +
    geom_line(color = "red") +
    labs(title = "PM2.5 Levels Over Time",
         x = "Date",
         y = "PM2.5 (µg/m³)") +
    theme_minimal()
}

# Step 9: Correlation Plot - SO2 vs NO2
if ("so2" %in% names(df) && "no2" %in% names(df)) {
  ggplot(df, aes(x = so2, y = no2)) +
    geom_point(color = "purple", alpha = 0.6) +
    geom_smooth(method = "lm", color = "black") +
    labs(title = "Correlation: SO₂ vs NO₂",
         x = "SO₂ (µg/m³)",
         y = "NO₂ (µg/m³)") +
    theme_minimal()
}

# Step 10: PM2.5 Forecasting
if ("pm2_5" %in% names(df) && "date" %in% names(df)) {
  ts_data <- df %>%
    filter(!is.na(pm2_5)) %>%
    arrange(date) %>%
    mutate(month = floor_date(date, "month")) %>%
    group_by(month) %>%
    summarise(pm2_5 = mean(pm2_5, na.rm = TRUE))
  
  # Convert to time series object
  pm_ts <- ts(ts_data$pm2_5, start = c(year(min(ts_data$month)), month(min(ts_data$month))), frequency = 12)
  
  # Fit ARIMA model
  fit <- auto.arima(pm_ts, seasonal = TRUE)
  
  # Forecast next 12 months
  fc <- forecast(fit, h = 12)
  
  # Plot forecast
  autoplot(fc) +
    labs(title = "Forecast of PM2.5 Levels for Next 12 Months",
         x = "Date",
         y = "PM2.5 (µg/m³)") +
    theme_minimal()
}
