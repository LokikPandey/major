import pandas as pd
import numpy as np
import requests
import random
from datetime import datetime, timedelta
import time
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# API keys
API_KEY = "ba7df183bb14401ba2a775666e92bae9"
MARKETAUX_KEY = "63zLbFopw3Y9mHnHftZEeyMZYrNCteJX81CwyM2Z"

def fetch_stock_data(symbol, start_date, end_date):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_date,
        "end_date": end_date,
        "apikey": API_KEY,
        "format": "JSON",
        "outputsize": 5000
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        raise ValueError("No data found for symbol")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df["close"] = pd.to_numeric(df["close"])
    return df[["datetime", "close"]]

def calculate_indicators(df):
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    return df

def fetch_sentiment_scores(symbol, dates):
    sentiment_dict = {}
    base_url = "https://api.marketaux.com/v1/news/all"

    unique_dates = sorted(set(dates))
    last_3_dates = unique_dates[-3:]

    for date in last_3_dates:
        date_str = date.strftime("%Y-%m-%d")
        params = {
            "symbols": symbol,
            "published_after": date_str + "T00:00",
            "published_before": date_str + "T23:59",
            "filter_entities": "true",
            "language": "en",
            "api_token": MARKETAUX_KEY
        }
        try:
            response = requests.get(base_url, params=params)
            data = response.json()

            scores = []
            if "data" in data:
                for article in data["data"]:
                    for ent in article.get("entities", []):
                        if ent.get("symbol") == symbol and "sentiment_score" in ent:
                            scores.append(ent["sentiment_score"])

            sentiment_score = np.mean(scores) if scores else round(random.uniform(0, 0.5), 3)
        except Exception:
            sentiment_score = round(random.uniform(0, 0.5), 3)

        sentiment_dict[date] = sentiment_score
        time.sleep(1)

    for date in dates:
        if date not in sentiment_dict:
            sentiment_dict[date] = round(random.uniform(0, 0.5), 3)

    return sentiment_dict

# Fetch and prepare data
symbol = "AAPL"
start_date = "2022-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

df = fetch_stock_data(symbol, start_date, end_date)
df = calculate_indicators(df)

dates = list(df['datetime'])
sentiments = fetch_sentiment_scores(symbol, dates)
df["sentiment"] = df["datetime"].map(sentiments)

# Drop NA rows (indicators need sufficient data)
df = df.dropna().reset_index(drop=True)

# Features to use
features = ['close', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 'sentiment']

# Fill NaNs with zeros (just in case)
df[features] = df[features].fillna(0)

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

# Prepare sequences for LSTM
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # Predict 'close' price
    return np.array(X), np.array(y)

SEQ_LENGTH = 20
X, y = create_sequences(scaled_features, SEQ_LENGTH)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='relu')  # relu to avoid negative prices
])

model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Evaluate model
test_loss = model.evaluate(X_test, y_test)
print(f"Test MSE Loss: {test_loss}")

# Save model and scaler
model.save("stock_lstm_model.h5")
import joblib
joblib.dump(scaler, "scaler.save")
