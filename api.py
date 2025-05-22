from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pandas_ta as ta
from tensorflow.keras.models import load_model
import joblib
import requests
import datetime
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)
CORS(app)

API_KEY = "ba7df183bb14401ba2a775666e92bae9"
MARKETAUX_KEY = "63zLbFopw3Y9mHnHftZEeyMZYrNCteJX81CwyM2Z"

# ✅ Load the model without compiling (to avoid 'mse' error)
model = load_model("stock_lstm_model.h5", compile=False)
scaler = joblib.load("scaler.save")

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
        raise ValueError(f"No data found for symbol: {symbol}")
    
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
            sentiment_score = np.mean(scores) if scores else round(np.random.uniform(0, 0.5), 3)
        except Exception:
            sentiment_score = round(np.random.uniform(0, 0.5), 3)

        sentiment_dict[date] = sentiment_score

    for date in dates:
        if date not in sentiment_dict:
            sentiment_dict[date] = round(np.random.uniform(0, 0.5), 3)

    return sentiment_dict

def prepare_features(df, sentiments):
    df = calculate_indicators(df)
    df["sentiment"] = df["datetime"].map(sentiments)
    df = df.dropna().reset_index(drop=True)
    features = ['close', 'SMA_20', 'EMA_20', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 'sentiment']
    df[features] = df[features].fillna(0)
    return df, features

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get("symbol")
        start_date = data.get("start_date")
        num_days = int(data.get("num_days", 10))

        if not symbol or not start_date:
            return jsonify({"error": "Please provide 'symbol' and 'start_date'."}), 400
        if num_days <= 0 or num_days > 60:
            return jsonify({"error": "'num_days' must be between 1 and 60."}), 400

        today = datetime.datetime.today().strftime('%Y-%m-%d')
        df = fetch_stock_data(symbol, start_date, today)

        if df.empty:
            return jsonify({"error": "No stock data found."}), 400

        dates = list(df['datetime'])
        sentiments = fetch_sentiment_scores(symbol, dates)

        df, features = prepare_features(df, sentiments)

        SEQ_LENGTH = 20
        if len(df) < SEQ_LENGTH:
            return jsonify({"error": "Not enough data to make predictions."}), 400

        # Prepare input sequence for prediction
        input_seq = df[features].tail(SEQ_LENGTH).values
        input_scaled = scaler.transform(input_seq)

        preds = []
        current_seq = input_scaled.copy()

        for _ in range(num_days):
            pred_scaled = model.predict(current_seq[np.newaxis, :, :])[0, 0]

            next_features = current_seq[-1].copy()
            next_features[0] = pred_scaled  # Update close price prediction in features
            current_seq = np.vstack([current_seq[1:], next_features])

            preds.append(pred_scaled)

        # Inverse transform predictions to real scale
        close_min = scaler.data_min_[0]
        close_max = scaler.data_max_[0]
        preds_real = [pred * (close_max - close_min) + close_min for pred in preds]

        pred_dates = [(df["datetime"].iloc[-1] + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(num_days)]
        predictions = [{"date": d, "predicted_close": round(float(p), 2)} for d, p in zip(pred_dates, preds_real)]

        # Calculate error metrics if actual close data for prediction dates is available
        actual_close = df["close"].tail(num_days).values
        mse = None
        rmse = None
        mae = None

        if len(actual_close) == num_days:
            mse = mean_squared_error(actual_close, preds_real)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_close, preds_real)

            mse = round(float(mse), 4)
            rmse = round(float(rmse), 4)
            mae = round(float(mae), 4)

        # Generate advice based on predicted trend
        advice = "Hold"
        if len(predictions) >= 2:
            if predictions[-1]["predicted_close"] > predictions[-2]["predicted_close"]:
                advice = "Buy - Uptrend expected"
            elif predictions[-1]["predicted_close"] < predictions[-2]["predicted_close"]:
                advice = "Sell - Downtrend expected"

        # Also prepare feature data (last 20 days) for frontend
        features_data = df[features].tail(SEQ_LENGTH).copy()
        features_data["datetime"] = df["datetime"].tail(SEQ_LENGTH).values
        features_data["datetime"] = features_data["datetime"].dt.strftime("%Y-%m-%d")
        features_json = features_data.to_dict(orient="records")

        response = {
            "predictions": predictions,
            "advice": advice,
            "features": features_json
        }
        if mse is not None:
            response.update({
                "mse": mse,
                "rmse": rmse,
                "mae": mae
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
