import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import requests
from flask import Flask, Response, json
import logging
from datetime import datetime
import joblib  # For loading the scaler

app = Flask(__name__)

# Configure basic logging without JSON formatting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Define the BiLSTM model architecture matching the training configuration
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Initialize the model with the same architecture as during training
# Replace the parameters below with those used during training
input_size = 3  # Example: normalized_close_SOLUSDT, hour, day_of_week
hidden_size = 128
num_layers = 2
output_size = 1
dropout = 0.2

model = BiLSTM(input_size=input_size, hidden_size=hidden_size, 
              num_layers=num_layers, output_size=output_size, dropout=dropout)

# Load the trained model weights
# Ensure that 'bilstm_solusdt.pth' is in the same directory as this script
model.load_state_dict(torch.load("bilstm_solusdt.pth", map_location=torch.device('cpu')))
model.eval()

# Load the scaler used during training
# Ensure that 'scaler.pkl' was saved during training and is in the same directory
scaler = joblib.load("scaler.pkl")

# Function to fetch historical data from Binance
def get_binance_url(symbol="SOLUSDT", interval="1m", limit=60):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

@app.route("/inference/<string:token>")
def get_inference(token):
    if model is None:
        return Response(json.dumps({"error": "Model is not available"}), status=500, mimetype='application/json')

    symbol_map = {
        'ETH': 'ETHUSDT',
        'BTC': 'BTCUSDT',
        'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT',
        'ARB': 'ARBUSDT'
    }

    token = token.upper()
    if token in symbol_map:
        symbol = symbol_map[token]
    else:
        return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

    url = get_binance_url(symbol=symbol)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)

        # Adjust the number of rows based on the symbol
        if symbol in ['BTCUSDT', 'SOLUSDT']:
            df = df.tail(60)  # Use last 60 minutes of data (assuming sequence_length=60)
        else:
            df = df.tail(60)  # Adjust as needed

        # Log the current price and the timestamp
        current_price = df.iloc[-1]["price"]
        current_time = df.iloc[-1]["date"]
        logger.info(f"Current Price: {current_price} at {current_time}")

        # Prepare data for the BiLSTM model
        # Assuming the training used features: normalized_close_SOLUSDT, hour, day_of_week
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month  # If month was used during training

        # Select the features used during training
        feature_columns = ['price', 'hour', 'day_of_week']  # Adjust if more features were used
        features = df[feature_columns].values

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features)

        # Create the input sequence
        seq = torch.FloatTensor(scaled_features).unsqueeze(0)  # Shape: (1, sequence_length, input_size)

        # Make prediction
        with torch.no_grad():
            y_pred = model(seq)

        # Inverse transform the prediction to get the actual price
        # Only inverse transform the 'price' feature
        predicted_price_scaled = y_pred.numpy().reshape(-1, 1)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

        # Log the prediction
        logger.info(f"Prediction: {predicted_price}")

        # Return only the predicted price in JSON response
        return Response(json.dumps({"predicted_price": round(float(predicted_price), 2)}), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}), 
                        status=response.status_code, 
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
