# Import necessary libraries
import yfinance as yf  # Library for fetching historical stock data
import numpy as np  # Library for numerical computations
import pandas as pd  # Library for data manipulation
import matplotlib.pyplot as plt  # Library for data visualization
from sklearn.preprocessing import MinMaxScaler  # Library for data normalization
from tensorflow.keras.models import Sequential  # Keras sequential model API
from tensorflow.keras.layers import LSTM, Dense  # LSTM and Dense layers for building the model

# Fetch historical stock data
symbol = "AAPL"
start_date = "2022-01-01"
end_date = "2024-01-01"
data = yf.download(symbol, start=start_date, end=end_date)

# Use only the 'Close' prices for simplicity
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Function to prepare data for LSTM
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        a = data[i:(i + time_steps), 0]
        X.append(a)
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Set the number of time steps
time_steps = 10

# Prepare the training data
X, y = prepare_data(scaled_data, time_steps)

# Reshape the data for LSTM input
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions on the test set
predictions = model.predict(X_test)

# Invert the normalization for better visualization
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize the predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index[split_index + time_steps:], y_test, label='Actual Prices')
plt.plot(data.index[split_index + time_steps:], predictions, label='Predicted Prices', linestyle='--')
plt.title(f'Stock Price Prediction for {symbol}')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Create a DataFrame for predictions
predicted_df = pd.DataFrame({
    'Date': data.index[split_index + time_steps:],
    'Actual Close Price': y_test.flatten(),
    'Predicted Close Price': predictions.flatten()
})

# Display predictions in tabular format
print(predicted_df)