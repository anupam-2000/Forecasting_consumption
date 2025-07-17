import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from tcn import TCN

# --- UI Setup ---
st.set_page_config(page_title="Power Forecast (TCN)", layout="wide")
st.title("⚡ Power Demand Forecast using TCN")
st.write("Forecast hourly PJMW power demand for the next N days (up to 30) using a Temporal Convolutional Network (TCN).")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_hourly.csv", parse_dates=['Datetime'])
    df.rename(columns={'Datetime': 'date&time'}, inplace=True)
    df.set_index('date&time', inplace=True)
    df = df.sort_index()
    df = df[df.index >= df.index.max() - pd.DateOffset(years=4)]
    df['PJMW_MW'] = df['PJMW_MW'].fillna(method='ffill').astype(int)
    return df

df = load_data()

# --- Sidebar Controls ---
forecast_days = st.slider("🔁 Select forecast horizon (days)", min_value=1, max_value=30, value=30)
OUTPUT_HOURS = forecast_days * 24
INPUT_HOURS = 168  # Past 1 week

# --- Prepare data for TCN ---
st.subheader("🔧 Training TCN Model")
scaler = StandardScaler()
scaled_series = scaler.fit_transform(df[["PJMW_MW"]]).flatten()

# Include last 4 years plot
st.subheader("📉 Last 4 Years of Hourly Power Demand")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df.index, y=df['PJMW_MW'], mode='lines', name='PJMW_MW'))
fig_hist.update_layout(title="Hourly Power Demand (Last 4 Years)", xaxis_title="Date", yaxis_title="Power (MW)")
st.plotly_chart(fig_hist, use_container_width=True)

X, y = [], []
for i in range(len(scaled_series) - INPUT_HOURS - OUTPUT_HOURS + 1):
    X.append(scaled_series[i:i + INPUT_HOURS])
    y.append(scaled_series[i + INPUT_HOURS:i + INPUT_HOURS + OUTPUT_HOURS])

# Force model to train on latest sample
X.append(scaled_series[-INPUT_HOURS:])
y.append(scaled_series[-OUTPUT_HOURS:])

X = np.array(X).reshape(-1, INPUT_HOURS, 1)
y = np.array(y).reshape(-1, OUTPUT_HOURS, 1)

# --- Build & train TCN model ---
model = Sequential([
    TCN(input_shape=(INPUT_HOURS, 1), nb_filters=64, dilations=[1, 2, 4, 8], return_sequences=False),
    Dense(OUTPUT_HOURS),
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y.reshape(-1, OUTPUT_HOURS), epochs=5, batch_size=32, verbose=1)

# --- Forecast ---
st.subheader(f"🔮 Forecast for Next {forecast_days} Day(s) using TCN")
last_input = scaled_series[-INPUT_HOURS:].reshape(1, INPUT_HOURS, 1)
pred_scaled = model.predict(last_input)[0]
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# --- Forecast output ---
last_timestamp = df.index[-1]
future_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=OUTPUT_HOURS, freq='H')
forecast_series = pd.Series(pred, index=future_index, name='Forecast_PJMW_MW')

# --- Plot forecast ---
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df.index[-7*24:], y=df['PJMW_MW'].iloc[-7*24:], mode='lines', name='Last 7 Days'))
fig_forecast.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines', name='TCN Forecast'))
fig_forecast.update_layout(title="TCN Forecast (Hourly for {} Days)".format(forecast_days), xaxis_title="Date", yaxis_title="Power (MW)")
st.plotly_chart(fig_forecast, use_container_width=True)

# --- Show forecasted values ---
st.subheader("📈 Forecasted Hourly Series")
st.dataframe(forecast_series.reset_index().rename(columns={"index": "Datetime"}), use_container_width=True)

# --- Daily Summary ---
forecast_df = forecast_series.reset_index().rename(columns={"index": "Datetime"})
forecast_df['Date'] = forecast_df['Datetime'].dt.date
daily_summary = forecast_df.groupby('Date')['Forecast_PJMW_MW'].agg(['mean', 'std']).reset_index()
daily_summary.columns = ['Date', 'Average_MW', 'Std_Dev_MW']

st.subheader("📊 Daily Average and Std. Dev. of Forecast")
st.dataframe(daily_summary, use_container_width=True)

# --- Download ---
csv = forecast_df.to_csv(index=False)
st.download_button("📥 Download Forecast CSV", data=csv, file_name="tcn_forecast.csv", mime="text/csv")
