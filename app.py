import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(
    page_title="Udupi Shevanti Weather-Price Intelligence (Mid MVP)",
    page_icon="🌸",
    layout="wide"
)

# App Header
st.title("🌸 Weather-Induced Shevanti Price Volatility Engine")
st.caption("Mid-Project Review Prototype (Phases 1–7 Completed)")
st.markdown("---")

# Helper function to load data
@st.cache_data
def load_data():
    df_daily = pd.read_csv("feature_dataset_v1.csv")
    df_daily['DATE'] = pd.to_datetime(df_daily['DATE'])
    
    try:
        sig_ts = pd.read_csv("weather_signature_timeseries.csv")
        sig_db = pd.read_csv("Phase_6_Weather_Signature_Database.csv")
    except FileNotFoundError:
        sig_ts, sig_db = None, None
        
    try:
        ml_results = pd.read_csv("Phase_7_ML_Experiment_Results.csv")
    except FileNotFoundError:
        ml_results = None
        
    return df_daily, sig_ts, sig_db, ml_results

df_daily, sig_ts, sig_db, ml_results = load_data()

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select View:", [
    "1. Historical Explorer (Phase 1-2)",
    "2. Weather Event Signatures (Phase 4-6)",
    "3. Live Price Estimator (Phase 7 ML)",
    "4. Model Performance Comparison (Phase 7)"
])

# ==============================================================================
# TAB 1: HISTORICAL EXPLORER
# ==============================================================================
if page == "1. Historical Explorer (Phase 1-2)":
    st.header("📊 Phase 1 & 2: Historical Weather & Price Explorer")
    st.write("Explore raw and merged daily Shevanti flower prices alongside local weather indicators.")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df_daily['DATE'].min())
    with col2:
        end_date = st.date_input("End Date", df_daily['DATE'].max())
        
    filtered_df = df_daily[(df_daily['DATE'] >= pd.to_datetime(start_date)) & 
                           (df_daily['DATE'] <= pd.to_datetime(end_date))]
    
    # Dual-axis Price vs Rainfall Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['DATE'], y=filtered_df['PRICE_INR'], name="Price (₹/kg)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Bar(x=filtered_df['DATE'], y=filtered_df['RAINFALL_MM'], name="Rainfall (mm)", yaxis="y2", opacity=0.4, marker_color="blue"))
    
    fig.update_layout(
        title="Shevanti Price Trend vs Daily Rainfall",
        yaxis=dict(title="Price (₹)"),
        yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right"),
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df[['DATE', 'PRICE_INR', 'RAINFALL_MM', 'MAX_TEMP_C', 'HUMIDITY_PERCENT']].head(50), use_container_width=True)

# ==============================================================================
# TAB 2: WEATHER EVENT SIGNATURES
# ==============================================================================
elif page == "2. Weather Event Signatures (Phase 4-6)":
    st.header("⚡ Phase 4, 5 & 6: Weather Price Signatures")
    st.write("Normalized price response patterns across 15-day event windows (-7 days to +7 days).")
    
    if sig_ts is not None and sig_db is not None:
        # Signature Line Plot
        event_options = [c for c in sig_ts.columns if c != 'RELATIVE_DAY']
        selected_events = st.multiselect("Select Weather Events to Compare:", event_options, default=event_options)
        
        fig = go.Figure()
        for ev in selected_events:
            fig.add_trace(go.Scatter(x=sig_ts['RELATIVE_DAY'], y=sig_ts[ev], mode='lines+markers', name=ev))
            
        fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Event Day (0)")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        
        fig.update_layout(
            title="Comparative Weather Price Signatures (% Price Change from Baseline)",
            xaxis_title="Relative Days (0 = Event Day)",
            yaxis_title="Normalized Price Change (%)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Weather Signature Database")
        st.dataframe(sig_db, use_container_width=True)
    else:
        st.warning("Signature database files not found. Run Phase 5 and Phase 6 scripts first.")

# ==============================================================================
# TAB 3: LIVE PRICE ESTIMATOR
# ==============================================================================
elif page == "3. Live Price Estimator (Phase 7 ML)":
    st.header("🤖 Phase 7: Live Shevanti Price Estimator")
    st.write("Enter recent market prices and short-term weather conditions to estimate today's wholesale price.")
    
    # Quick train of RF model on the fly for demo responsiveness
    price_cols = ['PRICE_LAG_1', 'PRICE_LAG_2', 'PRICE_LAG_3', 'PRICE_LAG_7', 'PRICE_3D_MEAN', 'PRICE_7D_MEAN']
    weather_cols = ['RAIN_LAG_1', 'MAX_TEMP_LAG_1', 'HUMIDITY_LAG_1', 'RAIN_3D_SUM', 'RAIN_7D_SUM', 'CONSEC_RAIN_DAYS_LAG1', 'FLAG_HEAVY_RAIN', 'FLAG_HOT_DAY', 'FLAG_HUMID_DAY']
    
    X = df_daily[price_cols + weather_cols]
    y = df_daily['PRICE_INR']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Recent Price Indicators")
        p_lag1 = st.number_input("Yesterday's Price (₹):", min_value=10.0, max_value=2000.0, value=250.0)
        p_lag2 = st.number_input("Price 2 Days Ago (₹):", min_value=10.0, max_value=2000.0, value=245.0)
        p_lag3 = st.number_input("Price 3 Days Ago (₹):", min_value=10.0, max_value=2000.0, value=240.0)
        p_lag7 = st.number_input("Price 7 Days Ago (₹):", min_value=10.0, max_value=2000.0, value=230.0)
        
    with col2:
        st.subheader("2. Recent Weather Indicators")
        rain_lag1 = st.number_input("Yesterday's Rainfall (mm):", min_value=0.0, max_value=300.0, value=45.0)
        max_temp1 = st.number_input("Yesterday's Max Temp (°C):", min_value=15.0, max_value=45.0, value=31.0)
        humid1 = st.number_input("Yesterday's Humidity (%):", min_value=20.0, max_value=100.0, value=88.0)
        rain_3d_sum = st.number_input("3-Day Total Rainfall (mm):", min_value=0.0, max_value=800.0, value=110.0)
        consec_rain = st.number_input("Consecutive Rainy Days Streak:", min_value=0, max_value=30, value=4)

    # Derived inputs
    p_3d_mean = np.mean([p_lag1, p_lag2, p_lag3])
    p_7d_mean = np.mean([p_lag1, p_lag2, p_lag3, p_lag7])
    flag_heavy = 1 if rain_lag1 >= 30.0 else 0
    flag_hot = 1 if max_temp1 >= 34.0 else 0
    flag_humid = 1 if humid1 >= 90.0 else 0

    input_data = pd.DataFrame([{
        'PRICE_LAG_1': p_lag1, 'PRICE_LAG_2': p_lag2, 'PRICE_LAG_3': p_lag3, 'PRICE_LAG_7': p_lag7,
        'PRICE_3D_MEAN': round(p_3d_mean, 2), 'PRICE_7D_MEAN': round(p_7d_mean, 2),
        'RAIN_LAG_1': rain_lag1, 'MAX_TEMP_LAG_1': max_temp1, 'HUMIDITY_LAG_1': humid1,
        'RAIN_3D_SUM': rain_3d_sum, 'RAIN_7D_SUM': rain_3d_sum * 1.5,
        'CONSEC_RAIN_DAYS_LAG1': consec_rain,
        'FLAG_HEAVY_RAIN': flag_heavy, 'FLAG_HOT_DAY': flag_hot, 'FLAG_HUMID_DAY': flag_humid
    }])

    st.markdown("---")
    if st.button("🔮 Calculate Estimated Today's Price", type="primary"):
        prediction = model.predict(input_data)[0]
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Estimated Price Today", f"₹ {prediction:.2f} / kg", delta=f"{prediction - p_lag1:+.2f} ₹ vs yesterday")
        with res_col2:
            st.metric("Estimated Price Range (±10%)", f"₹ {prediction*0.9:.2f} - ₹ {prediction*1.1:.2f}")

# ==============================================================================
# TAB 4: MODEL PERFORMANCE COMPARISON
# ==============================================================================
elif page == "4. Model Performance Comparison (Phase 7)":
    st.header("📈 Phase 7: Model Experimentation Results")
    st.write("Evaluating whether incorporating weather data reduces price prediction error.")
    
    if ml_results is not None:
        st.subheader("Model Performance Comparison Table")
        st.dataframe(ml_results, use_container_width=True)
        
        # Bar chart comparing MAE across models
        fig = px.bar(
            ml_results, 
            x='Experiment', 
            y='MAE (₹)', 
            color='Algorithm', 
            barmode='group',
            title="Mean Absolute Error (MAE) Comparison (Lower is Better)",
            text='MAE (₹)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Phase 7 ML results CSV not found. Run Phase 7 script first.")