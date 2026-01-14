import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Forecasting libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Page config
st.set_page_config(
    page_title="Energy Weather Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('processed_energy_weather.csv')
    df['time'] = pd.to_datetime(df['time'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'processed_energy_weather.csv' not found!")
    st.stop()

# Sidebar
st.sidebar.title("📊 Control Panel")

# Date range filter
date_min = df['time'].min()
date_max = df['time'].max()
date_range = st.sidebar.date_input(
    "Select date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date()
)

# Filter data
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['time'].dt.date >= start_date) & (df['time'].dt.date <= end_date)
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()

# Main title
st.title("⚡ Energy Weather Analytics Dashboard")
st.markdown("---")

# Key metrics
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Average Load (MW)", f"{filtered_df['total load actual'].mean():,.0f}")

with col2:
    st.metric("Maximum Load (MW)", f"{filtered_df['total load actual'].max():,.0f}")

with col3:
    st.metric("Minimum Load (MW)", f"{filtered_df['total load actual'].min():,.0f}")

with col4:
    st.metric("Data Points", f"{len(filtered_df):,}")

st.markdown("---")

# Section 1: Time Series Charts
st.subheader("📈 Time Series Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Energy Load Trend**")
    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=filtered_df['time'],
        y=filtered_df['total load actual'],
        mode='lines',
        name='Total Load',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy'
    ))
    fig_load.update_layout(
        xaxis_title='Time',
        yaxis_title='Load (MW)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_load, use_container_width=True)

with col2:
    st.markdown("**Average Temperature Trend**")
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=filtered_df['time'],
        y=filtered_df['temp_avg'],
        mode='lines',
        name='Average Temperature',
        line=dict(color='#4ECDC4', width=2),
        fill='tozeroy'
    ))
    fig_temp.update_layout(
        xaxis_title='Time',
        yaxis_title='Temperature (K)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_temp, use_container_width=True)

st.markdown("---")

# Section 2: City Temperature Selection
st.subheader("🏙️ City Temperature Analysis")

cities = {
    'Barcelona': 'temp_ Barcelona',
    'Bilbao': 'temp_Bilbao',
    'Madrid': 'temp_Madrid',
    'Seville': 'temp_Seville',
    'Valencia': 'temp_Valencia'
}

# City selector in sidebar
st.sidebar.markdown("**City Selection**")
selected_cities = st.sidebar.multiselect(
    "Select cities to display",
    options=list(cities.keys()),
    default=['Barcelona', 'Madrid']
)

if selected_cities:
    fig_cities = go.Figure()
    for city in selected_cities:
        col_name = cities[city]
        fig_cities.add_trace(go.Scatter(
            x=filtered_df['time'],
            y=filtered_df[col_name],
            mode='lines',
            name=city,
            line=dict(width=2)
        ))
    
    fig_cities.update_layout(
        xaxis_title='Time',
        yaxis_title='Temperature (K)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig_cities, use_container_width=True)

st.markdown("---")

# Section 3: Correlation Analysis
st.subheader("📊 Load vs Temperature Correlation")

fig_correlation = go.Figure()
fig_correlation.add_trace(go.Scatter(
    x=filtered_df['temp_avg'],
    y=filtered_df['total load actual'],
    mode='markers',
    marker=dict(
        size=4,
        color=filtered_df['temp_avg'],
        colorscale='Viridis',
        showscale=True
    ),
    name='Load vs Temp'
))

fig_correlation.update_layout(
    xaxis_title='Average Temperature (K)',
    yaxis_title='Load (MW)',
    hovermode='closest',
    template='plotly_white',
    height=400
)
st.plotly_chart(fig_correlation, use_container_width=True)

st.markdown("---")

# Section 4: Hourly and Daily Analysis
st.subheader("📅 Hourly & Daily Patterns")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average Load by Hour**")
    hourly_avg = filtered_df.groupby('hour')['total load actual'].mean().reset_index()
    fig_hourly = px.bar(
        hourly_avg, 
        x='hour', 
        y='total load actual',
        labels={'hour': 'Hour', 'total load actual': 'Average Load (MW)'},
        color='total load actual',
        color_continuous_scale='Blues'
    )
    fig_hourly.update_layout(template='plotly_white', height=400, showlegend=False)
    st.plotly_chart(fig_hourly, use_container_width=True)

with col2:
    st.markdown("**Average Load by Day of Week**")
    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    daily_avg = filtered_df.groupby('day_of_week')['total load actual'].mean().reset_index()
    daily_avg['day_name'] = daily_avg['day_of_week'].map(day_names)
    fig_daily = px.bar(
        daily_avg, 
        x='day_name', 
        y='total load actual',
        labels={'day_name': 'Day', 'total load actual': 'Average Load (MW)'},
        color='total load actual',
        color_continuous_scale='Reds'
    )
    fig_daily.update_layout(template='plotly_white', height=400, showlegend=False)
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("---")

# # Section 5: Data Table
# st.subheader("📋 Data Table")

# show_data = st.sidebar.checkbox("Show raw data", value=False)

# if show_data:
#     st.dataframe(filtered_df, use_container_width=True)
    
#     # Download filtered data
#     csv_data = filtered_df.to_csv(index=False)
#     st.download_button(
#         label="📥 Download Filtered Data (CSV)",
#         data=csv_data,
#         file_name=f"energy_weather_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
#         mime="text/csv",
#         help="Download the filtered dataset as CSV"
#     )

# Statistics Summary
st.sidebar.markdown("---")
with st.sidebar.expander("📈 Statistics Summary"):
    st.write(f"**Date Range:** {date_min.date()} to {date_max.date()}")
    st.write(f"**Selected Records:** {len(filtered_df)}")
    st.write(f"\n**Energy Load Statistics:**")
    st.write(f"- Min: {filtered_df['total load actual'].min():,.0f} MW")
    st.write(f"- Avg: {filtered_df['total load actual'].mean():,.0f} MW")
    st.write(f"- Max: {filtered_df['total load actual'].max():,.0f} MW")
    st.write(f"\n**Temperature Statistics:**")
    st.write(f"- Min: {filtered_df['temp_avg'].min():.2f} K")
    st.write(f"- Avg: {filtered_df['temp_avg'].mean():.2f} K")
    st.write(f"- Max: {filtered_df['temp_avg'].max():.2f} K")

st.markdown("---")
st.markdown("---")

# Section 6: Forecasting
st.title("🔮 Energy Load Forecasting")
st.write("Predict future energy load using SARIMAX and LSTM models")

# Forecast settings
st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Forecast Settings**")

model_choice = st.sidebar.radio(
    "Select forecasting model",
    options=["SARIMAX", "LSTM"],
    index=0,
    help="Choose which model to use for forecasting"
)

forecast_days = st.sidebar.selectbox(
    "Select forecast horizon",
    options=[1, 3, 7],
    index=0,
    help="Number of days to forecast ahead"
)

train_ratio = st.sidebar.slider(
    "Train/Test Split",
    min_value=0.7,
    max_value=0.9,
    value=0.8,
    step=0.05,
    help="Proportion of data used for training"
)

run_forecast = st.sidebar.button("🚀 Run Forecast", type="primary")

if run_forecast:
    with st.spinner(f"Running {model_choice} model for {forecast_days}-day forecast... Please wait..."):
        
        # Prepare data
        data = df['total load actual'].values.astype(float)
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        steps = forecast_days * 24  # Convert days to hours
        
        st.subheader(f"📊 {forecast_days}-Day Forecast Results ({steps} hours)")
        st.markdown(f"**Model:** {model_choice}")
        
        forecast_result = None
        model_metrics = {}
        
        # Run selected model
        if model_choice == "SARIMAX":
            st.markdown("### 📈 SARIMAX Model")
            
            try:
                with st.spinner("Loading SARIMAX model..."):
                    # Try multiple loading strategies to handle different save formats/compression
                    results = None
                    load_errors = []

                    # 1) statsmodels native loader
                    try:
                        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
                        results = SARIMAXResults.load('final_sarimax_model.pkl')
                    except Exception as e:
                        load_errors.append(f"statsmodels.load: {str(e)[:120]}")

                    # 2) joblib (often used with compression)
                    if results is None:
                        try:
                            import joblib
                            results = joblib.load('final_sarimax_model.pkl')
                        except Exception as e:
                            load_errors.append(f"joblib.load: {str(e)[:120]}")

                    # 3) gzip + pickle (if file is gzipped but named .pkl)
                    if results is None:
                        try:
                            import gzip, pickle
                            with gzip.open('final_sarimax_model.pkl', 'rb') as f:
                                results = pickle.load(f)
                        except Exception as e:
                            load_errors.append(f"gzip+pickle: {str(e)[:120]}")

                    # 4) plain pickle as fallback
                    if results is None:
                        try:
                            import pickle
                            with open('final_sarimax_model.pkl', 'rb') as f:
                                results = pickle.load(f)
                        except Exception as e:
                            load_errors.append(f"pickle.load: {str(e)[:120]}")

                    if results is None:
                        raise RuntimeError("Unable to load SARIMAX model. Tried statsmodels, joblib, gzip+pickle, and pickle.\n" + " | ".join(load_errors))

                    # Forecast (handle exogenous regressors if present)
                    if hasattr(results, 'get_forecast'):
                        try:
                            k_exog = int(getattr(results.model, 'k_exog', 0) or 0)
                        except Exception:
                            k_exog = 0

                        exog_future = None
                        if k_exog > 0:
                            # Attempt to construct future exog based on model's exog names
                            try:
                                exog_names = getattr(results.model, 'exog_names', None)
                                # Normalize names and detect constant
                                const_names = {"const", "intercept"}
                                add_const = False
                                cols_to_use = []

                                if isinstance(exog_names, (list, tuple)) and len(exog_names) == k_exog:
                                    for nm in exog_names:
                                        if isinstance(nm, str) and nm.lower() in const_names:
                                            add_const = True
                                        else:
                                            cols_to_use.append(nm)

                                    # Validate columns exist
                                    missing = [c for c in cols_to_use if c not in df.columns]
                                    if missing:
                                        raise KeyError(f"Missing exogenous columns in data: {missing}")

                                    # Slice future exog window
                                    exog_df = df.loc[split_idx: split_idx + steps - 1, cols_to_use].astype(float)

                                    if add_const:
                                        import statsmodels.api as sm
                                        exog_df = sm.add_constant(exog_df, has_constant='add')
                                        # Reorder to match original exog names
                                        exog_df = exog_df[[c if isinstance(c, str) else c for c in exog_names]]

                                    exog_future = exog_df
                                else:
                                    # Fallback: infer by taking first k_exog numeric columns excluding target/time
                                    numeric_cols = [
                                        c for c in df.columns
                                        if c not in ['time', 'total load actual'] and np.issubdtype(df[c].dtype, np.number)
                                    ]
                                    if len(numeric_cols) < k_exog:
                                        raise ValueError(
                                            f"Model expects {k_exog} exogenous features but only found {len(numeric_cols)} numeric columns."
                                        )
                                    inferred_cols = numeric_cols[:k_exog]
                                    exog_future = df.loc[split_idx: split_idx + steps - 1, inferred_cols].astype(float)
                                    st.warning(f"Using inferred exogenous columns for SARIMAX: {inferred_cols}")

                            except Exception as ex:
                                raise RuntimeError(
                                    f"Unable to build exogenous inputs for forecasting. Details: {str(ex)[:200]}"
                                )

                        # Run forecast
                        forecast = results.get_forecast(steps=steps, exog=exog_future) if k_exog > 0 else results.get_forecast(steps=steps)
                        forecast_result = forecast.predicted_mean.values
                    else:
                        raise TypeError("Loaded object does not support get_forecast(). Ensure the saved file is a fitted SARIMAX results object.")
                    
                    # Calculate metrics if test data available
                    if len(test_data) >= steps:
                        model_metrics['RMSE'] = np.sqrt(mean_squared_error(test_data[:steps], forecast_result))
                        model_metrics['MAE'] = mean_absolute_error(test_data[:steps], forecast_result)
                        model_metrics['MAPE'] = np.mean(np.abs((test_data[:steps] - forecast_result) / test_data[:steps])) * 100
                    
                    st.success("✅ SARIMAX model completed successfully!")
                    
            except ImportError:
                st.error("❌ statsmodels not installed. Run: pip install statsmodels")
            except Exception as e:
                st.error(f"❌ SARIMAX error: {str(e)[:200]}")
        
        elif model_choice == "LSTM":
            st.markdown("### 🧠 LSTM Model")
            
            try:
                import tensorflow as tf
                from tensorflow.keras.models import load_model
                
                with st.spinner("Loading LSTM model..."):
                    # Load pre-trained LSTM model (no compile to bypass metrics deserialization)
                    model = load_model('final_lstm_model.h5', compile=False)
                    
                    # Infer lookback (timesteps) from model input shape when possible
                    try:
                        inferred_lookback = model.input_shape[1] if hasattr(model, 'input_shape') and model.input_shape and len(model.input_shape) >= 3 else None
                    except Exception:
                        inferred_lookback = None
                    lookback = int(inferred_lookback) if inferred_lookback and inferred_lookback > 0 else 24
                    
                    # Determine expected number of features from the model
                    try:
                        expected_n_features = model.input_shape[-1] if hasattr(model, 'input_shape') else 1
                        if expected_n_features is None:
                            expected_n_features = 1
                    except Exception:
                        expected_n_features = 1

                    # Build feature list to match the model's expected input features
                    candidate_features = [
                        'total load actual',
                        'temp_avg',
                        'temp_Barcelona',
                        'temp_Bilbao',
                        'temp_Madrid',
                        'temp_Seville',
                        'temp_Valencia'
                    ]
                    available_features = [c for c in candidate_features if c in df.columns]

                    if 'total load actual' not in available_features:
                        raise KeyError("Required column 'total load actual' not found in dataset.")

                    feature_list = []
                    # Ensure target is the first feature
                    feature_list.append('total load actual')
                    for c in available_features:
                        if c != 'total load actual' and len(feature_list) < int(expected_n_features):
                            feature_list.append(c)

                    # If not enough features available, pad by duplicating the target to satisfy shape
                    if len(feature_list) < int(expected_n_features):
                        feature_list += ['total load actual'] * (int(expected_n_features) - len(feature_list))

                    # Prepare features for scaling and sequence construction
                    features_df = df[feature_list].astype(float)
                    train_features = features_df.iloc[:split_idx].values

                    if int(expected_n_features) == 1:
                        # Scale single feature and forecast iteratively
                        x_scaler = MinMaxScaler(feature_range=(0, 1))
                        train_x_scaled = x_scaler.fit_transform(train_features)

                        last_seq = train_x_scaled[-lookback:, :].copy()  # (lookback, 1)
                        current_seq = last_seq.copy()
                        lstm_forecast_scaled = []

                        for _ in range(steps):
                            pred_scaled = model.predict(current_seq.reshape(1, lookback, 1), verbose=0)[0, 0]
                            lstm_forecast_scaled.append(pred_scaled)
                            next_vec = np.array([pred_scaled]).reshape(1, -1)  # (1,1)
                            current_seq = np.vstack([current_seq[1:], next_vec])

                        forecast_result = x_scaler.inverse_transform(
                            np.array(lstm_forecast_scaled).reshape(-1, 1)
                        ).flatten()
                    else:
                        # Scale inputs (X) per feature and target (y) separately
                        x_scaler = MinMaxScaler()
                        train_x_scaled = x_scaler.fit_transform(train_features)

                        y_scaler = MinMaxScaler()
                        train_target = df['total load actual'].iloc[:split_idx].values.reshape(-1, 1).astype(float)
                        y_scaler.fit(train_target)

                        last_seq = train_x_scaled[-lookback:, :].copy()  # (lookback, n_features)
                        current_seq = last_seq.copy()
                        lstm_forecast_scaled = []

                        for _ in range(steps):
                            pred_scaled = model.predict(current_seq.reshape(1, lookback, int(expected_n_features)), verbose=0)[0, 0]
                            lstm_forecast_scaled.append(pred_scaled)

                            # Build next feature vector: set target (index 0) to pred; keep others as last observed
                            next_vec = current_seq[-1, :].copy()
                            next_vec[0] = pred_scaled
                            current_seq = np.vstack([current_seq[1:], next_vec])

                        forecast_result = y_scaler.inverse_transform(
                            np.array(lstm_forecast_scaled).reshape(-1, 1)
                        ).flatten()
                    
                    # Calculate metrics
                    if len(test_data) >= steps:
                        model_metrics['RMSE'] = np.sqrt(mean_squared_error(test_data[:steps], forecast_result))
                        model_metrics['MAE'] = mean_absolute_error(test_data[:steps], forecast_result)
                        model_metrics['MAPE'] = np.mean(np.abs((test_data[:steps] - forecast_result) / test_data[:steps])) * 100
                    
                    st.success("✅ LSTM model completed successfully!")
                    
            except ImportError:
                st.error("❌ TensorFlow not installed. Run: pip install tensorflow")
            except Exception as e:
                st.error(f"❌ LSTM error: {str(e)[:200]}")
        
        # Display metrics if available
        if model_metrics:
            st.markdown("---")
            st.subheader("📊 Model Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{model_metrics['RMSE']:.2f} MW", 
                         help="Root Mean Square Error - lower is better")
            with col2:
                st.metric("MAE", f"{model_metrics['MAE']:.2f} MW",
                         help="Mean Absolute Error - lower is better")
            with col3:
                st.metric("MAPE", f"{model_metrics['MAPE']:.2f}%",
                         help="Mean Absolute Percentage Error - lower is better")
        
        # Visualization
        if forecast_result is not None:
            st.markdown("---")
            st.subheader("📈 Forecast Visualization")
            
            fig = go.Figure()
            
            # Historical data (last 7 days)
            history_window = min(7 * 24, len(test_data))
            hours_range = np.arange(-history_window, steps)
            
            fig.add_trace(go.Scatter(
                x=hours_range[:history_window],
                y=test_data[:history_window] if len(test_data) >= history_window else test_data,
                mode='lines',
                name='Historical Data',
                line=dict(color='#333333', width=2.5)
            ))
            
            # Model forecast
            fig.add_trace(go.Scatter(
                x=np.arange(0, steps),
                y=forecast_result,
                mode='lines+markers',
                name=f'{model_choice} Forecast',
                line=dict(color='#FF6B6B', width=2.5, dash='dash'),
                marker=dict(size=5)
            ))
            
            # Actual test data (if available)
            if len(test_data) >= steps:
                fig.add_trace(go.Scatter(
                    x=np.arange(0, steps),
                    y=test_data[:steps],
                    mode='lines',
                    name='Actual Data',
                    line=dict(color='#4ECDC4', width=2),
                    opacity=0.7
                ))
            
            fig.update_layout(
                xaxis_title='Hours from Now',
                yaxis_title='Energy Load (MW)',
                template='plotly_white',
                height=500,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast statistics
            st.markdown("---")
            st.subheader("📈 Forecast Statistics")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Min Forecast", f"{forecast_result.min():,.0f} MW")
            with stat_col2:
                st.metric("Avg Forecast", f"{forecast_result.mean():,.0f} MW")
            with stat_col3:
                st.metric("Max Forecast", f"{forecast_result.max():,.0f} MW")
            with stat_col4:
                st.metric("Std Forecast", f"{forecast_result.std():,.0f} MW")
            
            # Comparison table
            st.markdown("---")
            st.subheader("📋 Detailed Forecast (First 24 Hours)")
            
            comparison_data = {
                'Hour': np.arange(1, min(steps + 1, 25)),
                f'{model_choice} Forecast (MW)': forecast_result[:24]
            }
            
            if len(test_data) >= 24:
                comparison_data['Actual (MW)'] = test_data[:24]
                comparison_data['Error (MW)'] = forecast_result[:24] - test_data[:24]
                comparison_data['Error (%)'] = (comparison_data['Error (MW)'] / test_data[:24] * 100).round(2)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Download forecast
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Results (CSV)",
                data=csv,
                file_name=f"forecast_{model_choice}_{forecast_days}d_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download detailed forecast data"
            )
            
            st.success(f"🎉 {forecast_days}-day forecast completed using {model_choice} model!")
        else:
            st.error("⚠️ Forecast failed. Please check the error messages above.")

else:
    st.info("👈 Configure forecast settings in the sidebar and click 'Run Forecast' to start")
    
    st.markdown("### 📚 Model Information")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **SARIMAX Model**
        - Seasonal AutoRegressive Integrated Moving Average with eXogenous variables
        - Good for capturing seasonal patterns
        - Parameters: (1,1,1) x (1,1,1,24)
        - Suitable for time series with trend and seasonality
        """)
    
    with col_info2:
        st.markdown("""
        **LSTM Model**
        - Long Short-Term Memory neural network
        - Deep learning approach for sequence prediction
        - Architecture: 50-32-16-1 neurons
        - Good for capturing complex non-linear patterns
        """)
