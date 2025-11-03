import os
os.environ['DARTS_CONFIGURE_MATPLOTLIB'] = '0'
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import calendar
import numpy as np
import seaborn as sns
import base64
import matplotlib.pyplot as plt
from PredictSage_model import (
    ExponentialSmoothingModel,
    ArimaModel,
    AutoArimaModel,
    SarimaModel,
    ThetaModel,
    NbeatsModel,
    EnsembleModel,
    LstmModel,
    RnnModel,
    VarModel,
    VarmaxModel,
    GruModel,
    MvRnnModel,
    MvLstmModel,
    StatsBaseModel,
    DartsBaseModel,
    TfBaseModel,
    MultivariateStatsBaseModel,
    MultivariateTfBaseModel
)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
import random
sns.set_style("white")
plt.style.use("dark_background")
import plotly.express as px
import plotly.graph_objects as go 
import tensorflow as tf
import tempfile
import sqlite3
from sqlalchemy import create_engine, inspect

# Import connector functions from connectors.py
from connectors import load_file_data, get_sqlite_tables, load_sqlite_data, get_db_tables, load_db_data, get_excel_sheets

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

MIN_ROW_COUNT = 50

# Configuration dictionaries
STATS_BASE_MODEL_CONFIG = {
    "ArimaOrder": (2, 1, 2),
    "AutoArimaSeasonality": 12,
    "SarimaOrder": (1, 1, 1),
    "SeasonalityOrder": (1, 1, 1, 12)
}
DARTS_BASE_MODEL_CONFIG = {
    "nbeatsepochs": 50,
    "Ensembleepochs": 50
}
TF_BASE_MODEL_CONFIG = {
    "NNEpochs": 50
}
MV_STATS_BASE_MODEL_CONFIG = {
    "maxlags": 30,
    "order": (1, 1)
}
MV_TF_BASE_MODEL_CONFIG = {
    "epochs": 200,
    "early_stopping_patience": 20,
    "verbose": 0,
    "GRU_activation": "tanh",
    "LSTM_activation": "tanh",
    "RNN_activation": "relu"
}

models_explanation = {
    "Exponential Smoothing": "Suitable for data with a clear trend and seasonality. It is fast and easy to implement.",
    "Auto Arima": "Automatically selects the best ARIMA model for your data. It is good for data with trends and seasonality",
    "ARIMA": "A powerful model for time series forecasting that can handle data with trends and seasonality.",
    "SARIMA": "An extension of ARIMA that supports seasonality. It is useful for data with seasonal patterns.",
    "Theta": "A simple yet effective model for forecasting data with trends.",
    "LSTM": "A type of recurrent neural network (RNN) that is well-suited for sequential data and can capture long-term dependencies",
    "RNN": "Good for modeling sequential data, but may not struggle with long-term dependencies compared to LSTM.",
    "N-beats": "A deep learning model specifically designed for time series forecasting. It can handle complex patterns in the data.",
    "Ensemble": "Combines multiple models to improve forecasting accuracy. It is useful for capturing different aspects of the data."
}

def load_landing():
    """Displays the landing page with user guide and app introduction."""
    st.image("./PredictSage_logo.png", use_container_width=True)
    st.subheader("User Guide")
    st.header("Data Visualization")
    st.write("""
    1. **Time Series Plot**:
    - The app will display a time series plot of your selected metric over time. This helps you visualize the trends and patterns in your data.
    2. **Seasonal Decomposition**:
    - The app performs seasonal decomposition of your time series data and displays the decomposed components (trend, seasonal, and residual).
    """)

    st.header("Model Selection and Forecasting")
    st.write("We have split the models into simple and complex categories. Simple models run faster but may not be best suited for complex data. Users can choose between these two categories based on their needs.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Simple Models")
        st.write("Exponential Smoothing | Auto ARIMA,ARIMA | SARIMA | Theta")
        st.markdown("""
        **Why Choose Simple Models?**
        - **Speed**
        - **Interpretability**
        - **Simplicity**
        """)
          
    with col2:
        st.subheader("Complex Models")
        st.write("LSTM | RNN | N-beats | Ensemble")
        st.markdown("""
        **Why Choose Complex Models?**
        - **Accuracy**
        - **Flexibility**
        - **Advanced Features**
        """)

    st.subheader("Model Metrics")
    st.write("""
    Evaluate model accuracy using RMSE and MAE in the 'Model Metrics' table:
    1. **Lower RMSE = Better Accuracy**  
       - Smaller RMSE means predictions are closer to actual values. 
    2. **Lower MAE = Smaller Average Error**   
       - Lower MAE shows smaller average mistakes. 
    3. **RMSE > MAE = Watch for Big Errors**  
       - If RMSE is much higher than MAE, some predictions have large errors.
    """)

    st.header("Visualization of Forecast")
    st.write("""
    1. **Forecast Data**:
    - The app displays the forecasted data along with the original data in a table format.
    
    2. **Forecast Plot**:
    - The app plots the forecasted values along with the original time series data for visual comparison.
    """)

def infer_file_type(filename):
    """
    Infer the file type from the file extension.
    
    Args:
        filename (str): Name of the uploaded file.
    
    Returns:
        str or None: Inferred file type ('csv', 'json', 'excel', 'text', 'sqlite') or None if unsupported.
    """
    extension = os.path.splitext(filename.lower())[1]
    extension_map = {
        '.csv': 'csv',
        '.json': 'json',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.txt': 'text',
        '.db': 'sqlite',
        '.sqlite': 'sqlite'
    }
    return extension_map.get(extension)

@st.cache_data
def plot_results(y_true, y_pred, title, target_col=None):
    fig = go.Figure()
    
    if isinstance(y_true, pd.Series):
        col = target_col if target_col else y_true.name if y_true.name else 'Value'
        true_values = y_true
    elif isinstance(y_true, pd.DataFrame):
        col = target_col if target_col else y_true.columns[0]
        true_values = y_true[col]
    else:
        raise ValueError("y_true must be a pandas Series or DataFrame")
    
    fig.add_trace(go.Scatter(x=y_true.index, y=true_values, mode='lines', name=f'Actual {col}'))
    
    if not y_pred.empty:
        if isinstance(y_pred, pd.Series):
            pred_values = y_pred
        elif isinstance(y_pred, pd.DataFrame):
            pred_values = y_pred[col]
        else:
            raise ValueError("y_pred must be a pandas Series or DataFrame")
        fig.add_trace(go.Scatter(x=y_pred.index, y=pred_values, mode='lines', name=f'Forecast {col}', line=dict(dash='dash')))
    
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')
    return fig

@st.cache_data
def plot_acf_pacf(series, lags=30, title_prefix=""):
    acf_fig = plt.figure(figsize=(10, 4))
    plot_acf(series, lags=lags, ax=acf_fig.add_subplot(111))
    acf_fig.tight_layout()
    
    pacf_fig = plt.figure(figsize=(10, 4))
    plot_pacf(series, lags=lags, ax=pacf_fig.add_subplot(111))
    pacf_fig.tight_layout()
    return acf_fig, pacf_fig

@st.cache_data
def plot_heatmap(df, title="Correlation Heatmap"):
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    return fig

def preprocess_data(df_or_series, window=3):
    if isinstance(df_or_series, pd.Series):
        if df_or_series.isnull().any():
            st.warning("Null values detected in the series. Filling with rolling mean.")
            filled = df_or_series.fillna(df_or_series.rolling(window, min_periods=1, center=True).mean())
            filled = filled.fillna(method='ffill').fillna(method='bfill')
            return filled
        return df_or_series
    elif isinstance(df_or_series, pd.DataFrame):
        if df_or_series.isnull().any().any():
            st.warning("Null values detected in the DataFrame. Filling with rolling mean.")
            filled = df_or_series.fillna(df_or_series.rolling(window, min_periods=1, center=True).mean())
            filled = filled.fillna(method='ffill').fillna(method='bfill')
            return filled
        return df_or_series

def load_models(column: pd.Series, data: pd.DataFrame, type: str, models_type: str):
    processed_column = preprocess_data(column)
    processed_data = preprocess_data(data)

    StatsBaseModel.initialize_data(processed_column, **STATS_BASE_MODEL_CONFIG)
    DartsBaseModel.initialize_data(processed_column, **DARTS_BASE_MODEL_CONFIG)
    TfBaseModel.initialize_data(processed_column, **TF_BASE_MODEL_CONFIG)
    MultivariateStatsBaseModel.initialize_data(processed_data)
    MultivariateTfBaseModel.initialize_data(processed_data)

    simple_models = {
        "Exponential Smoothing": ExponentialSmoothingModel(),
        "Auto Arima": AutoArimaModel(),
        "ARIMA": ArimaModel(),
        "SARIMA": SarimaModel(),
        "Theta": ThetaModel(),
    }
    complex_models = {
        "LSTM": LstmModel(),
        "RNN": RnnModel(),
        "N-beats": NbeatsModel(),
        "Ensemble": EnsembleModel()
    }
    multivariate_models = {
        "VAR": VarModel(),
        "VARMAX": VarmaxModel(),
        "GRU": GruModel(),
        "RNN": MvRnnModel(),
        "LSTM": MvLstmModel()
    }
    if type == "Multivariate":
        return multivariate_models
    elif models_type == "Simple":
        return simple_models
    else:
        return complex_models

@st.cache_data
def find_metrics(_models, column: pd.Series, type: str, target_col=None):
    processed_column = preprocess_data(column)
    results = []
    for model_key, model in _models.items():
        test, forecast, mae, rmse = model.calculate_metrics()
        if type == "Multivariate" and isinstance(mae, dict):
            mae_val = mae.get(target_col, mae[list(mae.keys())[0]])
            rmse_val = rmse.get(target_col, rmse[list(rmse.keys())[0]])
        else:
            mae_val = mae
            rmse_val = rmse
        results.append([model_key, mae_val, rmse_val])
    
    result_df = pd.DataFrame(results, columns=["Model Name", "MAE", "RMSE"])
    result_df = result_df.sort_values(by='RMSE')
    result_df = result_df.reset_index(drop=True)
    return result_df

def style_df(df, model_type):
    def highlight_min_rmse(row, model_type):
        if row['Model Name'] == model_type:
            return ['background-color: yellow; font-weight: bold'] * len(row)
        return [''] * len(row)
    styled_results_df = df.style.apply(lambda x: highlight_min_rmse(x, model_type), axis=1)
    styled_results_df = styled_results_df.format({'RMSE': '{:.2f}', 'MAE': '{:.2f}'})
    return styled_results_df

def check_regularity(column):
    if (pd.infer_freq(column)):
        return True
    diff = None
    for idx in range(0, column.shape[0]-1):
        if diff and diff != column[idx+1] - column[idx]:
            return False
        diff = column[idx+1] - column[idx]
    return True

def group_by_period(df, period):
    original_freq = pd.infer_freq(df.index)
    if original_freq == 'MS' and period in ['week', 'day']:
        st.warning("Original data frequency is monthly. Resampling to a lower frequency is not recommended.")
        return df
    elif original_freq == 'W' and period == 'day':
        st.warning("Original data frequency is weekly. Resampling to a lower frequency is not recommended.")
        return df
    else:
        if period == 'month':
            return df.resample('MS').sum()
        elif period == 'week':
            return df.resample('W').sum()
        elif period == 'day':
            return df.resample('D').sum()
        else:
            raise ValueError("Period must be 'month', 'week', or 'day'")
        
@st.dialog("User Guider", width='large')
def open_landing():
    load_landing()

if st.sidebar.button("User Guide"):
    open_landing()

# Initialize session state for tab navigation
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Data source selection (in center)

df = None
source_type = st.sidebar.selectbox("Select Source:", [None, "File", "Server"], key="source_type")

if not source_type:
    # load_landing()
    st.stop()

elif source_type == "File":
    with st.container():
        file = st.sidebar.file_uploader("Please upload a file:", ['csv', 'xls', 'xlsx', 'txt', 'json', 'db', 'sqlite'], key="file_uploader")
        
        if not file:
            # load_landing()
            st.stop()
        
        file_type = infer_file_type(file.name)
        excel_sheet = None
        if file_type == "excel":
            file.seek(0)
            try:
                sheets = get_excel_sheets(file)
                excel_sheet = st.sidebar.selectbox("Select a sheet:", sheets, key="excel_sheet")
            except Exception as e:
                # load_landing()
                st.error(f"Error loading Excel sheets: {e}")
                st.stop()
        
        if file_type == "sqlite":
            file.seek(0)
            try:
                tables = get_sqlite_tables(file)
            except Exception as e:
                # load_landing()
                st.error(f"Error loading tables: {e}")
                st.stop()
                
            selected_table = st.sidebar.selectbox("Select a table:", tables, key="sqlite_table")
            file.seek(0)
            try:
                df = load_sqlite_data(file_type, file=file, sqlite_table=selected_table)
            except Exception as e:
                # load_landing()
                st.error(f"Error loading data: {e}")
                st.stop()
        else:
            has_headers = st.sidebar.selectbox("Does the file have column headers?", [True, False], index=0, key="has_headers")
            try:
                df = load_file_data(file, file_type, has_headers=has_headers, excel_sheet=excel_sheet)
            except Exception as e:
                # load_landing()
                st.error(f"Error loading data: {e}")
                st.stop()

elif source_type == "Server":
    with st.container():
        db_type = st.selectbox("Database type:", ["PostgreSQL", "MySQL", "SQL Server"], key="db_type")
        user = st.text_input("Username", key="db_user")
        password = st.text_input("Password", type="password", key="db_password")
        host = st.text_input("Host", value="localhost", key="db_host")
        port = st.text_input("Port", value="5432", key="db_port")
        database = st.text_input("Database name", key="db_name")
        st.button('Login')
        try:
            tables = get_db_tables(db_type.lower().replace(" ", ""), user, password, host, port, database)
        except Exception as e:
            # load_landing()
            st.error(f"Error loading tables: {e}")
            st.stop()
        
        selected_table = st.sidebar.selectbox("Select a table:", tables, key="db_table")
        try:
            df = load_db_data(db_type.lower().replace(" ", ""), user, password, host, port, database, table_name=selected_table)
        except Exception as e:
            # load_landing()
            st.error(f"Error loading data: {e}")
            st.stop()

# elif source_type in ["Tableau Server (TBD)", "PowerBI Service (TBD)"]:
#     st.warning("This data source is planned for future implementation (TBD). Please select another source for this POC.")
#     st.stop()

# Main app logic
if df is None or df.empty:
    # load_landing()
    st.warning("No data loaded. Please select a valid data source.")
    st.stop()

# Analysis type selection
# st.header("Select Analysis Type")
analysis_type = st.sidebar.selectbox("Select Analysis Type", [None, "Univariate", "Multivariate"], key="analysis_type")
st.title("PredictSage📊")
date_column = None
metric_column = None
metric_columns = None
target_col = None
model_category = None
model_type = None
forecast_period = None

if analysis_type:
   
    date_column = st.sidebar.selectbox("Select the date column", [None] + list(df.columns), key="date_column")
    if date_column and analysis_type:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            if df[date_column].dtype not in ['datetime64[ns]'] or not check_regularity(df[date_column]):
                st.error("Selected Date Column is not an ideal data type for analysis. Please use date type with regular intervals.")
                date_column = None
            else:
                df.set_index(date_column, inplace=True)
                if analysis_type == "Univariate":
                    metric_column = st.sidebar.selectbox("Select the metric column", [None] + list(df.columns), key="metric_column")
                    if metric_column:
                        model_category = st.sidebar.selectbox("Choose model category", [None, "Simple", "Complex"], key="model_category")
                        if model_category:
                            models = load_models(df[metric_column], df, "Univariate", model_category)
                            model_type = st.sidebar.selectbox("Select Forecasting Model", [None] + list(models.keys()), key="model_type")
                            forecast_period = st.sidebar.selectbox("Select Forecast Period", [3, 6, 9, 100], key="forecast_period")
                elif analysis_type == "Multivariate":
                    metric_columns = st.sidebar.multiselect("Select metric columns (at least two)", list(df.columns), key="metric_columns")
                    if len(metric_columns) >= 2:
                        target_col = st.sidebar.selectbox("Select target column", metric_columns, key="target_col")
                        if target_col:
                            models = load_models(pd.Series([1,2,3]), df[metric_columns], "Multivariate", "")
                            model_type = st.sidebar.selectbox("Select Forecasting Model", [None] + list(models.keys()), key="model_type")
                            forecast_period = st.sidebar.selectbox("Select Forecast Period", [3, 6, 9, 100], key="forecast_period")
        except Exception as e:
            st.error(f"Error processing date column: {e}")
            st.stop()

# Main content with output tabs
if date_column and analysis_type:
    if analysis_type == "Univariate" and metric_column:
        df = df[[metric_column]]
        df = preprocess_data(df)
        if df.shape[0] < 50:
            st.error("Less than 50 rows. Please use different data.")
            st.stop()
        elif df[metric_column].isna().all():
            st.error("Selected column contains only NaN values. Please select a different column.")
            st.stop()
        else:
            tab_names = ["Data Preview", "Time Series Plot", "Seasonal Decomposition", "Model Metrics", "Actual vs Predicted", "Forecasted Results"]
            
            # Function to navigate to next tab
            def next_tab(max_tabs):
                if st.session_state.active_tab < max_tabs - 1:
                    st.session_state.active_tab += 1
                st.rerun()

            # Display tab navigation buttons
            col1, col2, col3 = st.columns([2 ,6 ,2])
            with col1:
                if st.session_state.active_tab > 0:
                    if st.button("⬅️ Previous", key="prev_tab"):
                        st.session_state.active_tab -= 1
                        st.rerun()
            with col2:
                st.markdown(f"**{tab_names[st.session_state.active_tab]}**", unsafe_allow_html=True)
            with col3:
                if st.session_state.active_tab < len(tab_names) - 1:
                    if st.button("➡️ Next", key="next_tab"):
                        next_tab(len(tab_names))

            # Display content based on active tab
            with st.container():
                if st.session_state.active_tab == 0:  # Data Preview
                    st.write("Data loaded successfully!")
                    st.write(df.head())

                elif st.session_state.active_tab == 1:  # Time Series Plot
                    st.subheader("Time Series Plot")
                    fig = plot_results(df, pd.DataFrame(), "Time Series Plot")
                    st.plotly_chart(fig, use_container_width=True)

                elif st.session_state.active_tab == 2:  # Seasonal Decomposition
                    st.subheader("Seasonal Decomposition")
                    decomposition = seasonal_decompose(df[metric_column], model='additive', period=12)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'))
                    fig.update_layout(title="Seasonal Decomposition", xaxis_title="Date", yaxis_title='Value')
                    st.plotly_chart(fig, use_container_width=True)

                elif st.session_state.active_tab == 3:  # Model Metrics
                    if model_category and model_type:
                        st.subheader("Model Metrics")
                        result_df = find_metrics(models, df[metric_column], model_category)
                        st.dataframe(style_df(result_df, model_type))
                    else:
                        st.write("Select a model category and type to view metrics.")

                elif st.session_state.active_tab == 4:  # Actual vs Predicted
                    if model_category and model_type:
                        st.subheader(f"Selected: {model_type}")
                        st.write(models_explanation.get(model_type, "No explanation available."))
                        st.subheader("Actual Vs Predicted")
                        test, forecast, mae, rmse = models[model_type].calculate_metrics()
                        st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                        perf_df = pd.DataFrame({"Actual": test, "Predicted": forecast})
                        perf_df['Predicted'] = perf_df["Predicted"].map(lambda x: int(x) if not pd.isnull(x) else None)
                        def find_percentage_diff(row):
                            res = (math.fabs(row['Actual'] - row['Predicted']) / row['Actual'] * 100)
                            out = str(round(res, 2)) + "%"
                            return out
                        perf_df['Delta %'] = perf_df.apply(find_percentage_diff, axis=1)
                        st.dataframe(perf_df)
                        fig = plot_results(test, forecast, "Actual vs Predicted")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Select a model category and type to view predictions.")

                elif st.session_state.active_tab == 5:  # Forecasted Results
                    if model_category and model_type:
                        st.subheader("🔮 Forecasted Results")
                        forecast_df = models[model_type].forecast(forecast_period)
                        forecast_df_res = pd.concat([df, forecast_df.to_frame(name=metric_column)])
                        st.dataframe(forecast_df_res)

                        forecast_df = forecast_df.rename("Forecast")
                        forecast_df_res = pd.concat([df, forecast_df])
                        forecast_df_res['Forecast'] = forecast_df_res['Forecast'].map(lambda x: int(x) if not pd.isnull(x) else None)
                        forecast_df = forecast_df.map(lambda x: int(x) if not pd.isnull(x) else None)

                        future_df_segregated = pd.DataFrame(forecast_df_res)
                        future_df_segregated['year'] = future_df_segregated.index.year
                        future_df_segregated['month'] = future_df_segregated.index.month

                        last_actual_year = df.index.year.max()
                        future_df_segregated = future_df_segregated[
                            ((future_df_segregated['year'] == last_actual_year) & future_df_segregated[metric_column].notnull()) |
                            future_df_segregated['Forecast'].notnull()
                        ]

                        future_df_pivot = pd.pivot_table(
                            future_df_segregated,
                            values=[metric_column, 'Forecast'],
                            index='month',
                            columns='year',
                            aggfunc='mean'
                        )
  
                        future_df_pivot.columns = ['_'.join(map(str, col)).strip() for col in future_df_pivot.columns.values]
                        fig_forecast = go.Figure()
                        months = list(range(1, 13))
                        for col in [c for c in future_df_pivot.columns if c.startswith(metric_column) and str(last_actual_year) in c]:
                            year = col.split('_')[-1]
                            fig_forecast.add_trace(go.Scatter(
                                x=months,
                                y=future_df_pivot[col],
                                mode='lines',
                                name=f'Actual {year}',
                                line=dict(dash='dot', color='red')
                            ))

                        for col in [c for c in future_df_pivot.columns if c.startswith('Forecast')]:
                            year = col.split('_')[-1]
                            fig_forecast.add_trace(go.Scatter(
                                x=months,
                                y=future_df_pivot[col],
                                mode='lines',
                                name=f'Forecast {year}',
                                line=dict(color='blue')
                            ))

                        fig_forecast.update_layout(
                            title="Forecasted Results",
                            xaxis_title="Month",
                            yaxis_title="Value",
                            xaxis=dict(
                                tickmode='array',
                                tickvals=months,
                                ticktext=[calendar.month_name[tick] for tick in months]
                            ),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )

                        st.plotly_chart(fig_forecast, use_container_width=True)
                    else:
                        st.write("Select a model category and type to view forecasts.")

    elif analysis_type == "Multivariate" and metric_columns and len(metric_columns) >= 2 and target_col:
        df = df[metric_columns]
        df = preprocess_data(df)
        if df.shape[0] < 50:
            st.error("Less than 50 rows. Please use different data.")
            st.stop()
        else:
            tab_names = ["Data Preview", "Time Series Plot", "Seasonal Decomposition", "ACF/PACF", "Correlation Heatmap", "Model Metrics", "Actual vs Predicted", "Forecasted Results"]
            
            # Function to navigate to next tab
            def next_tab(max_tabs):
                if st.session_state.active_tab < max_tabs - 1:
                    st.session_state.active_tab += 1
                st.rerun()

            # Display tab navigation buttons
            col1, col2, col3 = st.columns([2, 6, 2])
            with col1:
                if st.session_state.active_tab > 0:
                    if st.button("⬅️ Previous", key="multi_prev_tab"):
                        st.session_state.active_tab -= 1
                        st.rerun()
            with col2:
                st.markdown(f"**{tab_names[st.session_state.active_tab]}**", unsafe_allow_html=True)
            with col3:
                if st.session_state.active_tab < len(tab_names) - 1:
                    if st.button("➡️ Next", key="multi_next_tab"):
                        next_tab(len(tab_names))

            # Display content based on active tab
            with st.container():
                if st.session_state.active_tab == 0:  # Data Preview
                    st.write("Data loaded successfully!")
                    st.write(df.head())

                elif st.session_state.active_tab == 1:  # Time Series Plot
                    st.subheader("Time Series Plot")
                    fig = plot_results(df, pd.DataFrame(), "Time Series Plot", target_col)
                    st.plotly_chart(fig, use_container_width=True)

                elif st.session_state.active_tab == 2:  # Seasonal Decomposition
                    st.subheader("Seasonal Decomposition of Target")
                    decomposition = seasonal_decompose(df[target_col], model='additive', period=12)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'))
                    fig.update_layout(title=f"Seasonal Decomposition of {target_col}", xaxis_title="Date", yaxis_title='Value')
                    st.plotly_chart(fig, use_container_width=True)

                elif st.session_state.active_tab == 3:  # ACF/PACF
                    st.subheader("ACF and PACF Plots of Target")
                    acf_fig, pacf_fig = plot_acf_pacf(df[target_col], lags=min(30, len(df)//2), title_prefix=target_col)
                    st.pyplot(acf_fig)
                    st.pyplot(pacf_fig)
                    plt.close(acf_fig)
                    plt.close(pacf_fig)

                elif st.session_state.active_tab == 4:  # Correlation Heatmap
                    st.subheader("Correlation Heatmap")
                    fig = plot_heatmap(df)
                    st.pyplot(fig)
                    plt.close(fig)

                elif st.session_state.active_tab == 5:  # Model Metrics
                    if model_type:
                        st.subheader("Model Metrics")
                        result_df = find_metrics(models, df[target_col], "Multivariate", target_col)
                        st.dataframe(style_df(result_df, model_type))
                    else:
                        st.write("Select a model type to view metrics.")

                elif st.session_state.active_tab == 6:  # Actual vs Predicted
                    if model_type:
                        st.subheader(f"Selected: {model_type}")
                        multi_explanations = {
                            "VAR": "Models the target series using past values of all selected series.",
                            "VARMAX": "Vector ARIMA-like model using past values of all series with exogenous handling.",
                            "GRU": "Deep learning GRU model for multivariate sequences.",
                            "RNN": "Simple RNN for multivariate data.",
                            "LSTM": "LSTM for multivariate time series."
                        }
                        st.write(multi_explanations.get(model_type, "No explanation available."))
                        st.subheader("Actual Vs Predicted")
                        test, forecast, mae, rmse = models[model_type].calculate_metrics()
                        mae_val = mae[target_col] if isinstance(mae, dict) else mae
                        rmse_val = rmse[target_col] if isinstance(rmse, dict) else rmse
                        st.write(f"MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")
                        test = test[target_col]
                        forecast = forecast[target_col]
                        perf_df = pd.DataFrame({"Actual": test, "Predicted": forecast})
                        perf_df['Predicted'] = perf_df["Predicted"].map(lambda x: int(x) if not pd.isnull(x) else None)
                        def find_percentage_diff(row):
                            res = (math.fabs(row['Actual'] - row['Predicted']) / row['Actual'] * 100)
                            out = str(round(res, 2)) + "%"
                            return out
                        perf_df['Delta %'] = perf_df.apply(find_percentage_diff, axis=1)
                        st.dataframe(perf_df)
                        fig = plot_results(test, forecast, "Actual vs Predicted", target_col)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Select a model type to view predictions.")

                elif st.session_state.active_tab == 7:  # Forecasted Results
                    if model_type:
                        st.subheader("🔮 Forecasted Results")
                        forecast_df = models[model_type].forecast(forecast_period)
                        forecast_df_res = pd.concat([df[[target_col]], forecast_df[[target_col]]])
                        st.dataframe(forecast_df_res)
                        fig = plot_results(df[[target_col]], forecast_df, "Historical and Forecasted Data", target_col)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Select a model type to view forecasts.")

    elif analysis_type == "Multivariate":
        tab_names = ["Data Preview", "Correlation Heatmap"]
        
        # Function to navigate to next tab
        def next_tab(max_tabs):
            if st.session_state.active_tab < max_tabs - 1:
                st.session_state.active_tab += 1
            st.rerun()

        # Display tab navigation buttons
        col1, col2, col3 = st.columns([2, 6, 2])
        with col1:
            if st.session_state.active_tab > 0:
                if st.button("⬅️ Previous", key="multi_no_metrics_prev_tab"):
                    st.session_state.active_tab -= 1
                    st.rerun()
        with col2:
            st.markdown(f"**{tab_names[st.session_state.active_tab]}**", unsafe_allow_html=True)
        with col3:
            if st.session_state.active_tab < len(tab_names) - 1:
                if st.button("➡️ Next", key="multi_no_metrics_next_tab"):
                    next_tab(len(tab_names))

        # Display content based on active tab
        with st.container():
            if st.session_state.active_tab == 0:  # Data Preview
                st.write("Data loaded successfully!")
                st.write(df.head())

            elif st.session_state.active_tab == 1:  # Correlation Heatmap
                st.subheader("Correlation Heatmap")
                fig = plot_heatmap(df)
                st.pyplot(fig)
                plt.close(fig)

else:
    st.write("Please select an analysis type to proceed.") 