
####################################################################################
# App Name: PredictSage_app
# Date Modified:19/03/2025
# version: V6
####################################################################################


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



# import subprocess
# import sys

# def install_package(package_name):
#     try:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
#         print(f"Successfully installed {package_name}")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install {package_name}: {e}")
#         return False
#     except Exception as e:
#         print(f"An error occurred while installing {package_name}: {e}")
#         return False

# required_packages = [
#     "streamlit",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "numpy",
#     "plotly",
#     "tensorflow",
#     "statsmodels"
# ]

# # st.write("Checking and installing required packages...")
# for package in required_packages:
#     if not install_package(package):
#         st.error(f"Failed to install {package}. Please install it manually using 'pip install {package}' and try again.")
#         st.stop()

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
MV_STATS_BASE_MODEL_CONFIG={
    "maxlags":30,
    "order":(1,1)
}
MV_TF_BASE_MODEL_CONFIG ={
    "epochs":200,
    "early_stopping_patience":20,
    "verbose":0,
    "GRU_activation":"tanh",
    "LSTM_activation":"tanh",
    "RNN_activation":"relu"
}

models_explanation = {
    "Exponential Smoothing": "Suitable for data with a clear trend and seasonality. It is fast and easy to implement.",
    "Auto Arima": "Automatically selects the best ARIMA model for your data. It is good for data with trends and seasonality",
    "ARIMA": "A powerful model for time series forecasting that can handle data with trends and seasonality.",
    "SARIMA": "An extension of ARIMA that supports seasonality. It is useful for data with seasonal patterns.",
    "Theta": "A simple yet effective model for forecasting data with trends.",
    "LSTM": "A type of recurrent neural network (RNN) that is well-suited for sequential data and can capture long-term dependencies",
    "RNN": "Good for modeling sequential data, but may struggle with long-term dependencies compared to LSTM.",
    "N-beats": "A deep learning model specifically designed for time series forecasting. It can handle complex patterns in the data.",
    "Ensemble": "Combines multiple models to improve forecasting accuracy. It is useful for capturing different aspects of the data."
}

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
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_acf_pacf(series, lags=30, title_prefix=""):
    acf_fig = plt.figure(figsize=(10, 4))
    plot_acf(series, lags=lags, ax=acf_fig.add_subplot(111))
    acf_fig.tight_layout()
    st.pyplot(acf_fig)
    plt.close(acf_fig)
    
    pacf_fig = plt.figure(figsize=(10, 4))
    plot_pacf(series, lags=lags, ax=pacf_fig.add_subplot(111))
    pacf_fig.tight_layout()
    st.pyplot(pacf_fig)
    plt.close(pacf_fig)

@st.cache_data
def plot_heatmap(df, title="Correlation Heatmap"):
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

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

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

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
if uploaded_file is not None:
    st.title("PredictSage📊")
    st.write("File uploaded successfully!")
    data = pd.read_csv(uploaded_file)
    df = pd.DataFrame(data)
    st.write(df.head())

    analysis_type = st.sidebar.selectbox("Select Analysis Type", [None, "Univariate", "Multivariate"])
    date_column = st.sidebar.selectbox("Select the date column", [None] + list(df.columns))

    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        if df[date_column].dtype not in ['datetime64[ns]'] or not check_regularity(df[date_column]):
            st.write("Selected Date Column is not an ideal data type for analysis. Please use date type with regular intervals.")
            date_column = None

    if date_column and analysis_type:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)

        if analysis_type == "Univariate":
            metric_column = st.sidebar.selectbox("Select the metric column", [None] + list(df.columns))
            if metric_column:
                df = df[[metric_column]]
                df = preprocess_data(df)
                if df.shape[0] < 50:
                    st.write("Less than 50 rows. Please use different data.")
                elif df[metric_column].isna().all():
                    st.error("Selected column contains only NaN values. Please select a different column.")
                else:
                    st.subheader("Time Series Plot")
                    plot_results(df, pd.DataFrame(), "Time Series Plot")

                    st.subheader("Seasonal Decomposition")
                    decomposition = seasonal_decompose(df[metric_column], model='additive', period=12)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'))
                    fig.update_layout(title="Seasonal Decomposition", xaxis_title="Date", yaxis_title='Value')
                    st.plotly_chart(fig)

                    # st.subheader("ACF and PACF Plots")
                    # plot_acf_pacf(df[metric_column], lags=min(30, len(df)//2), title_prefix=metric_column)

                    # st.subheader("Correlation Heatmap")
                    # plot_heatmap(df)

                    model_category = st.sidebar.selectbox("Choose model category", [None, "Simple", "Complex"])
                    if model_category:
                        models = load_models(df[metric_column], df, "Univariate", model_category)
                        model_type = st.sidebar.selectbox("Select Forecasting Model", [None] + list(models.keys()))
                        forecast_period = st.sidebar.selectbox("Select Forecast Period", [3, 6, 9, 100])

                        if model_type:
                            st.subheader("Model Metrics")
                            result_df = find_metrics(models, df[metric_column], model_category)
                            st.dataframe(style_df(result_df, model_type))

                            st.subheader(f"Selected: {model_type}")
                            st.write(models_explanation.get(model_type, "No explanation available."))
                            st.subheader("Actual Vs Predicted")
                            test, forecast, mae, rmse = models[model_type].calculate_metrics()
                            st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                            test, forecast, mae, rmse = models[model_type].calculate_metrics()
                            perf_df = pd.DataFrame({"Actual": test, "Predicted": forecast})
                            perf_df['Predicted'] = perf_df["Predicted"].map(lambda x: int(x) if not pd.isnull(x) else None)
                            def find_percentage_diff(row):
                                res = (math.fabs(row['Actual'] - row['Predicted']) / row['Actual'] * 100)
                                out = str(round(res, 2)) + "%"
                                return out
                            perf_df['Delta %'] = perf_df.apply(find_percentage_diff, axis=1)
                            st.dataframe(perf_df)

                        
                            plot_results(test, forecast, "Actual vs Predicted")

                            st.subheader("🔮 Forecasted Results")
                            forecast_df = models[model_type].forecast(forecast_period)
                            forecast_df_res = pd.concat([df, forecast_df.to_frame(name=metric_column)])
                            st.dataframe(forecast_df_res)
                            # plot_results(forecast_df_res, forecast_df.to_frame(name=metric_column), "Historical and Forecasted Data")

                            forecast_df = models[model_type].forecast(forecast_period)
                            forecast_df = forecast_df.rename("Forecast")
                            forecast_df_res = pd.concat([df, forecast_df])
                            forecast_df_res['Forecast'] = forecast_df_res['Forecast'].map(lambda x: int(x) if not pd.isnull(x) else None)
                            forecast_df = forecast_df.map(lambda x: int(x) if not pd.isnull(x) else None)
                            # st.dataframe(forecast_df_res)

                            future_df_segregated = pd.DataFrame(forecast_df_res)
                            future_df_segregated['year'] = future_df_segregated.index.year
                            future_df_segregated['datetime'] = future_df_segregated.index.strftime('%m-%d')
                        
                            future_df_pivot = pd.pivot_table(
                                future_df_segregated,
                                values=[metric_column, 'Forecast'],
                                index='datetime',
                                columns='year',
                                aggfunc='first'
                            )
                            
                            idx = pd.IndexSlice
                            forecast_years = future_df_pivot.columns.get_level_values(1)[future_df_pivot.columns.get_level_values(0) == 'Forecast']
                            sales_years = future_df_pivot.columns.get_level_values(1)[future_df_pivot.columns.get_level_values(0) == metric_column][-1:]
                            selected_columns = future_df_pivot.loc[:, idx[[metric_column, 'Forecast'], sales_years.union(forecast_years)]]

                            def map_month(month_num):
                                month_name = calendar.month_name[month_num]
                                return month_name

                            pd.options.plotting.backend = "plotly"

                            selected_columns_flat = selected_columns.copy()
                            selected_columns_flat.columns = ['_'.join(map(str, col)).strip() for col in selected_columns.columns.values]
                            
                            fig_forecast = go.Figure()

                            months = list(range(1, 13))

                            for col in selected_columns[metric_column].columns:
                                fig_forecast.add_trace(go.Scatter(
                                    x=months,
                                    y=selected_columns[metric_column][col],
                                    mode='lines',
                                    name=col,
                                    line=dict(dash='dot', color='red')
                                ))

                            for col in selected_columns['Forecast'].columns:
                                fig_forecast.add_trace(go.Scatter(
                                    x=months,
                                    y=selected_columns['Forecast'][col],
                                    mode='lines',
                                    name=col
                                ))

                            fig_forecast.update_layout(
                                title="Forecasted Results",
                                xaxis_title="Date",
                                yaxis_title="Value",
                                xaxis=dict(
                                    tickmode='array',
                                    tickvals=months,
                                    ticktext=[map_month(tick) for tick in months]
                                )
                            )

                            st.plotly_chart(fig_forecast, use_container_width=True) 


        elif analysis_type == "Multivariate":
            metric_columns = st.sidebar.multiselect("Select metric columns (at least two)", list(df.columns))
            if len(metric_columns) >= 2:
                target_col = st.sidebar.selectbox("Select target column", metric_columns)
                if target_col:
                    df = df[metric_columns]
                    df = preprocess_data(df)
                    if df.shape[0] < 50:
                        st.write("Less than 50 rows. Please use different data.")
                    else:
                        st.subheader("Time Series Plot")
                        plot_results(df, pd.DataFrame(), "Time Series Plot", target_col)

                        st.subheader("Seasonal Decomposition of Target")
                        decomposition = seasonal_decompose(df[target_col], model='additive', period=12)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
                        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
                        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'))
                        fig.update_layout(title=f"Seasonal Decomposition of {target_col}", xaxis_title="Date", yaxis_title='Value')
                        st.plotly_chart(fig)

                        st.subheader("ACF and PACF Plots of Target")
                        plot_acf_pacf(df[target_col], lags=min(30, len(df)//2), title_prefix=target_col)

                        st.subheader("Correlation Heatmap")
                        plot_heatmap(df)

                        models = load_models(pd.Series([1,2,3]), df, "Multivariate", "")
                        model_type = st.sidebar.selectbox("Select Forecasting Model", [None] + list(models.keys()))
                        forecast_period = st.sidebar.selectbox("Select Forecast Period", [3, 6, 9, 100])

                        if model_type:
                            st.subheader("Model Metrics")
                            result_df = find_metrics(models, df[target_col], "Multivariate", target_col)
                            st.dataframe(style_df(result_df, model_type))

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
                            test, forecast, mae, rmse = models[model_type].calculate_metrics()
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
                            
                            plot_results(test, forecast, "Actual vs Predicted", target_col)

                            st.subheader("🔮 Forecasted Results")
                            forecast_df = models[model_type].forecast(forecast_period)
                            forecast_df_res = pd.concat([df[[target_col]], forecast_df[[target_col]]])
                            st.dataframe(forecast_df_res)
                            plot_results(df[[target_col]], forecast_df, "Historical and Forecasted Data", target_col)
            
            else:
                st.subheader("Correlation Heatmap")
                plot_heatmap(df)
                            
                            

else:
    # st.image("./PredictSage_logo.png", use_container_width=True)
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
        """ )
          
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