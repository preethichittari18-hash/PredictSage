import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from darts import TimeSeries
from darts.models import (
    ExponentialSmoothing as ES,
    ARIMA as ARIMADarts,
    NBEATSModel,
    Theta,
    RegressionEnsembleModel,
)
from darts.metrics import mae, rmse
from darts.utils.statistics import check_seasonality
from darts.dataprocessing.transformers import Scaler
from darts.utils.utils import SeasonalityMode
from termcolor import colored

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU , Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import random
import tensorflow as tf
import time
from functools import wraps


def time_execution(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"TIME_PERF: Model {self.__class__.__name__}'s function '{func.__name__}' took {execution_time:.4f} seconds to execute")
        return result
    return wrapper


class StatsBaseModel(ABC):
    data: pd.Series = None
    train: pd.Series = None
    test: pd.Series = None
        
    @classmethod
    def initialize_data(cls, data, **config):
        cls.data = data
        train_divide = int(0.8 * data.shape[0])
        cls.train = data.iloc[:train_divide]
        cls.test = data.iloc[train_divide:]
        cls.config = config

    @abstractmethod
    def forecast(self, n_rows: int):
        pass

    @abstractmethod
    def forecast_test(self):
        pass

    @time_execution
    def calculate_metrics(self) -> tuple[float]:
        forecast = self.forecast_test()
        mae_d = round(mean_absolute_error(self.test, forecast),2)
        rmse_d = round(np.sqrt(mean_squared_error(self.test, forecast)),2)
        return self.test, forecast, mae_d, rmse_d
    
    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        return state

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
    

class ExponentialSmoothingModel(StatsBaseModel):

    def forecast_test(self) -> pd.Series:
        model = ExponentialSmoothing(self.train, initialization_method='legacy-heuristic', seasonal='add').fit()
        forecast = model.forecast(len(self.test))
        return forecast
    @time_execution
    def forecast(self, n_rows: int) -> pd.Series:
        model = ExponentialSmoothing(self.data, initialization_method='legacy-heuristic', seasonal='add').fit()
        forecast = model.forecast(n_rows)
        return forecast


class ArimaModel(StatsBaseModel):

    def forecast_test(self) -> pd.Series:
        model = ARIMA(self.train, order=self.__class__.config.get("ArimaOrder", (1, 1, 1))).fit()
        forecast = model.forecast(len(self.test))
        return forecast
    @time_execution
    def forecast(self, n_rows: int) -> pd.Series:
        model = ARIMA(self.data, order=self.__class__.config.get("ArimaOrder", (1, 1, 1))).fit()
        forecast = model.forecast(n_rows)
        return forecast


class AutoArimaModel(StatsBaseModel):

    def forecast_test(self) -> pd.Series:
        model = auto_arima(self.train, seasonal=True,m=self.__class__.config.get("AutoArimaSeasonality",12))
        forecast = model.predict(len(self.test))
        return forecast
    @time_execution
    def forecast(self, n_rows: int) -> pd.Series:
        model = auto_arima(self.data, seasonal=True,m=self.__class__.config.get("AutoArimaSeasonality",12) )
        forecast = model.predict(n_rows)
        return forecast


class SarimaModel(StatsBaseModel):

    def forecast_test(self) -> pd.Series:
        model = SARIMAX(self.train, order=self.__class__.config.get("SarimaOrder", (1, 1, 1)), seasonal_order=self.__class__.config.get("SeasonalityOrder", (1, 1, 1, 12))).fit()
        forecast = model.forecast(len(self.test))
        return forecast
    @time_execution 
    def forecast(self, n_rows: int) -> pd.Series:
        model = SARIMAX(self.data, order=self.__class__.config.get("SarimaOrder", (1, 1, 1)), seasonal_order=self.__class__.config.get("SeasonalityOrder", (1, 1, 1, 12))).fit()
        forecast = model.forecast(n_rows)
        return forecast


class DartsBaseModel(ABC):
    data: pd.Series = None
    series: TimeSeries = None
    train: TimeSeries = None
    test: TimeSeries = None
    scaler_test: Scaler = Scaler()
    scaler_full: Scaler = Scaler()
    series_scaled: TimeSeries = None
    train_scaled: TimeSeries = None
    test_scaled: TimeSeries = None

    @classmethod
    def initialize_data(cls, data: pd.Series, **config: dict):
        cls.data = data
        cls.series = TimeSeries.from_series(data)
        cls.train, cls.test = cls.series.split_before(0.8)
        cls.series_scaled = cls.scaler_full.fit_transform(cls.series)
        cls.train_scaled = cls.scaler_test.fit_transform(cls.train)
        cls.test_scaled = cls.scaler_test.fit_transform(cls.test)
        cls.config = config

    @abstractmethod
    def forecast(self, n_rows):
        pass

    @abstractmethod
    def forecast_test(self):
        pass
    @time_execution
    def calculate_metrics(self):
        forecast = self.forecast_test()
        mae_d = mae(self.test, forecast)
        rmse_d = rmse(self.test, forecast)
        return self.test.pd_series(), forecast.pd_series(), mae_d, rmse_d
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the scaler objects if they cause issues
        if 'scaler_test' in state:
            del state['scaler_test']
        if 'scaler_full' in state:
            del state['scaler_full']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize scalers
        self.scaler_test = Scaler()
        self.scaler_full = Scaler()

class ThetaModel(DartsBaseModel):
    def forecast_test(self):
        model = Theta(theta=1, season_mode=SeasonalityMode.ADDITIVE).fit(self.train_scaled)
        forecast = self.scaler_test.inverse_transform(model.predict(len(self.test)))
        return forecast
    @time_execution
    def forecast(self, n_rows):
        model = Theta(theta=1, season_mode=SeasonalityMode.ADDITIVE).fit(self.series_scaled)
        forecast = self.scaler_full.inverse_transform(model.predict(n_rows))
        return forecast.pd_series()
    

class NbeatsModel(DartsBaseModel):
    def forecast_test(self):
        model = NBEATSModel(
            input_chunk_length=12, output_chunk_length=len(self.test), n_epochs=self.__class__.config.get("nbeatsepochs",50), random_state=42
        ).fit(self.train_scaled)
        forecast = self.scaler_test.inverse_transform(model.predict(len(self.test)))
        return forecast
    @time_execution
    def forecast(self, n_rows):
        model = NBEATSModel(
            input_chunk_length=12, output_chunk_length=n_rows, n_epochs=self.__class__.config.get("nbeatsepochs",50), random_state=42
        ).fit(self.series_scaled)
        forecast = self.scaler_full.inverse_transform(model.predict(n_rows))
        return forecast.pd_series()


class EnsembleModel(DartsBaseModel):
    def forecast_test(self):
        model_es = ES(seasonal=SeasonalityMode.ADDITIVE, seasonal_periods=12)
        model_arima = ARIMADarts()
        model_nbeats = NBEATSModel(
            input_chunk_length=12, output_chunk_length=6, n_epochs=self.__class__.config.get("Ensembleepochs",50), random_state=42
        )
        model_theta = Theta(theta=1,season_mode=SeasonalityMode.ADDITIVE)
        models = [model_es, model_arima, model_nbeats, model_theta]
        
        model = RegressionEnsembleModel(forecasting_models=models, regression_train_n_points=12)
        model.fit(self.train_scaled)
        forecast = self.scaler_test.inverse_transform(model.predict(len(self.test)))
        return forecast
    @time_execution
    def forecast(self, n_rows):
        model_es = ES(seasonal=SeasonalityMode.ADDITIVE, seasonal_periods=12)
        model_arima = ARIMADarts()
        model_nbeats = NBEATSModel(
            input_chunk_length=12, output_chunk_length=6, n_epochs=self.__class__.config.get("Ensembleepochs",50), random_state=42
        )
        model_theta = Theta(theta=1,season_mode=SeasonalityMode.ADDITIVE)
        models = [model_es, model_arima, model_nbeats, model_theta]
        
        model = RegressionEnsembleModel(forecasting_models=models, regression_train_n_points=12)
        model.fit(self.series_scaled)
        forecast = self.scaler_test.inverse_transform(model.predict(n_rows))
        return forecast.pd_series()


class TfBaseModel(ABC):
    data: pd.Series = None
    train: pd.Series = None
    test: pd.Series = None
    scaler: MinMaxScaler = MinMaxScaler()
    data_scaled: np.ndarray = None
    train_scaled: np.ndarray = None
    test_scaled: np.ndarray = None
    n_steps = 10

    @classmethod
    def initialize_data(cls, data: pd.Series, **config: dict):
        cls.data = data
        train_divide = int(0.8 * data.shape[0])
        cls.train = data.iloc[:train_divide]
        cls.test = data.iloc[train_divide:]
        cls.data_scaled = cls.scaler.fit_transform(cls.data.values.reshape(-1, 1))
        cls.train_scaled = cls.scaler.fit_transform(cls.train.values.reshape(-1, 1))
        cls.test_scaled = cls.scaler.transform(cls.test.values.reshape(-1, 1))
        cls.config = config

    @abstractmethod
    def build_model(self, input_shape):
        pass

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='mse')

    def fit_model(self, X_train, y_train, epochs=100, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def prepare_data(self, series, n_steps):
        X, y = [], []
        for i in range(len(series)):
            end_ix = i + n_steps
            if end_ix > len(series) - 1:
                break
            seq_x, seq_y = series[i:end_ix], series[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def forecast_test(self) -> pd.Series:
        X_train, y_train = self.prepare_data(self.train_scaled.flatten(), self.n_steps)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.build_model((X_train.shape[1], X_train.shape[2]))
        self.compile_model()
        self.fit_model(X_train, y_train, epochs=self.__class__.config.get("NNEpochs", 100))
        X_test, _ = self.prepare_data(self.test_scaled.flatten(), self.n_steps)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        forecast = self.model.predict(X_test)
        forecast = self.scaler.inverse_transform(forecast)

        freq = self.data.index.freq if self.data.index.freq else pd.infer_freq(self.data.index)
        new_dates = pd.date_range(start=self.test.index[self.n_steps-1], periods=self.test.shape[0]-self.n_steps+1, freq=freq)[1:]
        return pd.Series(forecast.flatten(), index=new_dates)
    @time_execution
    def forecast(self, n_rows: int) -> pd.Series:
        X_train, y_train = self.prepare_data(self.data_scaled, self.n_steps)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.build_model((X_train.shape[1], X_train.shape[2]))
        self.compile_model()
        self.fit_model(X_train, y_train, epochs=self.__class__.config.get("NNEpochs", 100))
        
   
        X_input = self.data_scaled[-self.n_steps:].reshape((1, self.n_steps, 1))
        forecast = []
        for _ in range(n_rows):
            yhat = self.model.predict(X_input, verbose=0)
            forecast.append(yhat[0, 0])
            X_input = np.append(X_input[:, 1:, :], yhat.reshape((1, 1, 1)), axis=1)

        forecast = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            
        freq = self.data.index.freq if self.data.index.freq else pd.infer_freq(self.data.index)
        new_dates = pd.date_range(start=self.data.index[-1], periods=n_rows+1, freq=freq)[1:]
        
        return pd.Series(forecast.flatten(), index=new_dates)
    @time_execution
    def calculate_metrics(self) -> tuple[float]:
        forecast = self.forecast_test()
        mae_d = mean_absolute_error(self.test[self.n_steps:], forecast)
        rmse_d = np.sqrt(mean_squared_error(self.test[self.n_steps:], forecast))
        return self.test[self.n_steps:], forecast, mae_d, rmse_d
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'model' in state:
            del state['model']
        if 'scaler' in state:
            del state['scaler']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
       
        self.scaler = MinMaxScaler()


class RnnModel(TfBaseModel):
    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(SimpleRNN(50, activation='relu', input_shape=input_shape))
        self.model.add(Dense(1))


class LstmModel(TfBaseModel):
    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(50,activation='relu',return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(50,activation='relu',return_sequences=False))
        self.model.add(Dense(1))

class MultivariateStatsBaseModel(ABC):
    data: pd.DataFrame = None
    train: pd.DataFrame = None
    test: pd.DataFrame = None
 
    @classmethod
    def initialize_data(cls, data: pd.DataFrame):
        cls.data = data
        train_divide = int(0.8 * data.shape[0])
        cls.train = data.iloc[:train_divide]
        cls.test = data.iloc[train_divide:]
 
    @abstractmethod
    def forecast(self, n_rows: int):
        pass
 
    @abstractmethod
    def forecast_test(self):
        pass
 
    @time_execution
    def calculate_metrics(self) -> tuple:
        forecast = self.forecast_test()
        mae_d = {col: round(mean_absolute_error(self.test[col], forecast[col]), 2) for col in self.test.columns}
        rmse_d = {col: round(np.sqrt(mean_squared_error(self.test[col], forecast[col])), 2) for col in self.test.columns}
        return self.test, forecast, mae_d, rmse_d
    
class VarModel(MultivariateStatsBaseModel):
    def forecast_test(self) -> pd.DataFrame:
        model = VAR(self.train).fit(maxlags=30)
        forecast = model.forecast(self.train.values[-model.k_ar:], steps=len(self.test))
        return pd.DataFrame(forecast, index=self.test.index, columns=self.data.columns)
 
    @time_execution
    def forecast(self, n_rows: int) -> pd.DataFrame:
        model = VAR(self.data).fit(maxlags=30)
        forecast = model.forecast(self.data.values[-model.k_ar:], steps=n_rows)
        freq = self.data.index.freq if self.data.index.freq else pd.infer_freq(self.data.index)
        new_dates = pd.date_range(start=self.data.index[-1], periods=n_rows + 1, freq=freq)[1:]
        return pd.DataFrame(forecast, index=new_dates, columns=self.data.columns)
 
class VarmaxModel(MultivariateStatsBaseModel):
    def forecast_test(self) -> pd.DataFrame:
        model = VARMAX(self.train, order=(1, 1)).fit(disp=False)
        forecast = model.forecast(steps=len(self.test))
        return pd.DataFrame(forecast, index=self.test.index, columns=self.data.columns)
 
    @time_execution
    def forecast(self, n_rows: int) -> pd.DataFrame:
        model = VARMAX(self.data, order=(1, 1)).fit(disp=False)
        forecast = model.forecast(steps=n_rows)
        freq = self.data.index.freq if self.data.index.freq else pd.infer_freq(self.data.index)
        new_dates = pd.date_range(start=self.data.index[-1], periods=n_rows + 1, freq=freq)[1:]
        return pd.DataFrame(forecast, index=new_dates, columns=self.data.columns)
    
class MultivariateTfBaseModel(ABC):
    data: pd.DataFrame = None
    train: pd.DataFrame = None
    test: pd.DataFrame = None
    scaler: MinMaxScaler = MinMaxScaler()
    data_scaled: np.ndarray = None
    train_scaled: np.ndarray = None
    test_scaled: np.ndarray = None
    n_steps = 20
 
    @classmethod
    def initialize_data(cls, data: pd.DataFrame):
        cls.data = data
        train_divide = int(0.8 * data.shape[0])
        cls.train = data.iloc[:train_divide]
        cls.test = data.iloc[train_divide:]
        cls.data_scaled = cls.scaler.fit_transform(cls.data)
        cls.train_scaled = cls.scaler.fit_transform(cls.train)
        cls.test_scaled = cls.scaler.transform(cls.test)
 
    @abstractmethod
    def build_model(self, input_shape):
        pass
 
    def compile_model(self):
        self.model.compile(optimizer='adam', loss='mse')
 
    def fit_model(self, X_train, y_train, epochs=200, batch_size=32):
        early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False, callbacks=[early_stopping])
        # self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
 
    def prepare_data(self, series: np.ndarray, n_steps: int):
        X, y = [], []
        for i in range(len(series) - n_steps):
            X.append(series[i:i + n_steps])
            y.append(series[i + n_steps])
        return np.array(X), np.array(y)
 
    def forecast_test(self) -> pd.DataFrame:
        X_train, y_train = self.prepare_data(self.train_scaled, self.n_steps)
        self.build_model((self.n_steps, self.train_scaled.shape[1]))
        self.compile_model()
        self.fit_model(X_train, y_train)
        X_test, _ = self.prepare_data(self.test_scaled, self.n_steps)
        forecast = self.model.predict(X_test)
        forecast = self.scaler.inverse_transform(forecast)
        freq = self.data.index.freq if self.data.index.freq else pd.infer_freq(self.data.index)
        new_dates = pd.date_range(start=self.test.index[self.n_steps], periods=len(forecast), freq=freq)
        return pd.DataFrame(forecast, index=new_dates, columns=self.data.columns)
 
    @time_execution
    def forecast(self, n_rows: int) -> pd.DataFrame:
        X_train, y_train = self.prepare_data(self.data_scaled, self.n_steps)
        self.build_model((self.n_steps, self.data_scaled.shape[1]))
        self.compile_model()
        self.fit_model(X_train, y_train)
        X_input = self.data_scaled[-self.n_steps:].reshape((1, self.n_steps, self.data_scaled.shape[1]))
        forecast = []
        for _ in range(n_rows):
            yhat = self.model.predict(X_input, verbose=0)
            forecast.append(yhat[0])
            X_input = np.append(X_input[:, 1:, :], yhat.reshape((1, 1, self.data_scaled.shape[1])), axis=1)
        forecast = self.scaler.inverse_transform(np.array(forecast))
        freq = self.data.index.freq if self.data.index.freq else pd.infer_freq(self.data.index)
        new_dates = pd.date_range(start=self.data.index[-1], periods=n_rows + 1, freq=freq)[1:]
        return pd.DataFrame(forecast, index=new_dates, columns=self.data.columns)
 
    @time_execution
    def calculate_metrics(self) -> tuple:
        forecast = self.forecast_test()
        mae_d = {col: round(mean_absolute_error(self.test[col].iloc[self.n_steps:], forecast[col]), 2) for col in self.test.columns}
        rmse_d = {col: round(np.sqrt(mean_squared_error(self.test[col].iloc[self.n_steps:], forecast[col])), 2) for col in self.test.columns}
        return self.test.iloc[self.n_steps:], forecast, mae_d, rmse_d
 
class GruModel(MultivariateTfBaseModel):
    def build_model(self, input_shape):
        self.model = Sequential([
            GRU(200, activation='tanh', input_shape=input_shape),
            Dense(self.data.shape[1])  
        ])

class MvRnnModel(MultivariateTfBaseModel):
    def build_model(self, input_shape):
        self.model = Sequential([
            SimpleRNN(100, activation='relu', input_shape=input_shape),
            Dense(self.data.shape[1]) 
        ])

class MvLstmModel(MultivariateTfBaseModel):
    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(100, activation='tanh', input_shape=input_shape),
            Dense(self.data.shape[1])
        ])

if __name__ == '__main__':
    df = pd.read_csv('files/sales dataa.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    
    TfBaseModel.initialize_data(df['Perrin Freres monthly champagne sales millions ?64-?72'])
    model = LstmModel()
    print(model.calculate_metrics())
    print(model.forecast(10))
    