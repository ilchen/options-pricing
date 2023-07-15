# coding: utf-8
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas.tseries.offsets import BDay
from math import sqrt, log, exp


class VolatilityTracker:
    """
    Represents a tracker for volatility and for forecasting future volatilities of a given asset. You instantiate an
    object of this class with a consecutive range of daily closing prices of a given asset, and then use it to inspect
    past daily and annual volatilities as well as to predict future volatilities.

    It forms a foundation for pricing options.
    """

    CLOSE = 'Close'
    DAILY_RETURN = 'ui'
    VARIANCE = 'Variance'
    TRADING_DAYS_IN_YEAR = 252
    TO_ANNUAL_MULTIPLIER = sqrt(TRADING_DAYS_IN_YEAR)

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        """
        Calculates daily volatilities from either a panda series object indexed by dates
        (i.e. asset_prices_series != None) or from a date range and a desired asset class (i.e. the 'start' abd 'end'
        arguments must be provided)
        :param asset_prices_series: a pandas Series object indexed by dates
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        :param asset: the ticker symbol of the asset whose asset price changes are to be analyzed. It expects
                      a Yahoo Finance convention for ticker symbols
        """
        if asset_prices_series is None:
            if start is None or end is None or asset is None:
                raise ValueError("Neither asset_price_series nor (start, end, asset) arguments are provided")
            data = web.get_data_yahoo(asset, start, end)
            asset_prices_series = data['Adj Close']
        # Dropping the first row as it doesn't contain a daily return value
        self.data = pd.DataFrame({self.CLOSE: asset_prices_series, self.DAILY_RETURN: asset_prices_series.pct_change()},
                                 index=asset_prices_series.index).iloc[1:]
        # Essentially only self.data[self.VARIANCE].iloc[1] needs to be set to self.data.ui.iloc[0]**2
        self.data[self.VARIANCE] = self.data.ui.iloc[0] ** 2

    def get_daily_volatilities(self):
        """
        Calculates the past volatilities for consecutive range of dates captured in self.data.index
        """
        return np.sqrt(self.data[self.VARIANCE].iloc[1:])

    def get_annual_volatilities(self):
        return self.get_daily_volatilities() * self.TO_ANNUAL_MULTIPLIER

    def get_annual_volatilities_for_dates(self, dates):
        """
        :dates the dates for which to obtain closing prices represented by a DatetimeIndex object
        """
        return self.get_daily_volatilities()[dates] * self.TO_ANNUAL_MULTIPLIER

    def get_dates(self):
        return self.data.index[1:]

    def get_adj_close_prices(self):
        return self.data[self.CLOSE].values[1:]

    def get_adj_close_prices_for_dates(self, dates):
        """
        :dates the dates for which to obtain closing prices represented by a DatetimeIndex object
        """
        return self.data[self.CLOSE][dates].values

    def get_next_business_day_volatility(self):
        """
        Returns the daily volatility on the next business day after the last day in the self.data.index
        """
        return pd.Series([None], index=[self.data.index[-1] + BDay()], dtype=self.data[self.VARIANCE].dtype)

    def get_next_business_day_annual_volatility(self):
        """
        Returns the annual volatility on the next business day after the last day in the self.data.index
        """
        return self.get_next_business_day_volatility() * self.TO_ANNUAL_MULTIPLIER

    def get_volatility_forecast(self, n):
        """
        :param n: an integer indicating for which business day in the future daily volatility should be forecast
        """
        return pd.Series([None], index=[self.data.index[-1] + n*BDay()], dtype=self.data[self.VARIANCE].dtype)

    def get_annual_volatility_forecast(self, n):
        """
        :param n: an integer indicating for which business day in the future annual volatility should be forecast
        """
        return self.get_volatility_forecast(n) * self.TO_ANNUAL_MULTIPLIER

    def get_volatility_forecast_for_next_n_days(self, n):
        """
        :param n: an integer indicating for how many business days in the future daily volatility should be forecast
        """
        return pd.concat([self.get_volatility_forecast(d) for d in range(1, n)])

    def get_annual_volatility_forecast_for_next_n_days(self, n):
        """
        :param n: an integer indicating for how many business days in the future annual volatility should be forecast
        """
        return self.get_volatility_forecast_for_next_n_days(n) * self.TO_ANNUAL_MULTIPLIER

    def get_term_volatility_forecast(self, t):
        """
        :param t: a float indicating for which future term (expressed in years) average volatility needs to be forecast.

        This is a key method for pricing options.
        """
        raise NotImplementedError

    def get_annual_term_volatility_forecast(self, t):
        """
        :param t: a float indicating for which future term (expressed in years) average volatility needs to be forecast.

        This is a key method for pricing options.
        """
        return self.get_term_volatility_forecast(t) * self.TO_ANNUAL_MULTIPLIER


class EWMAVolatilityTracker(VolatilityTracker):
    """
    Represents an Exponentially-Weighted Moving Average volatility tracker with a given λ parameter
    """

    def __init__(self, lamda, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)
        self.lamda = lamda

        # Unfortunately not vectorizable as the next value depends on the previous
        # self.data[self.VARIANCE].iloc[1:] = (1 - λ) * self.data[self.DAILY_RETURN].iloc[:-1]**2\
        #                                     + λ * self.data[self.VARIANCE].iloc[:-1]
        for i in range(2, len(self.data[self.DAILY_RETURN])):
            if np.isinf(self.data[self.DAILY_RETURN].iloc[i-1]):
                self.data[self.VARIANCE].iloc[i] = self.data[self.VARIANCE].iloc[i-1]
            else:
                self.data[self.VARIANCE].iloc[i] = (1-lamda) * self.data[self.DAILY_RETURN].iloc[i-1] ** 2 \
                                                   + lamda * self.data[self.VARIANCE].iloc[i-1]

    def get_next_business_day_volatility(self):
        s = super().get_next_business_day_volatility()
        last_idx = len(self.data) - 1
        s[0] = np.sqrt((1 - self.lamda) * self.data[self.DAILY_RETURN].iloc[last_idx] ** 2
                       + self.lamda * self.data[self.VARIANCE].iloc[last_idx])
        return s

    def get_volatility_forecast(self, n):
        """
        For EMWA the forecast for n business days in the future is the same as for the next business day
        """
        s = super().get_volatility_forecast(n)
        s[0] = self.get_next_business_day_volatility().values[0]
        return s

    def get_term_volatility_forecast(self, t):
        """
        For EMWA the forecast for a period of t years in the future is the same as for the next business day
        """
        return self.get_next_business_day_volatility()


class GARCHVolatilityTracker(VolatilityTracker):
    """
    Represents a GARCH(1, 1) volatility tracker with given ω, α, and β parameters
    """

    def __init__(self, omega, alpha, beta, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

        # Unfortunately not vectorizable as the next value depends on the previous
        # self.data[self.VARIANCE].iloc[1:] = ω + α * self.data[self.DAILY_RETURN].iloc[:-1]**2\
        #                                       + β * self.data[self.VARIANCE].iloc[:-1]
        for i in range(2, len(self.data[self.DAILY_RETURN])):
            if np.isinf(self.data[self.DAILY_RETURN].iloc[i - 1]):
                self.data[self.VARIANCE].iloc[i] = self.data[self.VARIANCE].iloc[i - 1]
            else:
                self.data[self.VARIANCE].iloc[i] = omega + alpha * self.data[self.DAILY_RETURN].iloc[i - 1] ** 2 \
                                                   + beta * self.data[self.VARIANCE].iloc[i - 1]

    def get_vl(self):
        """
        Returns the long-term daily variance rate
        """
        return self.omega / (1 - self.alpha - self.beta)

    def get_long_term_volatility(self):
        """
        Returns the long-term annual volatility
        """
        return sqrt(self.get_vl())

    def get_annual_long_term_volatility(self):
        """
        Returns the long-term annual volatility
        """
        return sqrt(self.get_vl()) * self.TO_ANNUAL_MULTIPLIER

    def get_next_business_day_volatility(self):
        s = super().get_next_business_day_volatility()
        last_idx = len(self.data) - 1
        s[0] = np.sqrt(self.omega + self.alpha * self.data[self.DAILY_RETURN].iloc[last_idx] ** 2
                       + self.beta * self.data[self.VARIANCE].iloc[last_idx])
        return s

    def get_volatility_forecast(self, n):
        s = super().get_volatility_forecast(n)
        vl = self.get_vl()
        next_bd_variance = self.get_next_business_day_volatility().values[0]**2
        s[0] = np.sqrt(vl + (self.alpha + self.beta)**n * (next_bd_variance - vl))
        return s

    def get_term_volatility_forecast(self, t):
        a = log(1/(self.alpha + self.beta))
        vl = self.get_vl()
        t_in_days = t * VolatilityTracker.TRADING_DAYS_IN_YEAR
        next_bd_variance = self.get_next_business_day_volatility().values[0] ** 2
        return sqrt(vl + (1 - exp(-a * t_in_days)) / (a * t_in_days) * (next_bd_variance - vl))\
            if t > 9.1e-5 else self.get_next_business_day_volatility()[0] # 9.1e-5 is about 8 hours
