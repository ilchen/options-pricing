# coding: utf-8
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint


class ParameterEstimator:
    """
    A base class for estimators of volatility forecasting parameters
    """

    CLOSE = 'Close'
    DAILY_RETURN = 'ui'
    VARIANCE = 'Variance'
    NUM_COLUMNS = 3  # Number of DataFrame columns per asset

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        """
        Constructs a volatility estimator object from either a panda Series or DataFrame object indexed by dates
        (i.e. asset_prices_series != None) or from a date range and a list of desired assets (in which case the
        'start' and 'end' arguments must be provided).

        You would typically estimate for multiple asset classes (by providing a DataFrame or a list of assets)
        when you want to construct a variance-covariance matrix, in which case it is essential that all volatilities
        and correlations be tracked using the same parameters, otherwise the matrix will not be internally consistent.

        :param asset_prices_series: a pandas Series object indexed by dates when we need to estimate a volatility
                                    forecasting parameter for one asset class or a DataFrame when estimating for
                                    multiple asset classes
        :param start: (string, int, date, datetime, Timestamp) – Starting date. Parses many different kind of date
                       representations (e.g., ‘JAN-01-2010’, ‘1/1/10’, ‘Jan, 1, 1980’). Defaults to 5 years before
                       current date.
        :param end: (string, int, date, datetime, Timestamp) – Ending date
        :param asset: the ticker symbol of the asset whose asset price changes are to be analyzed or an array of ticker
                      symbols. It expects a Yahoo Finance convention for ticker symbols
        """
        if asset_prices_series is None:
            if start is None or end is None or asset is None:
                raise ValueError("Neither asset_price_series nor (start, end, asset) arguments are provided")
            data = web.get_data_yahoo(asset, start, end)
            asset_prices_series = data['Adj Close']

        # Dropping the first row as it doesn't contain a daily return value
        if isinstance(asset_prices_series, pd.Series):
            self.number_assets = 1
            self.data = pd.DataFrame({self.CLOSE: asset_prices_series, self.DAILY_RETURN: asset_prices_series.pct_change()},
                                     index=asset_prices_series.index).iloc[1:]
            # Essentially only self.data[self.VARIANCE].iloc[1] needs to be set to self.data.ui.iloc[0]**2
            self.data[self.VARIANCE] = self.data.ui.iloc[0] ** 2
        else:
            # Produces a DataFrame with columns:
            # asset_class_1, asset_class_1ui,asset_class1Variance, asset_class_2, asset_class_2ui, asset_class_2Variance
            self.number_assets = len(asset_prices_series.columns)
            self.data = asset_prices_series.copy()
            for i in range(self.number_assets):
                uis = self.data.iloc[:,i*3].dropna().pct_change()
                self.data.insert(loc=i*3+1, column=self.data.columns[i*3]+self.DAILY_RETURN, value=uis)
                self.data.insert(loc=i*3+2, column=self.data.columns[i*3]+self.VARIANCE, value=uis[1] ** 2)
            self.data = self.data.iloc[1:]

            # Get rid of rows whose percentage changes are infinite
            self.data = self.data[np.isfinite(self.data)].dropna()


class GARCHParameterEstimator(ParameterEstimator):
    """
    A maximum likelihood estimator for the ω, α, and β parameters of the GARCH(1, 1) model of forecasting volatility
    """
    # Ensuring that ω, α, and β values we will search for have roughly equal values in terms of magnitude
    GARCH_PARAM_MULTIPLIERS = np.array([1e5, 10, 1], dtype=np.float64)

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)

        # Initial values for ω, α, and β parameters for GARCH
        x0 = np.array([1e-5, .1, .87], dtype=np.float64)

        def objective_func(x):
            """ This function searches for optimal values of the ω, α, and β parameters of the GARCH(1, 1)
            model given the sample of asset price changes stored in the 'self.data' DataFrame. Since SciPy only has
            optimization routines that minimize an objective function, this function negates the value of the
            log likelihood objective function for GARCH upon return.
            :param x: a tuple of the ω, α, and β parameters where ω, α, and β
                      should be appropriately scaled to be in approximately the same range. This greatly aids the
                      speed of optimization
            """
            ω, α, β = x / GARCHParameterEstimator.GARCH_PARAM_MULTIPLIERS

            # Unfortunately not vectorizable as the next value depends on the previous
            # self.data[self.VARIANCE].iloc[1:] = ω + α * self.data[self.DAILY_RETURN].iloc[:-1]**2\
            #                                       + β * self.data[self.VARIANCE].iloc[:-1]
            # for i in range(2, len(self.data)):
            #     for j in range(self.number_assets):
            #         self.data.iloc[i, j*3+2] = ω + α * self.data.iloc[i-1, j*3+1] ** 2 \
            #                                        + β * self.data.iloc[i-1, j*3+2]
            # sum = 0.
            # for j in range(self.number_assets):
            #     sum -= (-np.log(self.data.iloc[1:, j*3+2]) -
            #             self.data.iloc[1:, j*3+1] ** 2 / self.data.iloc[1:, j*3+2]).sum()

            # Catering to a case where some series in a DataFrame may have NaNs due to different trading days
            sum = 0.
            for j in range(self.number_assets):
                df_copy = self.data.iloc[:, j*3+1:j*3+3]
                if self.number_assets > 1:
                    df_copy = df_copy.dropna()
                for i in range(2, len(df_copy)):
                    # Skipping values that would lead to a propagation of infinity
                    if np.isinf(df_copy.iloc[i-1, 0]):
                        df_copy.iloc[i, 1] = df_copy.iloc[i-1, 1]
                    else:
                        df_copy.iloc[i, 1] = ω + α * df_copy.iloc[i-1, 0] ** 2 + β * df_copy.iloc[i-1, 1]
                # No need to take the first variance value into account as it doesn't depend on ω, α, β,
                # hence starting from the second row
                sum -= (-np.log(df_copy.iloc[2:, 1]) - df_copy.iloc[2:, 0] ** 2 / df_copy.iloc[2:, 1]).sum()

            return sum

        # print('Starting with objective function value of:', -objective_func(x0 * self.GARCH_PARAM_MULTIPLIERS))

        # omega [0;1], alpha[0;1], beta[0;1]
        bounds = Bounds([0., 0., 0.], np.array([1., 1., 1.]) * self.GARCH_PARAM_MULTIPLIERS)

        # 0*omega + alpha + beta <= 1
        constr = LinearConstraint([[0, 1 / self.GARCH_PARAM_MULTIPLIERS[1], 1 / self.GARCH_PARAM_MULTIPLIERS[2]]],
                                  [0], [1])

        # Can pass a Hessian matrix of zero as the objective function is linear with respect to ω, α, and β.
        # However, the optimization process then takes 3 times as long, but removes the warning.
        # Better to suppress it altogether.
        import warnings
        warnings.filterwarnings('ignore', message='delta_grad == 0.0. Check if the approximated function is linear.')

        res = minimize(objective_func, x0 * self.GARCH_PARAM_MULTIPLIERS, method='trust-constr',
                       bounds=bounds, constraints=constr)#, hess=lambda x: np.zeros((len(x0), len(x0))))
        if res.success:
            ω, α, β = res.x / self.GARCH_PARAM_MULTIPLIERS
            print('Objective function: %.5f after %d iterations' % (-res.fun, res.nit))
            self.omega = ω
            self.alpha = α
            self.beta = β
        else:
            raise ValueError("Optimizing the objective function with the passed asset price changes didn't succeed")


class GARCHVarianceTargetingParameterEstimator(ParameterEstimator):
    """
    A maximum likelihood estimator for the ω, α, and β parameters of the GARCH(1, 1) model of forecasting volatility.
    In contrast to the GARCHParameterEstimator class, it's faster because it sets ω based on the sample variance
    of price changes. Then it optimises for only two variables (α, β) instead of 3 (α, β, ω) as GARCHParameterEstimator.
    This class cannot be used to estimate GARCH parameters of multiple assets
    """
    # Ensuring that ω and β values we will search for have roughly equal values in terms of magnitude
    GARCH_PARAM_MULTIPLIERS = np.array([1e5, 1], dtype=np.float64)

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)
        if self.number_assets > 1:
            raise ValueError('Cannot use GARCHVarianceTargetingParameterEstimator for estimating GARCH(1,1) parameters'
                             ' of more than one asset')

        # Initial values for ω and β parameters for GARCH
        x0 = np.array([1e-5, .87], dtype=np.float64)

        vl = self.data.iloc[:, 1].var()   # sample variance

        def objective_func(x):
            """ This function searches for optimal values of the ω and β parameters of the GARCH(1, 1)
            model given the sample of asset price changes stored in the 'self.data' DataFrame. Since SciPy only has
            optimization routines that minimize an objective function, this function negates the value of the
            log likelihood objective function for GARCH upon return.
            :param x: a tuple of the ω, α, and β parameters where ω, α, and β
                      should be appropriately scaled to be in approximately the same range. This greatly aids the
                      speed of optimization
            """
            ω, β = x / GARCHVarianceTargetingParameterEstimator.GARCH_PARAM_MULTIPLIERS
            α = 1 - β - ω / vl

            # Unfortunately not vectorizable as the next value depends on the previous
            # self.data[self.VARIANCE].iloc[1:] = ω + α * self.data[self.DAILY_RETURN].iloc[:-1]**2\
            #                                       + β * self.data[self.VARIANCE].iloc[:-1]
            # for i in range(2, len(self.data)):
            #     for j in range(self.number_assets):
            #         self.data.iloc[i, j*3+2] = ω + α * self.data.iloc[i-1, j*3+1] ** 2 \
            #                                        + β * self.data.iloc[i-1, j*3+2]
            # sum = 0.
            # for j in range(self.number_assets):
            #     sum -= (-np.log(self.data.iloc[1:, j*3+2]) -
            #             self.data.iloc[1:, j*3+1] ** 2 / self.data.iloc[1:, j*3+2]).sum()

            # Catering to a case where some series in a DataFrame may have NaNs due to different trading days
            sum = 0.
            for i in range(2, len(self.data)):
                # Skipping values that would lead to a propagation of infinity
                if np.isinf(self.data.iloc[i-1, 1]):
                    self.data.iloc[i, 2] = self.data.iloc[i-1, 2]
                else:
                    self.data.iloc[i, 2] = ω + α * self.data.iloc[i-1, 1] ** 2 + β * self.data.iloc[i-1, 2]
            # No need to take the first variance value into account as it doesn't depend on ω, α, β,
            # hence starting from the second row
            sum -= (-np.log(self.data.iloc[2:, 2]) - self.data.iloc[2:, 1] ** 2 / self.data.iloc[2:, 2]).sum()

            return sum

        # print('Starting with objective function value of:', -objective_func(x0 * self.GARCH_PARAM_MULTIPLIERS))

        # ω[0;1], β[0;1]
        bounds = Bounds([0., 0.], np.array([1., 1.]) * self.GARCH_PARAM_MULTIPLIERS)

        # 0 <= ω/vl + β <=1
        constr = LinearConstraint([[1 / (vl * self.GARCH_PARAM_MULTIPLIERS[0]), 1 / self.GARCH_PARAM_MULTIPLIERS[1]]],
                                  [0], [1])

        # Can pass a Hessian matrix of zero as the objective function is linear with respect to ω, α, and β.
        # However, the optimization process then takes 3 times as long, but removes the warning.
        # Better to suppress it altogether.
        import warnings
        warnings.filterwarnings('ignore', message='delta_grad == 0.0. Check if the approximated function is linear.')

        # L-BFGS-B is extremely fast, yet may produce negative values for α while optimizing, whereby triggering
        # np.log errors. Resultant values are accurate though
        # res = minimize(objective_func, x0 * self.GARCH_PARAM_MULTIPLIERS, method='L-BFGS-B', bounds=bounds)

        res = minimize(objective_func, x0 * self.GARCH_PARAM_MULTIPLIERS, method='trust-constr',
                       bounds=bounds, constraints=constr)
        if res.success:
            ω, β = res.x / self.GARCH_PARAM_MULTIPLIERS
            print('Objective function: %.5f after %d iterations' % (-res.fun, res.nit))
            self.omega = ω
            self.alpha = 1 - β - ω / vl
            self.beta = β
        else:
            raise ValueError("Optimizing the objective function with the passed asset price changes didn't succeed")


class EWMAParameterEstimator(ParameterEstimator):
    """
    A maximum likelihood estimator for the λ parameter of the EWMA method of forecasting volatility
    """

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X'):
        super().__init__(asset_prices_series, start, end, asset)

        # Initial value of λ for EWMA
        λ = .9

        def objective_func(λ):
            """ This function searches for optimal values of the λ parameter of the EWMA
            model given the sample of asset price changes stored in the 'self.data' DataFrame. Since SciPy only has
            optimization routines that minimize an objective function, this function negates the value of the
            log likelihood objective function for EWMA upon return.
            :param λ: the λ parameter in EWMA method of estimating volatility
            """

            # Unfortunately not vectorizable as the next value depends on the previous
            # self.data[self.VARIANCE].iloc[1:] = (1 - λ) * self.data[self.DAILY_RETURN].iloc[:-1]**2\
            #                                     + λ * self.data[self.VARIANCE].iloc[:-1]
            # for i in range(1, len(self.data)):
            #     for j in range(self.number_assets):
            #         # We have 3 columns per asset, the first contains closing prices, the second percentage changes,
            #         # and the third the actual variance. So j*3+2 is variance and j*3+1 percentage changes
            #         self.data.iloc[i, j*3+2] = (1 - λ) * self.data.iloc[i-1, j*3+1] ** 2 \
            #                                        + λ * self.data.iloc[i-1, j*3+2]

            # Catering to a case where some series in a DataFrame may have NaNs due to different trading days
            sum = 0.
            for j in range(self.number_assets):
                df_copy = self.data.iloc[:, j*3+1:j*3+3].dropna()
                for i in range(2, len(df_copy)):
                    # Skipping values that would lead to a propagation of infinity
                    if np.isinf(df_copy.iloc[i - 1, 0]):
                        df_copy.iloc[i, 1] = df_copy.iloc[i-1, 1]
                    else:
                        df_copy.iloc[i, 1] = (1 - λ) * df_copy.iloc[i-1, 0] ** 2 + λ * df_copy.iloc[i-1, 1]
                sum -= (-np.log(df_copy.iloc[2:, 1]) - df_copy.iloc[2:, 0] ** 2 / df_copy.iloc[2:, 1]).sum()

            # sum = 0.
            # for j in range(self.number_assets):
            #     sum -= (-np.log(self.data.iloc[1:, j*3+2]) -
            #             self.data.iloc[1:, j*3+1] ** 2 / self.data.iloc[1:, j*3+2]).sum()
            return sum

        # print('Starting with objective function value of:', -objective_func(λ))
        res = minimize_scalar(objective_func, bounds=(0, 1), method='bounded')

        if res.success:
            print('Objective function: %.5f after %d iterations' % (-res.fun, res.nfev))
            self.lamda = res.x
        else:
            raise ValueError("Optimizing the objective function with the passed asset price changes didn't succeed")


class EWMAMinimumDifferenceParameterEstimator(ParameterEstimator):
    """
    A minimum difference estimator for the λ parameter of the EWMA method of forecasting volatility.
    Estimates the value of λ in the EWMA model such that it minimizes the value of Σ(νi - βi)^2, where νi is
    the variance forecast made at the end of day i − 1 and βi is the variance calculated from data between
    day i and day i + days_ahead.
    """

    def __init__(self, asset_prices_series=None, start=None, end=None, asset='EURUSD=X', days_ahead=25):
        super().__init__(asset_prices_series, start, end, asset)

        # Initial value of λ for EWMA
        λ = .9

        def objective_func(λ):
            """ This function searches for optimal values of the λ parameter of the EWMA
            model given the sample of asset price changes stored in the 'self.data' DataFrame.
            :param λ: the λ parameter in EWMA method of estimating volatility
            """

            sum = 0.
            for i in range(2, len(self.data) - days_ahead):
                for j in range(self.number_assets):
                    if np.isinf(self.data.iloc[i-1, j*3+1]):
                        self.data.iloc[i, j*3+2] = self.data.iloc[i-1, j*3+2]
                    else:
                        self.data.iloc[i, j*3+2] = (1 - λ) * self.data.iloc[i-1, j*3+1] ** 2 \
                                                   + λ * self.data.iloc[i-1, j*3+2]
                    # Ignoring the first 'days_ahead' observations of Σ(νi - βi)^2
                    # so that the results are not unduly influenced by the choice of starting values
                    if i >= days_ahead:
                        sum += (self.data.iloc[i, j*3+2]  # Taking an unbiased variance of βi
                                - (self.data.iloc[i:i + days_ahead, j*3+1] ** 2).sum() / (days_ahead - 1)) ** 2
            return sum

        # print('Starting with objective function value of:', -objective_func(λ))
        res = minimize_scalar(objective_func, bounds=(0, 1), method='bounded')

        if res.success:
            print('Objective function: %.5f after %d iterations' % (-res.fun, res.nfev))
            self.lamda = res.x
        else:
            raise ValueError("Optimizing the objective function with the passed asset price changes didn't succeed")


if __name__ == "__main__":
    import sys
    import os

    try:
        start = datetime.datetime(2005, 7, 27)
        end = datetime.datetime(2010, 7, 27)
        start = datetime.datetime(2019, 12, 15)
        end = datetime.datetime.today()
        data = web.get_data_yahoo('GBPUSD=X', start, end)
        asset_prices_series = data['Adj Close']
        ch10_ewma_md = EWMAMinimumDifferenceParameterEstimator(start=start, end=end, asset='EURUSD=X')
        print('Optimal value for λ using the minimum difference method for \'%s\': %.5f' % (
                'EURUSD=X', ch10_ewma_md.lamda))
        ch10_ewma_md = EWMAMinimumDifferenceParameterEstimator(start=start, end=end, asset='CADUSD=X')
        print('Optimal value for λ using the minimum difference method for \'%s\': %.5f' % (
                'CADUSD=X', ch10_ewma_md.lamda))
        ch10_ewma_md = EWMAMinimumDifferenceParameterEstimator(start=start, end=end, asset='GBPUSD=X')
        print('Optimal value for λ using the minimum difference method for \'%s\': %.5f' % (
                'GBPUSD=X', ch10_ewma_md.lamda))
        ch10_ewma_md = EWMAMinimumDifferenceParameterEstimator(start=start, end=end, asset='JPYUSD=X')
        print('Optimal value for λ using the minimum difference method for \'%s\': %.5f' % (
                'JPYUSD=X', ch10_ewma_md.lamda))
        # data = web.get_data_yahoo(['^GSPC', 'BTC-USD'], start, end)
        # asset_prices = data['Adj Close']
        ch10_ewma = GARCHParameterEstimator(asset_prices_series)
        print('Optimal value for λ: %.5f' % ch10_ewma.lamda)
        ch10_garch = GARCHParameterEstimator(asset_prices_series)
        print('Optimal values for GARCH parameters:\n\tω=%.12f, α=%.5f, β=%.5f'
              % (ch10_garch.omega, ch10_garch.alpha, ch10_garch.beta))

    except (IndexError, ValueError) as ex:
        print(
            '''Invalid number of arguments or incorrect values. Usage:
    {0:s} 
                '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())
