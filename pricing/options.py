from enum import Enum, unique
from math import sqrt, log, exp
from datetime import date, timedelta

import pandas as pd
from pandas.tseries.offsets import BDay
from scipy.stats import norm

from volatility import volatility_trackers
from pricing import curves


@unique
class OptionType(Enum):
    EUROPEAN = 0
    AMERICAN = 1

    def __str__(self):
        return self.name


class OptionsPricer:
    """
    Pricer for options base class
    """

    UNSPECIFIED_TICKER = 'Unspecified'

    def __init__(self, maturity, volatility_tracker, strike, curve, cur_price, is_call=True,
                 opt_type=OptionType.EUROPEAN, ticker=UNSPECIFIED_TICKER, q=0., dividends=None):
        """
        Constructs a pricer for based on specified maturities and rates.

        :param maturity: a datetime.date specifying the day the options matures on
        :param volatility_tracker: a VolatilityTracker object or a float value representing annual volatility
        :param strike: a numpy.float64 object representing the strike price of the option
        :param curve: a yield curve object representing a riskless discount curve to use
        :param cur_price: the current price of the asset
        :param is_call: indicates if we are pricing a call or a put option
        :param opt_type: the type of option -- European or American
        :param ticker: the ticker symbol of the underlying asset (optional)
        :param q: the yield the underlying asset provides (if any, expressed as annual yield
                                                           with continuous compounding)
        :param dividends: if the asset pays dividends during the specified maturity period, specifies a pandas.Series
                  object indexed by DatetimeIndex specifying expected dividend payments. The dates must be
                  based on the ex-dividend date for the asset. NB: if this parameter is not None,
                  the 'q' parameter must be 0
        """
        self.maturity_date = maturity
        self.vol_tracker = volatility_tracker if isinstance(volatility_tracker, volatility_trackers.VolatilityTracker)\
                                              else None
        self.strike = strike
        self.riskless_yield_curve = curve
        self.s0 = cur_price
        self.is_call = is_call
        self.opt_type = opt_type
        self.ticker = ticker
        self.T = curve.to_years(self.maturity_date)
        if self.T == 0.:
            self.T = 8 / (24 * (366 if curves.YieldCurve.is_leap_year(self.maturity_date.year) else 365))
        self.r = self.riskless_yield_curve.to_continuous_compounding(
            self.riskless_yield_curve.get_yield_for_maturity_date(self.maturity_date))
        self.annual_volatility = self.vol_tracker.get_annual_term_volatility_forecast(self.T)\
            if self.vol_tracker is not None else volatility_tracker

        self.q = q

        if dividends is None:
            self.divs = None
        else:
            assert isinstance(dividends.index, (pd.core.indexes.datetimes.DatetimeIndex, pd.core.indexes.base.Index))\
                and dividends.index.is_monotonic_increasing and q == 0.
            # Get rid of dividends not relevant for valuing this option
            self.divs = dividends.truncate(after=self.maturity_date).truncate(before=self.riskless_yield_curve.date+BDay())
            if self.divs.empty:
                self.divs = None

    def __str__(self):
        return '%s %s %s option with strike %s and maturity %s, price: %.2f, \u03c3: %.4f, '\
               '\u0394: %.3f, \u0393: %.3f, \u03Bd: %.3f'\
              % (self.ticker, self.opt_type, 'call' if self.is_call else 'put',
                 self.strike, self.maturity_date.strftime('%Y-%m-%d'),
                 self.get_price(), self.annual_volatility, self.get_delta(), self.get_gamma(), self.get_vega())

    def get_npv_dividends(self, t=0.):
        """
        Returns the net present value of all future cash dividends from the viewpoint of time t

        :param t: time in years from self.riskless_yield_curve.date
        """
        assert self.q == 0.

        npv_divs = 0.  # net present value of all dividends
        if self.divs is not None:
            future_datetime = self.riskless_yield_curve.to_datetime(t)
            for d in self.divs.index:
                index_as_date = d
                if type(d) is not date:
                    index_as_date = d.date()
                if index_as_date > self.maturity_date or index_as_date <= future_datetime.date():
                    continue
                dcf = self.riskless_yield_curve.get_discount_factor_for_maturity_date(d) if t == 0.\
                    else self.riskless_yield_curve.get_forward_discount_factor_for_maturity_date(future_datetime, d)
                npv_divs += self.divs[d] * dcf
        return npv_divs

    def get_price(self):
        """
        Calculates the present value of the option
        """
        raise NotImplementedError

    def get_delta(self):
        raise NotImplementedError

    def get_theta(self):
        raise NotImplementedError

    def get_gamma(self):
        raise NotImplementedError

    def get_vega(self):
        raise NotImplementedError

    def get_rho(self):
        raise NotImplementedError


class BlackScholesMertonPricer(OptionsPricer):
    """
    The Black-Scholes-Merton model pricer. It can be used to price European options, including on stock that pays
    dividends. It can also be used to price American call options (even those paying dividends) doing it is based
    on the fact that it is never optimal to exercise an American call option prematurely except on days immediately
    preceding ex-dividend days for the stock.
    """

    def __init__(self, maturity, volatility_tracker, strike, curve, cur_price, is_call=True,
                 opt_type=OptionType.EUROPEAN, ticker=OptionsPricer.UNSPECIFIED_TICKER, q=0., dividends=None):
        """
        Constructs a pricer based on the Black-Scholes-Merton options pricing model. It can price any European option
        and any American call option including those whose underlying stock pays dividend.

        :param maturity: a datetime.date specifying the day the options matures on
        :param volatility_tracker: a VolatilityTracker object
        :param strike: a numpy.float64 object representing the strike price of the option
        :param curve: a yield curve object representing a riskless discount curve to use
        :param cur_price: the current price of the asset
        :param is_call: indicates if we are pricing a call or a put option
        :param opt_type: the type of option -- European or American
        :param ticker: the ticker symbol of the underlying asset (optional)
        :param q: the yield the underlying asset provides (if any, expressed as annual yield
                                                           with continuous compounding)
        :param dividends: if the asset pays dividends during the specified maturity period, specifies a pandas.Series
                          object indexed by DatetimeIndex specifying expected dividend payments. The dates must be
                          based on the ex-dividend date for the asset. NB: if this parameter is not None,
                          the 'q' parameter must be 0

        """

        super().__init__(maturity, volatility_tracker, strike, curve, cur_price, is_call,
                         opt_type, ticker, q, dividends)

        if self.opt_type == OptionType.AMERICAN and not self.is_call:
            raise NotImplementedError('Cannot price an American Put option using the Black-Scholes-Merton model')

        # Initializing the rest of the fields
        self._init_fields()

    def _init_fields(self):
        """
        Constructor helper method to initialize additional fields
        """
        adjusted_s0 = self.s0 - self.get_npv_dividends() if self.q == 0. else self.s0
        vol_to_maturity = self.annual_volatility * sqrt(self.T)
        d1 = (log(adjusted_s0 / self.strike) + (self.r - self.q + self.annual_volatility ** 2 / 2.) * self.T)\
             / vol_to_maturity
        d2 = d1 - vol_to_maturity

        self.early_exercise_pricer = None
        if self.is_call:
            call_price = adjusted_s0 * norm.cdf(d1) * exp(-self.q * self.T)\
                         - self.strike * exp(-self.r * self.T) * norm.cdf(d2)
            if self.opt_type == OptionType.AMERICAN and self.divs is not None:
                # We need to check if it's optimal to exercise immediately before the ex-dividend date.
                # It's best to do that recursively
                pricer = BlackScholesMertonPricer((self.divs.index[-1] - BDay(1)).date(),
                        self.vol_tracker if self.vol_tracker is not None else self.annual_volatility,
                        self.strike, self.riskless_yield_curve, self.s0, opt_type=OptionType.AMERICAN,
                        dividends=self.divs.truncate(after=self.divs.index[-2]) if len(self.divs.index) > 1 else None)
                if pricer.get_price() > call_price:
                    self.early_exercise_pricer = pricer
            self.price = call_price
        else:
            self.price = self.strike * exp(-self.r * self.T) * norm.cdf(-d2)\
                         - adjusted_s0 * norm.cdf(-d1) * exp(-self.q * self.T)
        self.adjusted_s0 = adjusted_s0
        self.d1 = d1
        self.d2 = d2

    def get_price(self):
        # In case we are dealing with an American call option where early exercise is advantageous
        return self.price if self.early_exercise_pricer is None else self.early_exercise_pricer.get_price()
    
    def get_delta(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_delta()
        return exp(-self.q * self.T) * (norm.cdf(self.d1) if self.is_call else norm.cdf(self.d1) - 1)

    def get_theta(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_theta()
        summand = self.r * self.strike * exp(-self.r * self.T)
        summand2 = exp(-self.q * self.T) * self.q * self.adjusted_s0
        if self.is_call:
            summand *= -norm.cdf(self.d2)
            summand2 *= norm.cdf(self.d1)
        else:
            summand *= norm.cdf(-self.d2)
            summand2 *= -norm.cfg(-self.d1)
        return -self.adjusted_s0 * norm.pdf(self.d1) * self.annual_volatility * exp(-self.q * self.T)\
               / (2 * sqrt(self.T)) + summand + summand2

    def get_gamma(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_gamma()
        return norm.pdf(self.d1) * exp(-self.q * self.T) / (self.adjusted_s0 * self.annual_volatility * sqrt(self.T))

    def get_vega(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_vega()
        return self.adjusted_s0 * sqrt(self.T) * norm.pdf(self.d1) * exp(-self.q * self.T)

    def get_rho(self):
        if self.early_exercise_pricer:
            return self.early_exercise_pricer.get_rho()
        multiplicand = self.strike * self.T * exp(-self.r * self.T)
        return multiplicand * (norm.cdf(self.d2) if self.is_call else -norm.cdf(-self.d2))


class BinomialTreePricer(OptionsPricer):
    """
    The Binomial Tree model pricer. It can be used to price any European and American options, including on stock
    that pays dividends or where a dividend yield estimate is known.
    """

    def __init__(self, maturity, volatility_tracker, strike, curve, cur_price, is_call=True,
                 opt_type=OptionType.EUROPEAN, ticker=OptionsPricer.UNSPECIFIED_TICKER, q=0., dividends=None,
                 num_steps_binomial_tree=26):
        """
        Constructs a pricer based on the Binomial Tree options pricing model. It can price any European option
        and any American option including those whose underlying stock pays dividend.

        :param maturity: a datetime.date specifying the day the options matures on
        :param volatility_tracker: a VolatilityTracker object
        :param strike: a numpy.float64 object representing the strike price of the option
        :param curve: a yield curve object representing a riskless discount curve to use
        :param cur_price: the current price of the asset
        :param is_call: indicates if we are pricing a call or a put option
        :param opt_type: the type of option -- European or American
        :param ticker: the ticker symbol of the underlying asset (optional)
        :param q: the yield the underlying asset provides (if any, expressed as annual yield
                                                           with continuous compounding)
        :param dividends: if the asset pays dividends during the specified maturity period, specifies a pandas.Series
                          object indexed by DatetimeIndex specifying expected dividend payments
        :param num_steps_binomial_tree: number of steps in the binomial tree that will be constructed, must be >=2,
                                        the higher the number the more accurate the price
        """

        super().__init__(maturity, volatility_tracker, strike, curve, cur_price, is_call,
                         opt_type, ticker, q, dividends)

        assert num_steps_binomial_tree >= 2

        year_diff = self.riskless_yield_curve.to_years(self.maturity_date)
        self.steps = num_steps_binomial_tree
        self.delta_t = year_diff / self.steps

        self.u = exp(self.annual_volatility * sqrt(self.delta_t))
        self.d = exp(-self.annual_volatility * sqrt(self.delta_t))
        self.a = exp((self.r - self.q) * self.delta_t)
        self.p = (self.a - self.d) / (self.u - self.d)

        self.tree = []

        # Constructing the binomial tree

        # First initializing the prices of the underlying assets, striping out the NPV of all dividends (if any)
        adjusted_s0 = self.s0 - self.get_npv_dividends() if self.q == 0. else self.s0
        self.tree.append([[adjusted_s0, 0.]])
        for i in range(1, self.steps+1):
            # Constructing tree nodes for level i
            # print('Level %d corresponding to %s' % (i, self.riskless_yield_curve.to_datetime(i*self.delta_t)))
            level_nodes = [[self.tree[i - 1][0][0] * self.u, 0.]]
            for j in range(len(self.tree[i-1])):
                level_nodes.append([self.tree[i-1][j][0] * self.d, 0.])
            self.tree.append(level_nodes)

        # Adding the NPV of dividends and calculating the option's price
        for j in range(self.steps+1):
            self.tree[self.steps][j][1] = max(0., self.tree[self.steps][j][0] - self.strike) if self.is_call \
                                            else max(0., self.strike - self.tree[self.steps][j][0])
        for i in range(self.steps-1, -1, -1):
            npv_divs = self.get_npv_dividends(self.delta_t * i) if self.divs is not None else 0.
            dcf = self.riskless_yield_curve.get_forward_discount_factor_for_maturity_date(
                    self.riskless_yield_curve.to_datetime(self.delta_t * i),
                    self.riskless_yield_curve.to_datetime(self.delta_t * (i + 1)))
            for j in range(i+1):
                self.tree[i][j][0] += npv_divs
                opt_price = (self.p * self.tree[i+1][j][1] + (1 - self.p) * self.tree[i+1][j+1][1]) * dcf
                if self.opt_type == OptionType.EUROPEAN:
                    self.tree[i][j][1] = opt_price
                else:
                    self.tree[i][j][1] = max(opt_price, self.tree[i][j][0] - self.strike) if self.is_call\
                                            else max(opt_price, self.strike - self.tree[i][j][0])

    def get_price(self):
        return self.tree[0][0][1]

    def get_delta(self):
        return (self.tree[1][0][1] - self.tree[1][1][1]) / (self.tree[1][0][0] - self.tree[1][1][0])

    def get_gamma(self):
        h = (self.tree[2][0][0] - self.tree[2][2][0]) * .5
        return ((self.tree[2][0][1] - self.tree[2][1][1]) / (self.tree[2][0][0] - self.tree[0][0][0])
                - (self.tree[2][1][1] - self.tree[2][2][1]) / (self.tree[0][0][0] - self.tree[2][2][0])) / h

    def get_vega(self):
        delta_sigma = 1e-4 # 1 bp
        return (BinomialTreePricer(self.maturity_date, self.annual_volatility + delta_sigma, self.strike,
                                   self.riskless_yield_curve, self.s0, self.is_call, self.opt_type, self.ticker,
                                   self.q, self.divs, self.steps).get_price() - self.get_price()) / delta_sigma

    def get_theta(self):
        return (self.tree[2][1][1] - self.tree[0][0][1]) / (2 * self.delta_t)

    def get_rho(self):
        num_bp = 1
        return (BinomialTreePricer(self.maturity_date, self.annual_volatility, self.strike,
                                   self.riskless_yield_curve.parallel_shift(num_bp), self.s0, self.is_call,
                                   self.opt_type, self.ticker, self.q, self.divs, self.steps).get_price()
                - self.get_price()) / (num_bp * 1e-4)


if __name__ == "__main__":
    import sys
    import locale
    from datetime import date, datetime

    from dateutil.relativedelta import relativedelta
    import pandas_datareader.data as web
    import numpy as np

    from volatility import parameter_estimators
    from pricing import curves

    TICKER = 'AAPL'

    try:
        locale.setlocale(locale.LC_ALL, '')
        start = date(2018, 1, 1)
        end = date.today()
        data = web.get_data_yahoo(TICKER, start, end)
        asset_prices = data['Adj Close']
        # cur_date = max(date.today(), asset_prices.index[-1].date())
        cur_date = asset_prices.index[-1].date()
        cur_price = asset_prices[-1]
        print('S0 of %s on %s:\t%.5f' % (TICKER, date.strftime(cur_date, '%Y-%m-%d'), cur_price))

        maturity_date = date(2023, month=1, day=20)

        # Constructing the riskless yield curve based on the current fed funds rate and treasury yields
        data = web.get_data_fred(
            ['DFF', 'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30'],
            end - BDay(3), end)
        data.dropna(inplace=True)

        cur_date_curve = data.index[-1].date()

        # Convert to percentage points
        data /= 100.

        # Some adjustments are required:
        # 1. https://www.federalreserve.gov/releases/h15/default.htm -> day count convention for Fed Funds Rate needs
        # to be changed to actual/actual
        # 2. Conversion to APY: https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions
        data.DFF *= (366 if curves.YieldCurve.is_leap_year(cur_date_curve.year) else 365) / 360 # to x/actual
        data.DFF = 2 * (np.sqrt(data.DFF + 1) - 1)

        offsets = [relativedelta(), relativedelta(months=+1), relativedelta(months=+3), relativedelta(months=+6),
                   relativedelta(years=+1), relativedelta(years=+2), relativedelta(years=+3), relativedelta(years=+5),
                   relativedelta(years=+7), relativedelta(years=+10), relativedelta(years=+20),
                   relativedelta(years=+30)]

        # Define yield curves
        curve = curves.YieldCurve(cur_date, offsets, data[cur_date_curve:cur_date_curve + BDay()].to_numpy()[0, :],
                                  compounding_freq=2)

        cp = curve.get_curve_points(12)
        cps = curve.parallel_shift(1).get_curve_points(12)

        # Obtaining a volatility estimate for maturity
        # vol_estimator = parameter_estimators.GARCHParameterEstimator(asset_prices)
        # print('Optimal values for GARCH(1, 1) parameters:\n\t??=%.12f, ??=%.5f, ??=%.5f'
        #       % (vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta))

        # vol_estimator2 = parameter_estimators.GARCHVarianceTargetingParameterEstimator(asset_prices)
        # print('Optimal values for GARCH(1, 1) parameters:\n\t??=%.12f, ??=%.5f, ??=%.5f'
        #       % (vol_estimator2.omega, vol_estimator2.alpha, vol_estimator2.beta))
        #
        # print('Optimal values for GARCH(1, 1) parameters:\n\t??=%.12f, ??=%.5f, ??=%.5f'
        #       % (vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta))
        #
        # vol_tracker = volatility_trackers.GARCHVolatilityTracker(vol_estimator.omega, vol_estimator.alpha,
        #                                                          vol_estimator.beta, asset_prices)
        vol_tracker = volatility_trackers.GARCHVolatilityTracker(0.000020633369, 0.12462, 0.83220, asset_prices)

        # vol_estimator = parameter_estimators.EWMAParameterEstimator(asset_prices)
        # vol_tracker = volatility_trackers.EWMAVolatilityTracker(vol_estimator.lamda, asset_prices)
        # Volatility of AAPL for term 0.2137: 0.32047

        vol = vol_tracker.get_annual_term_volatility_forecast(curve.to_years(maturity_date))
        # vol = 0.3134162504522202 (1 Aug 2022)
        # vol = 0.29448 # (2 Aug 2022)
        print('Volatility of %s for term %.4f: %.5f' % (TICKER, curve.to_years(maturity_date), vol))

        strike = 180.

        # Yahoo-dividends returns the most recent ex-dividend date in the first row
        last_divs = web.DataReader(TICKER, 'yahoo-dividends', cur_date.year).value

        # An approximate rule for Apple's ex-dividend dates -- ex-dividend date is on the first Friday
        # of the last month of a season.
        idx = (pd.date_range(last_divs.index[0].date(), freq='WOM-1FRI', periods=30)[::3])
        divs = pd.Series([last_divs[0]] * len(idx), index=idx, name=TICKER + ' Dividends')

        pricer = BlackScholesMertonPricer(maturity_date, vol_tracker, strike, curve, cur_price,
                                          ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer)

        pricer_put = BlackScholesMertonPricer(maturity_date, vol_tracker, strike, curve, cur_price, is_call=False,
                                              ticker=TICKER, dividends=divs)
        print(pricer_put)

        pricer = BinomialTreePricer(maturity_date, vol_tracker, strike, curve, cur_price,
                                    ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer)

        pricer_put = BinomialTreePricer(maturity_date, vol_tracker, strike, curve, cur_price, is_call=False,
                                        ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer_put)
        pricer_put = BinomialTreePricer(maturity_date, vol_tracker, strike, curve, cur_price, is_call=False,
                                        ticker=TICKER, dividends=divs, opt_type=OptionType.EUROPEAN)
        print(pricer_put)

        maturity_date = date(2023, month=3, day=17)
        vol = vol_tracker.get_annual_term_volatility_forecast(curve.to_years(maturity_date))
        print('\nVolatility of %s for term %.4f: %.5f' % (TICKER, curve.to_years(maturity_date), vol))

        pricer = BlackScholesMertonPricer(maturity_date, vol_tracker, strike, curve, cur_price,
                                          ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        pricer = BinomialTreePricer(maturity_date, vol_tracker, strike, curve, cur_price,
                                    ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        pricer_put = BinomialTreePricer(maturity_date, vol_tracker, strike, curve, cur_price, is_call=False,
                                        ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer_put)
        print('\u03c1: %.3f' % pricer.get_rho())

        impl_vol = .3084
        print('\nPricing with an implied volatility of %.2f' % impl_vol)
        pricer = BlackScholesMertonPricer(maturity_date, impl_vol, strike, curve, cur_price,
                                          ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer)
        pricer = BinomialTreePricer(maturity_date, impl_vol, strike, curve, cur_price,
                                    ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(pricer)

        put_pricer = BinomialTreePricer(maturity_date, impl_vol, strike, curve, cur_price, is_call=False,
                                        ticker=TICKER, dividends=divs, opt_type=OptionType.AMERICAN)
        print(put_pricer)

        TICKER = '^GSPC'

        # I'll use price changes since 1st Jan 2018 to estimate GARCH(1, 1) ??, ??, and ?? parameters
        data = web.get_data_yahoo(TICKER, start, end)
        asset_prices = data['Adj Close']

        # vol_estimator = parameter_estimators.GARCHParameterEstimator(asset_prices)
        # print('Optimal values for GARCH(1, 1) parameters:\n\t??=%.12f, ??=%.5f, ??=%.5f'
        #       % (vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta))
        # Optimal values for GARCH(1, 1) parameters:
        # 	??=0.000005476805, ??=0.20899, ??=0.76364
        #
        # vol_tracker = volatility_trackers.GARCHVolatilityTracker(vol_estimator.omega, vol_estimator.alpha,
        #                                                          vol_estimator.beta, asset_prices)
        vol_tracker = volatility_trackers.GARCHVolatilityTracker(.000005476805, .20899, .76364, asset_prices)

        # Let's get volatility forecast for June 30th 2023 options
        maturity_date = date(2023, month=6, day=30)
        vol = vol_tracker.get_annual_term_volatility_forecast(curve.to_years(maturity_date))
        print('Volatility of %s for term %.4f years: %.5f' % (TICKER, curve.to_years(maturity_date), vol))

        strike = 3900.

        # Expected S&P 500 dividend yield
        q = .0163

        cur_price = asset_prices[-1]
        pricer = BlackScholesMertonPricer(maturity_date, vol_tracker, strike, curve, cur_price,
                                          ticker=TICKER, q=q)
        print(pricer)

    # except (IndexError, ValueError) as ex:
    #     print(
    #         '''Invalid number of arguments or incorrect values. Usage:
    # {0:s}
    #             '''.format(sys.argv[0].split(os.sep)[-1]))
    except:
        print("Unexpected error: ", sys.exc_info())

