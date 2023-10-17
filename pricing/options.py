from enum import Enum, unique
from math import sqrt, log, exp
from datetime import date

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
                 opt_type=OptionType.EUROPEAN, ticker=UNSPECIFIED_TICKER, q=0., dividends=None, holidays=None):
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
        :param holidays: array_like of datetime64[D], optional, allows for more accurate pricing by taking additional
                  non-trading days into account. If not specified, the year duration of the option is translated using
                  calendar days, if specified, it is calculated based on the number of business days to maturity
                  divided by 252
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
        self.holidays = holidays
        self.T = curve.to_years(self.maturity_date) if holidays is None\
            else curve.to_years_busdays_based(self.maturity_date, holidays)
        # When measuring the lifetime of an option in trading days rather than calendar days,
        # the maturity correction coefficient will be > 1 most of the time. It is needed to
        # translate from timescales implied by the option's lifetime to real ones expressed in calendar days
        # to get correct discount factors and ex-dividend dates. It's needed when pricing using Binomial trees.
        self.maturity_correction_coef = 1. if holidays is None else curve.to_years(self.maturity_date) / self.T
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
            self.divs = dividends.truncate(after=self.maturity_date)\
                .truncate(before=self.riskless_yield_curve.date+BDay())
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
                 opt_type=OptionType.EUROPEAN, ticker=OptionsPricer.UNSPECIFIED_TICKER, q=0., dividends=None,
                 holidays=None):
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
        :param holidays: array_like of datetime64[D], optional, allows for more accurate pricing by taking additional
                  non-trading days into account. If not specified, the year duration of the option is translated using
                  calendar days, if specified, it is calculated based on the number of business days to maturity
                  divided by 252
        """

        super().__init__(maturity, volatility_tracker, strike, curve, cur_price, is_call,
                         opt_type, ticker, q, dividends, holidays)

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
                        dividends=self.divs.truncate(after=self.divs.index[-2]) if len(self.divs.index) > 1 else None,
                        holidays=self.holidays)
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
            summand2 *= -norm.cdf(-self.d1)
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
                 holidays=None, num_steps_binomial_tree=26):
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
        :param holidays: array_like of datetime64[D], optional, allows for more accurate pricing by taking additional
                  non-trading days into account. If not specified, the year duration of the option is translated using
                  calendar days, if specified, it is calculated based on the number of business days to maturity
                  divided by 252
        :param num_steps_binomial_tree: number of steps in the binomial tree that will be constructed, must be >=2,
                                        the higher the number the more accurate the price
        """

        super().__init__(maturity, volatility_tracker, strike, curve, cur_price, is_call,
                         opt_type, ticker, q, dividends, holidays)

        assert num_steps_binomial_tree >= 2

        self.steps = num_steps_binomial_tree
        self.delta_t = self.T / self.steps

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
        # maturity_correction_coef differs from 1. if we are pricing the option by determining its lifetime
        # expressed in years based on the number of trading days till option maturity divided by 252.
        for i in range(self.steps-1, -1, -1):
            npv_divs = self.get_npv_dividends(self.delta_t * i * self.maturity_correction_coef)\
                if self.divs is not None else 0.
            dcf = self.riskless_yield_curve.get_forward_discount_factor_for_maturity_date(
                    self.riskless_yield_curve.to_datetime(self.delta_t * i * self.maturity_correction_coef),
                    self.riskless_yield_curve.to_datetime(self.delta_t * (i + 1) * self.maturity_correction_coef))
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
        delta_sigma = 1e-4  # 1 bp
        return (BinomialTreePricer(self.maturity_date, self.annual_volatility + delta_sigma, self.strike,
                                   self.riskless_yield_curve, self.s0, self.is_call, self.opt_type, self.ticker,
                                   self.q, self.divs, self.holidays, self.steps).get_price() - self.get_price())\
            / delta_sigma

    def get_theta(self):
        return (self.tree[2][1][1] - self.tree[0][0][1]) / (2 * self.delta_t)

    def get_rho(self):
        num_bp = 1
        return (BinomialTreePricer(self.maturity_date, self.annual_volatility, self.strike,
                                   self.riskless_yield_curve.parallel_shift(num_bp), self.s0, self.is_call,
                                   self.opt_type, self.ticker, self.q, self.divs, self.holidays, self.steps).get_price()
                - self.get_price()) / (num_bp * 1e-4)
