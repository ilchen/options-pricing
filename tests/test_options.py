import unittest
from math import exp
from datetime import date, datetime, time
from dateutil.relativedelta import relativedelta
from dateutil.relativedelta import FR

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import pandas_datareader.data as web
import pandas_market_calendars as mcal
import yfinance as yfin

import pricing.options
from pricing import curves, options
from volatility import volatility_trackers
from volatility import parameter_estimators

MSG_LONGER_DURATION_HIGHER_PRICE = 'Longer duration should lead to higher price'
global today, curve, holidays


def setUpModule():
    global today
    yfin.pdr_override()
    today = date.today()

    # Offsetting to the next business day if today is not a business day, not doing it would artificially raise
    # the price of options priced
    today = BDay(1).rollforward(today)

    # Constructing the riskless yield curve based on the current fed funds rate and treasury yields
    data = web.get_data_fred(
        ['AMERIBOR', 'AMBOR1W', 'AMBOR1M', 'AMBOR3M', 'AMBOR6M', 'AMBOR1Y', 'AMBOR2Y'],
        today - BDay(3), today)
    data.dropna(inplace=True)

    cur_date_curve = data.index[-1].date()

    # Convert to percentage points
    data /= 100.

    offsets = [relativedelta(), relativedelta(weeks=+1), relativedelta(months=+1), relativedelta(months=+3),
               relativedelta(months=+6),
               relativedelta(years=+1), relativedelta(years=+2)]

    # Some adjustments are required to bring AMERIBOR rates to an actual/actual day count convention
    data *= 365 / 360  # to x/actual

    # Define the riskless yield curve
    global curve, holidays
    curve = curves.YieldCurve(today, offsets, data[cur_date_curve:cur_date_curve + BDay()].to_numpy()[0, :],
                              compounding_freq=1)
    # Define the trading days calendar for more accurate pricing
    holidays = mcal.get_calendar('NYSE').holidays().holidays


class BaseOptionsPricingTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cur_price = np.NAN

    def assert_price_and_greeks_for_call(self, pricer):
        self.assertTrue(0. <= pricer.get_price() <= pricer.s0)
        # For
        if pricer.opt_type == options.OptionType.EUROPEAN:
            lower_bound = -pricer.strike * exp(-pricer.r * pricer.T * pricer.maturity_correction_coef)
            if pricer.q == 0:
                lower_bound -= pricer.get_npv_dividends()
            lower_bound += pricer.s0 * exp(-pricer.q * pricer.T * pricer.maturity_correction_coef)
            self.assertGreaterEqual(pricer.get_price(), lower_bound)
        self.assertTrue(0. <= pricer.get_delta() <= 1.)
        self.assertTrue(0. <= pricer.get_gamma() <= 1.)
        self.assertTrue(0. <= pricer.get_vega() <= self.cur_price * 2)
        self.assertTrue(0. <= pricer.get_rho() <= self.cur_price)
        self.assertTrue(-self.cur_price <= pricer.get_theta() <= 0)

    def assert_price_and_greeks_for_put(self, pricer):
        self.assertTrue(0. <= pricer.get_price() <= pricer.strike if pricer.opt_type == options.OptionType.AMERICAN
                        else pricer.strike * exp(-pricer.r * pricer.T * pricer.maturity_correction_coef))
        if pricer.opt_type == options.OptionType.EUROPEAN:
            lower_bound = pricer.strike * exp(-pricer.r * pricer.T * pricer.maturity_correction_coef)
            if pricer.q == 0:
                lower_bound += pricer.get_npv_dividends()
            lower_bound -= pricer.s0 * exp(-pricer.q * pricer.T * pricer.maturity_correction_coef)
            self.assertGreaterEqual(pricer.get_price(), lower_bound)
        elif pricer.opt_type == options.OptionType.AMERICAN:
            if pricer.q == 0 and pricer.divs is None:
                self.assertGreaterEqual(pricer.get_price(), pricer.strike - pricer.s0)
        self.assertTrue(-1. <= pricer.get_delta() <= 0.)
        self.assertTrue(0 <= pricer.get_gamma() <= 1.)
        self.assertTrue(0. <= pricer.get_vega() <= self.cur_price * 2)
        self.assertTrue(-self.cur_price <= pricer.get_rho() <= 0.)
        self.assertTrue(-self.cur_price <= pricer.get_theta() <= 0.)

    def assert_put_call_parity_for_index_option(self, pricer, put_pricer):
        self.assertEqual(pricer.s0, put_pricer.s0)
        self.assertEqual(pricer.strike, put_pricer.strike)
        self.assertEqual(pricer.r, put_pricer.r)
        self.assertEqual(pricer.T, put_pricer.T)
        self.assertEqual(pricer.q, put_pricer.q)
        self.assertEqual(pricer.maturity_correction_coef, put_pricer.maturity_correction_coef)
        self.assertAlmostEqual(pricer.get_price() - put_pricer.get_price(),
                               pricer.s0 * exp(-pricer.q * pricer.T)
                               - pricer.strike * exp(-pricer.r * pricer.T),
                               7 if isinstance(pricer, pricing.options.BlackScholesMertonPricer)
                               else 1)

    def assert_put_call_parity_for_american_equity_option(self, pricer, put_pricer):
        self.assertEqual(pricer.s0, put_pricer.s0)
        self.assertEqual(pricer.strike, put_pricer.strike)
        self.assertEqual(pricer.r, put_pricer.r)
        self.assertEqual(pricer.T, put_pricer.T)
        self.assertEqual(pricer.maturity_correction_coef, put_pricer.maturity_correction_coef)
        price_difference = pricer.get_price() - put_pricer.get_price()
        self.assertGreaterEqual(price_difference,
                                pricer.s0 - pricer.get_npv_dividends() - pricer.strike)
        self.assertLessEqual(price_difference,
                             pricer.s0 - pricer.strike * exp(-pricer.r * pricer.T))
        print(f'Put-Call parity: {pricer.s0 - pricer.get_npv_dividends() - pricer.strike:.2f} <= {price_difference:.2f}'
              f' <= {pricer.s0 - pricer.strike * exp(-pricer.r * pricer.T):.2f}')

    def assert_invariants(self, pricer):
        # Either a yield or dividends but not both
        self.assertTrue(not (pricer.q != 0 and pricer.divs is not None))

        # If pricing using a trading calendar, the conversion from time measured in trading days to
        # time measured in calendar days must work correctly
        self.assertTrue(pricer.maturity_correction_coef == 1. or pricer.holidays is not None)
        if pricer.maturity_correction_coef != 1.:
            dt = pricer.riskless_yield_curve.to_datetime(pricer.T * pricer.maturity_correction_coef)
            self.assertEqual(dt, datetime.combine(pricer.maturity_date, time()))


class NonDivPayingEquityOptionsPricingTestCase(BaseOptionsPricingTestCase):
    TICKER = 'TSLA'

    @classmethod
    def setUpClass(cls):
        # Define a volatility tracker
        start = BDay(1).rollback(today - relativedelta(years=+2))
        data = web.get_data_yahoo(NonDivPayingEquityOptionsPricingTestCase.TICKER, start, today)
        asset_prices = data['Adj Close']
        cls.cur_price = asset_prices[-1]

        # vol_estimator = parameter_estimators.GARCHParameterEstimator(asset_prices)
        # print('Optimal values for GARCH(1, 1) parameters:\n\tω=%.12f, α=%.5f, β=%.5f'
        #        % (vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta))
        # ω=0.000042359902, α=0.02906, β=0.94300, as of 17th October 2023
        # cls.vol_tracker = volatility_trackers.GARCHVolatilityTracker(vol_estimator.omega, vol_estimator.alpha,
        #                                                              vol_estimator.beta, asset_prices)

        # Unit tests must run fast, therefore creating a volatility tracker with pre-computed ω, α, and β values
        cls.vol_tracker = volatility_trackers.GARCHVolatilityTracker(.000042359902, 0.02906, 0.94300, asset_prices)

        cls.strike = 250.

    def test_next_Jan_american_call_BS(self):
        # The 3rd Friday of next January
        maturity_date = curve.date + relativedelta(years=1, month=1, day=1, weekday=FR(3))
        pricer = options.BlackScholesMertonPricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                                  ticker=self.TICKER, opt_type=options.OptionType.AMERICAN,
                                                  holidays=holidays)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

    def test_next_Jan_american_call_and_put_BT(self):
        # The 3rd Friday of next January
        maturity_date = curve.date + relativedelta(years=1, month=1, day=1, weekday=FR(3))
        pricer = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                            ticker=EquityOptionsPricingTestCase.TICKER,
                                            opt_type=options.OptionType.AMERICAN, holidays=holidays)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

        # Pricing without a business calendar
        pricer3 = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                             ticker=EquityOptionsPricingTestCase.TICKER,
                                             opt_type=options.OptionType.AMERICAN)
        print(pricer3)
        print('\u03c1: %.3f' % pricer3.get_rho())
        print('\u03b8: %.3f' % pricer3.get_theta())
        self.assert_price_and_greeks_for_call(pricer3)
        self.assert_invariants(pricer3)
        self.assertNotEqual(pricer.T, pricer3.T, 'Expecting different time to maturity')
        if pricer.T > pricer3.T:
            self.assertGreater(pricer.get_price(), pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)
        else:
            self.assertLessEqual(pricer.get_price(), pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)

        put_pricer = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                                is_call=False, ticker=self.TICKER,
                                                opt_type=options.OptionType.AMERICAN, holidays=holidays)
        print(put_pricer)
        print('\u03c1: %.3f' % put_pricer.get_rho())
        print('\u03b8: %.3f' % put_pricer.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer)
        self.assert_invariants(put_pricer)
        self.assert_put_call_parity_for_american_equity_option(pricer, put_pricer)

        # Pricing without a business calendar
        put_pricer3 = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                                 is_call=False, ticker=EquityOptionsPricingTestCase.TICKER,
                                                 opt_type=options.OptionType.AMERICAN)
        print(put_pricer3)
        print('\u03c1: %.3f' % put_pricer3.get_rho())
        print('\u03b8: %.3f' % put_pricer3.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer3)
        self.assert_invariants(put_pricer3)
        self.assertNotEqual(put_pricer.T, put_pricer3.T, 'Expecting different time to maturity')
        self.assert_put_call_parity_for_american_equity_option(pricer3, put_pricer3)
        if pricer.T > pricer3.T:
            self.assertGreater(put_pricer.get_price(), put_pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)
        else:
            self.assertLessEqual(put_pricer.get_price(), put_pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)


class EquityOptionsPricingTestCase(BaseOptionsPricingTestCase):
    TICKER = 'AAPL'

    @classmethod
    def setUpClass(cls):
        # Define a volatility tracker
        start = BDay(1).rollback(today - relativedelta(years=+2))
        data = web.get_data_yahoo(EquityOptionsPricingTestCase.TICKER, start, today)
        asset_prices = data['Adj Close']
        cls.cur_price = asset_prices[-1]

        # vol_estimator = parameter_estimators.GARCHParameterEstimator(asset_prices)
        # print('Optimal values for GARCH(1, 1) parameters:\n\tω=%.12f, α=%.5f, β=%.5f'
        #        % (vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta))
        # ω=0.000005857805, α=0.04725, β=0.93669, as of 15th October 2023
        # cls.vol_tracker = volatility_trackers.GARCHVolatilityTracker(vol_estimator.omega, vol_estimator.alpha,
        #                                                               vol_estimator.beta, asset_prices)

        # Unit tests must run fast, therefore creating a volatility tracker with pre-computed ω, α, and β values
        cls.vol_tracker = volatility_trackers.GARCHVolatilityTracker(.000005857805, 0.04725, 0.93669, asset_prices)
        cls.strike = 180.

        # Define upcoming dividends
        # yfinance returns the most recent ex-dividend date in the last row
        ticker = yfin.Ticker(EquityOptionsPricingTestCase.TICKER)
        last_divs = ticker.dividends[-1:]

        # An approximate rule for Apple's ex-dividend dates -- ex-dividend date is on the first Friday
        # of the last month of a season if that Friday is the 5th day of the month or later, otherwise
        # it falls on the second Friday of that month.
        idx = (pd.date_range(last_divs.index[0].date(), freq='WOM-1FRI', periods=30)[::3])
        idx = idx.map(lambda dt: dt if dt.day >= 5 else dt + BDay(5))
        cls.divs = pd.Series([last_divs[0]] * len(idx), index=idx,
                             name=EquityOptionsPricingTestCase.TICKER + ' Dividends')

    def test_next_Jan_american_call_BS(self):
        # The 3rd Friday of next January
        maturity_date = curve.date + relativedelta(years=1, month=1, day=1, weekday=FR(3))
        pricer = options.BlackScholesMertonPricer(maturity_date, self.vol_tracker, self.strike, curve,
                                                  self.cur_price, ticker=EquityOptionsPricingTestCase.TICKER,
                                                  dividends=self.divs, opt_type=options.OptionType.AMERICAN,
                                                  holidays=holidays)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

        # Pricing without a business calendar
        pricer3 = options.BlackScholesMertonPricer(maturity_date, self.vol_tracker, self.strike, curve,
                                                   self.cur_price, ticker=self.TICKER,
                                                   dividends=self.divs, opt_type=options.OptionType.AMERICAN)
        print(pricer3)
        print('\u03c1: %.3f' % pricer3.get_rho())
        print('\u03b8: %.3f' % pricer3.get_theta())
        self.assert_price_and_greeks_for_call(pricer3)
        self.assert_invariants(pricer3)
        self.assertNotEqual(pricer.T, pricer3.T, 'Expecting different time to maturity')

        # Pricing with an implied volatility of 27.42%
        impl_vol = .2742
        pricer2 = options.BlackScholesMertonPricer(maturity_date, impl_vol, self.strike, curve,
                                                   self.cur_price, ticker=self.TICKER,
                                                   dividends=self.divs, opt_type=options.OptionType.AMERICAN,
                                                   holidays=holidays)
        print(pricer2)
        print('\u03c1: %.3f' % pricer2.get_rho())
        print('\u03b8: %.3f' % pricer2.get_theta())
        self.assert_price_and_greeks_for_call(pricer2)
        self.assert_invariants(pricer2)
        self.assertEqual(pricer2.annual_volatility, impl_vol)

    def test_next_Jan_american_call_and_put_BT(self):
        # The 3rd Friday of next January
        maturity_date = curve.date + relativedelta(years=1, month=1, day=1, weekday=FR(3))
        pricer = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                            ticker=EquityOptionsPricingTestCase.TICKER, dividends=self.divs,
                                            opt_type=options.OptionType.AMERICAN, holidays=holidays)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

        # Pricing without a business calendar
        pricer3 = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                             ticker=self.TICKER, dividends=self.divs,
                                             opt_type=options.OptionType.AMERICAN)
        print(pricer3)
        print('\u03c1: %.3f' % pricer3.get_rho())
        print('\u03b8: %.3f' % pricer3.get_theta())
        self.assert_price_and_greeks_for_call(pricer3)
        self.assert_invariants(pricer3)
        self.assertNotEqual(pricer.T, pricer3.T, 'Expecting different time to maturity')
        if pricer.T > pricer3.T:
            self.assertGreater(pricer.get_price(), pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)
        else:
            self.assertLessEqual(pricer.get_price(), pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)

        put_pricer = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                                is_call=False, ticker=self.TICKER, dividends=self.divs,
                                                opt_type=options.OptionType.AMERICAN, holidays=holidays)
        print(put_pricer)
        print('\u03c1: %.3f' % put_pricer.get_rho())
        print('\u03b8: %.3f' % put_pricer.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer)
        self.assert_invariants(put_pricer)
        self.assert_put_call_parity_for_american_equity_option(pricer, put_pricer)

        # Pricing without a business calendar
        put_pricer3 = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                                 is_call=False, ticker=self.TICKER,
                                                 dividends=self.divs, opt_type=options.OptionType.AMERICAN)
        print(put_pricer3)
        print('\u03c1: %.3f' % put_pricer3.get_rho())
        print('\u03b8: %.3f' % put_pricer3.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer3)
        self.assert_invariants(put_pricer3)
        self.assertNotEqual(put_pricer.T, put_pricer3.T, 'Expecting different time to maturity')
        self.assert_put_call_parity_for_american_equity_option(pricer3, put_pricer3)
        if pricer.T > pricer3.T:
            self.assertGreater(put_pricer.get_price(), put_pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)
        else:
            self.assertLessEqual(put_pricer.get_price(), put_pricer3.get_price(), MSG_LONGER_DURATION_HIGHER_PRICE)


class EquityIndexOptionsPricingTestCase(BaseOptionsPricingTestCase):
    TICKER = '^GSPC'

    @classmethod
    def setUpClass(cls):
        # Define a volatility tracker
        start = BDay(1).rollback(today - relativedelta(years=+2))
        data = web.get_data_yahoo(EquityIndexOptionsPricingTestCase.TICKER, start, today)
        asset_prices = data['Adj Close']
        cls.cur_price = asset_prices[-1]

        # vol_estimator = parameter_estimators.GARCHParameterEstimator(asset_prices)
        # print('Optimal values for GARCH(1, 1) parameters:\n\tω=%.12f, α=%.5f, β=%.5f'
        #        % (vol_estimator.omega, vol_estimator.alpha, vol_estimator.beta))
        # cls.assertGreater(vol_estimator.omega, 0.)
        # cls.assertLess(vol_estimator.omega, 1e-4)
        # cls.assertGreater(vol_estimator.alpha, 0.)
        # cls.assertLess(vol_estimator.alpha, .2)
        # cls.assertGreater(vol_estimator.beta, 0.)
        # cls.assertLess(vol_estimator.beta, 1.)
        # ω=0.000001906502, α=0.06303, β=0.92557, as of 16th October 2023

        # cls.vol_tracker = volatility_trackers.GARCHVolatilityTracker(vol_estimator.omega, vol_estimator.alpha,
        #                                                              vol_estimator.beta, asset_prices)

        # Unit tests must run fast, therefore creating a volatility tracker with pre-computed ω, α, and β values
        cls.vol_tracker = volatility_trackers.GARCHVolatilityTracker(.000001906502, 0.06303, 0.92557, asset_prices)
        cls.strike = 3900.
        cls.impl_vol = 0.22

        # Expected S&P 500 dividend yield
        cls.q = .017049  # Updated for October 2023

    def test_next_Mar_european_call_and_put_BS(self):
        # The 3rd Friday of next March
        maturity_date = curve.date + relativedelta(years=1, month=3, day=1, weekday=FR(3))
        pricer = options.BlackScholesMertonPricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                                  ticker=self.TICKER, q=self.q, holidays=holidays)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

        put_pricer = options.BlackScholesMertonPricer(maturity_date, self.vol_tracker, self.strike, curve,
                                                      self.cur_price, is_call=False, ticker=self.TICKER, q=self.q,
                                                      holidays=holidays)
        print(put_pricer)
        print('\u03c1: %.3f' % put_pricer.get_rho())
        print('\u03b8: %.3f' % put_pricer.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer)
        self.assert_invariants(put_pricer)

        # Test for a put-call parity
        self.assert_put_call_parity_for_index_option(pricer, put_pricer)

        # Pricing with an implied volatility
        pricer = options.BlackScholesMertonPricer(maturity_date, self.impl_vol, self.strike, curve, self.cur_price,
                                                  ticker=self.TICKER, q=self.q, holidays=holidays)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

        put_pricer = options.BlackScholesMertonPricer(maturity_date, self.impl_vol, self.strike, curve, self.cur_price,
                                                      is_call=False, ticker=self.TICKER, q=self.q, holidays=holidays)
        print(put_pricer)
        print('\u03c1: %.3f' % put_pricer.get_rho())
        print('\u03b8: %.3f' % put_pricer.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer)
        self.assert_invariants(put_pricer)

        # Test for a put-call parity
        self.assert_put_call_parity_for_index_option(pricer, put_pricer)

        # Pricing with an implied volatility and without a holiday calendar
        pricer = options.BlackScholesMertonPricer(maturity_date, self.impl_vol, self.strike, curve, self.cur_price,
                                                  ticker=self.TICKER, q=self.q)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

        put_pricer = options.BlackScholesMertonPricer(maturity_date, self.impl_vol, self.strike, curve,
                                                      self.cur_price, is_call=False, ticker=self.TICKER, q=self.q)
        print(put_pricer)
        print('\u03c1: %.3f' % put_pricer.get_rho())
        print('\u03b8: %.3f' % put_pricer.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer)
        self.assert_invariants(put_pricer)

        # Test for a put-call parity
        self.assert_put_call_parity_for_index_option(pricer, put_pricer)

    def test_next_Mar_european_call_and_put_BT(self):
        # The 3rd Friday of next March
        maturity_date = curve.date + relativedelta(years=1, month=3, day=1, weekday=FR(3))
        pricer = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                            ticker=self.TICKER, q=self.q, holidays=holidays)
        print(pricer)
        print('\u03c1: %.3f' % pricer.get_rho())
        print('\u03b8: %.3f' % pricer.get_theta())
        self.assert_price_and_greeks_for_call(pricer)
        self.assert_invariants(pricer)

        put_pricer = options.BinomialTreePricer(maturity_date, self.vol_tracker, self.strike, curve, self.cur_price,
                                                is_call=False, ticker=self.TICKER, q=self.q, holidays=holidays)
        print(put_pricer)
        print('\u03c1: %.3f' % put_pricer.get_rho())
        print('\u03b8: %.3f' % put_pricer.get_theta())
        self.assert_price_and_greeks_for_put(put_pricer)
        self.assert_invariants(put_pricer)

        # Test for a put-call parity, difficult to make precise due to never knowing q exactly
        self.assert_put_call_parity_for_index_option(pricer, put_pricer)


if __name__ == '__main__':
    unittest.main()
