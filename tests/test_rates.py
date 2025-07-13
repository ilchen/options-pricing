import unittest
from math import exp, sqrt
from datetime import date
import numpy as np

from dateutil.relativedelta import relativedelta

from pricing import futures_rates, par_yield_converter

class CashFlowDescriptorTestCase(unittest.TestCase):
    def test_duration(self):
        notional1, notional2, notional3 = (2000, 6000, 5000)
        t1, t2, t3 = (1, 10, 5.95)
        coupon_rate = 0.
        coupon_frequency = 0
        risk_free_rate = .1   # continuous compounding
        cashflow_descr1 = futures_rates.CashflowDescriptor(coupon_rate, coupon_frequency, notional1, t1)
        cashflow_descr2 = futures_rates.CashflowDescriptor(coupon_rate, coupon_frequency, notional2, t2)
        cashflow_descr3 = futures_rates.CashflowDescriptor(coupon_rate, coupon_frequency, notional3, t3)
        portfolio1 = futures_rates.BondPortfolio([cashflow_descr1, cashflow_descr2])
        self.assertTrue(np.allclose(cashflow_descr3.get_duration(risk_free_rate), t3))
        self.assertTrue(np.allclose(cashflow_descr3.get_duration(risk_free_rate),
                                    portfolio1.get_duration(risk_free_rate), 1e-3))

    def test_pv_calculation(self):
        # First let's figure out the present value of $1 received at the same frequency as coupons. This is
        # the present value of a spread of 2% per annum that pays coupons half-yearly.
        # This is equivalent to pricing an annuity that pays $1 every 6 months.

        # This is the present value of a riskless bond with a notional of $100 that pays 2% coupon every 6 months minus
        # the present value of the bond notional.
        coupon_rate = .02
        coupon_frequency = 2  # semiannual payments
        libor_swap_rate = .06  # continuous compounding
        libor_swap_rate_hy = 2 * (sqrt(exp(libor_swap_rate)) - 1)
        t = 5  # it's a five-year bond
        notional = 100

        cashflow_descr = futures_rates.CashflowDescriptor(coupon_rate, coupon_frequency, notional, t)
        one_dollar_annuity = cashflow_descr.pv_all_cashflows(libor_swap_rate) - notional * exp(-libor_swap_rate * t)

        # Or we can price it using the standard formula for annuity.
        # For which we'll need to switch to half-yearly compounding frequency.
        r = libor_swap_rate_hy / 2
        one_dollar_annuity_ = notional * coupon_rate / coupon_frequency\
            * (1 - (1 + r) ** -len(cashflow_descr.timeline)) / r
        self.assertTrue(np.allclose(one_dollar_annuity, one_dollar_annuity_))


class CME10YearTNoteFuturesTestCase(unittest.TestCase):
    def setUp(self):
        self.past_date = date(2023, 10, 1)

    def test_ticker_symbols(self):
        next_4_expiry_dates = (date(2023, 12, 1), date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1))
        inferrer = futures_rates.CME10YearTNoteFuturesYields(self.past_date)
        tickers, months = list(zip(*inferrer.get_next_n_quarter_tickers(4)))
        self.assertEqual(tickers, ('ZNZ23.CBT', 'ZNH24.CBT', 'ZNM24.CBT', 'ZNU24.CBT'))
        self.assertEqual(months, next_4_expiry_dates)

    def test_tnote_yields(self):
        # Inferring future Fed Funds and 10 Year Treasury Note yields for the futures contracts traded on CME
        cur_date = date.today()
        date_1_week_ago = cur_date - relativedelta(weeks=+1)
        inferrer_tnote_yields = futures_rates.CME10YearTNoteFuturesYields(cur_date)
        tr10_yields = inferrer_tnote_yields.get_yields_for_next_n_quarters(3)
        print(tr10_yields)
        tr10_yields_1_week_ago = inferrer_tnote_yields.get_yields_for_next_n_quarters(3, date_1_week_ago)
        print(tr10_yields_1_week_ago)

        # Surely implied yields must have changed in one week
        self.assertFalse(tr10_yields.equals(tr10_yields_1_week_ago))

        # Expecting yields to be positive and less than 6%
        self.assertTrue(((tr10_yields < .06) & (tr10_yields > 0.)).all())
        self.assertTrue(((tr10_yields_1_week_ago < .06) & (tr10_yields_1_week_ago > 0.)).all())


class ParYieldToSpotRateConverterTestCase(unittest.TestCase):
    def test_par_yield_to_spot(self):
        maturities = [1/2, 1, 1.5, 2, 2.5, 3]
        spot_rates_dict = par_yield_converter.par_yields_to_spot(
            [.02, .024, .0276, .03084, .033756, .03638], maturities, 2)
        spot_rates = np.asarray([spot_rates_dict[maturity] for maturity in maturities], dtype=np.float64)
        expected_spot_rates = np.asarray([.02, .02402405, .02766857, .03097376, .03397442, .03670043], dtype=np.float64)
        self.assertTrue(np.allclose(spot_rates, expected_spot_rates))

    def test_par_yield_to_spot_with_interpolation(self):
        """
        This test case is based on real Treasury Yields at constant maturities data from 10th July 2025 provided
        by the FRED
        """
        maturities = [0, 1 / 12, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30]
        spot_rates_dict = par_yield_converter.par_yields_to_spot(
            [.04342985, .0436, .0442, .0431, .0407, .0386, .0382, .0393, .0412, .0435, .0487, .0486],
            maturities, 2)
        spot_rates = np.asarray([spot_rates_dict[maturity] for maturity in maturities], dtype=np.float64)
        expected_spot_rates = np.asarray([.04342985, .0436, .0442, .0431, .04067560912311485, .03854709732599071,
                                          .03815107195441536, .039351797004421485, .041459146720940865,
                                          .04410230023109163, .05090930831088958, .049508740485613156], dtype=np.float64)
        self.assertTrue(np.allclose(spot_rates, expected_spot_rates))


if __name__ == '__main__':
    unittest.main()
