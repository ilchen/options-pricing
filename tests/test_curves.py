import unittest
import numpy as np
from datetime import date, datetime, time
from dateutil.relativedelta import relativedelta
from dateutil.relativedelta import FR

from pandas.tseries.offsets import BDay

from pricing import curves


class YieldCurveTestCase(unittest.TestCase):
    def setUp(self):
        today = date.today()
        today = BDay(1).rollforward(datetime.combine(today, time()))
        offsets = [relativedelta(), relativedelta(weeks=+1), relativedelta(months=+1), relativedelta(months=+3),
                   relativedelta(months=+6),
                   relativedelta(years=+1), relativedelta(years=+2)]
        self.curve = curves.YieldCurve(today, offsets, [.02] * len(offsets), compounding_freq=1)

    def test_year_to_date_conversion(self):
        # The 3rd Friday of next January
        maturity_date = self.curve.date + relativedelta(years=1, month=1, day=1, weekday=FR(3))
        t = self.curve.to_years(maturity_date)
        dt = self.curve.to_datetime(t)
        self.assertEqual(dt, datetime.combine(maturity_date, time()))

        # The 3rd Friday of March next year after maturity_date
        maturity_date2 = maturity_date + relativedelta(years=1, month=3, day=1, weekday=FR(3))
        t = self.curve.to_years(maturity_date2)
        dt = self.curve.to_datetime(t)
        self.assertEqual(dt, datetime.combine(maturity_date2, time()))

        # The 3rd Friday of June next year after maturity_date2 lies outside of this curve's range
        maturity_date3 = maturity_date2 + relativedelta(years=1, month=6, day=1, weekday=FR(3))
        with self.assertRaises(ValueError) as context:
            t = self.curve.to_years(maturity_date3)
            self.curve.to_datetime(t)
        self.assertTrue('date is in the past or outside this curve' in str(context.exception))

    def test_curve_points_conversion(self):
        self.assertTrue(np.isclose(self.curve.get_curve_points(120), .02).all())
        self.assertTrue(np.isclose(self.curve.get_curve_points(120, 2), .01990098876724).all())
        self.assertTrue(np.isclose(self.curve.get_curve_points(120, 4), .019851726292815).all())
        self.assertTrue(np.isclose(self.curve.get_curve_points(120, 12), .01981897562304269).all())
        self.assertTrue(np.isclose(self.curve.get_curve_points(120, 0), .0198026272962).all())

if __name__ == '__main__':
    unittest.main