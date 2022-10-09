# options-pricing
Python code for pricing European and American options.

The repository contains the following packages :
* `volatility.parameter_estimators` &mdash; contains classes implementing maximum likelihood methods for estimating
   the parameters of the Exponentially Weighted Moving Average (EWMA) and GARCH(1, 1) models for tracking volatility.
   You can read about these models on the Internet or delve into John C. Hull's
   [Risk Management and Financial Institutions](http://www-2.rotman.utoronto.ca/~hull/riskman/index.html)
   or [Options, Futures, and Other Derivatives](http://www-2.rotman.utoronto.ca/~hull/ofod/index.html).
* `volatility.volatility_trackers` &mdash; contains classes to track past and forecast future volatilities using
  EWMA and GARCH(1, 1) models. For the purposes of pricing options GARCH(1, 1) is preferred because it supports
  volatility forecasting for future maturities by incorporating mean reversion (and volatility of equities lends itself
  to mean reversion).
* `pricing.curves` &mdash; contains classes to construct yield curves and obtain discount factors as well as forward
  discount factors.
* `pricing.options` &mdash; contains classes implementing a Black-Scholes-Merton and Binomial-Tree pricers.

I created this repository with a view to being able to utilize freely available data from [FRED](https://fred.stlouisfed.org),
[Eurostat](https://ec.europa.eu/eurostat/web/main/data/database), and [Yahoo-Finance](https://finance.yahoo.com).
Since [pandas-datareader](https://pydata.github.io/pandas-datareader/index.html)
provides an excellent way of tapping into these datasets, I opted for using it.

## Requirements
You'll need python3 and pip. `brew install python` will do if you are on MacOS. You can even forgo installing anything
and run the Jupyter notebooks of this repository in Google cloud, as I outline below.

In case you opt for a local installation, the rest of the dependencies can be installed as follows:
```commandline
python3 -m pip install -r requirements.txt
```

## How to get started
The best way to learn how to use the classes from this repository is to run the example Jupyter notebooks. I created
one each for pricing different kinds of options and put ample comments and explanations to them. To use the notebooks,
please proceed as follows:

After you clone the repo and `cd` into its directory, please run one of the below commands depending on which notebook you are interested in:

### Pricing equity options on a cash dividend paying stock
I prepared one example notebook for pricing an option in USD on a US stock (Apple):
```commandline
jupyter notebook equity-options-pricing-example.ipynb
```
A full run of this notebook can be seen [here for Equity Options Pricing](https://github.com/ilchen/options-pricing/blob/main/equity-options-pricing-example.ipynb).

And another example notebook for pricing an option in Euro on a stock priced in Euro (Shell plc):
```commandline
jupyter notebook euro-equity-options-pricing-example.ipynb
```
A full run of this notebook can be seen [here for Euro Equity Options Pricing](https://github.com/ilchen/options-pricing/blob/main/euro-equity-options-pricing-example.ipynb).

### Pricing equity index options
I prepared one example notebook for pricing an option in USD on SP500:
```commandline
jupyter notebook equity-index-options-pricing-example.ipynb
```
A full run of this notebook can be seen [here for Equity Index Options Pricing](https://github.com/ilchen/options-pricing/blob/main/equity-index-options-pricing-example.ipynb).

And another example notebook for pricing an option in Euro on AEX (a capitalization-weighted index of 25 largest Dutch companies:
```commandline
jupyter notebook euro-equity-index-options-pricing-example.ipynb
```
A full run of this notebook can be seen [here for Euro Equity Index Options Pricing](https://github.com/ilchen/options-pricing/blob/main/euro-equity-index-options-pricing-example.ipynb).


You can also run these notebooks in Google cloud. This way you don't need to install anything locally. This takes just a few seconds:
1. Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) in your browser
2. In the modal window that appears select `GitHub`
3. Enter the URL of this repository's notebook, e.g.: `https://github.com/ilchen/options-pricing/blob/main/equity-options-pricing-exmple.ipynb`
4. Click the search icon
5. Enjoy
