# options-pricing
Python code for pricing European and American options on different asset classes.

The repository contains the following packages:
* `volatility.parameter_estimators` &mdash; contains classes implementing maximum likelihood methods for estimating
   the parameters of the Exponentially Weighted Moving Average (EWMA) and GARCH(1, 1) models for tracking volatility.
   You can read about these models on the Internet or delve into John C. Hull's
   [Risk Management and Financial Institutions](http://www-2.rotman.utoronto.ca/~hull/riskman/index.html)
   or [Options, Futures, and Other Derivatives](http://www-2.rotman.utoronto.ca/~hull/ofod/index.html). There are
   two implementations for GARCH parameter estimation:
  * a standard one `GARCHParameterEstimator`, which optimizes for all the three GARCH parameters (ω, α, and β);
  * and `GARCHVarianceTargetingParameterEstimator`, which is faster because it uses the so-called variance targeting
      method whereby it sets ω based on the sample variance of price changes. Then it optimises for only two variables
      instead of three as `GARCHParameterEstimator` does. It's marginally less accurate.

* `volatility.volatility_trackers` &mdash; contains classes to track past and forecast future volatilities using
  EWMA and GARCH(1, 1) models. For the purposes of pricing options GARCH(1, 1) is preferred because it supports
  volatility forecasting for future maturities by incorporating mean reversion (and volatility of equities lends itself
  to mean reversion).

* `pricing.curves` &mdash; contains classes to construct yield curves and obtain discount factors as well as forward
  discount factors. Parallel shifts to curve points are supported as well.

* `tests.test_curves` &mdash; a  set of unit tests written with [Python's unittest library](https://docs.python.org/3/library/unittest.html)
  that validate the correctness of the discount curve logic as well as conversions from maturities expressed with
  `datetime.date` objects to maturities expressed in years and back.

* `pricing.options` &mdash; contains classes implementing a Black-Scholes-Merton and Binomial-Tree pricers.

* `tests.test_options` &mdash; a fairly extensive set of unit tests written with [Python's unittest library](https://docs.python.org/3/library/unittest.html)
  that validate the correctness of options pricing logic for both Black-Scholes-Merton and Binomial-Tree pricers. I test
  for the correctness of calculated greeks, the put-call parity and do many additional checks.

I created this repository with a view to being able to utilize freely available data from [FRED](https://fred.stlouisfed.org),
[ECB](https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/html/index.en.html), and [Yahoo-Finance](https://finance.yahoo.com).
I use the following libraries for working with these datasets:
* [pandas-datareader](https://pydata.github.io/pandas-datareader/index.html) for FRED data
* [sdmx](https://sdmx1.readthedocs.io/en/latest/) for the ECB data
* [yfinance](https://pypi.org/project/yfinance/) for Yahoo-Finance data

## Requirements
You'll need python3 and pip. `brew install python` will do if you are on MacOS. You can even forgo installing anything
and run the Jupyter notebooks of this repository in Google cloud, as I outline below.

In case you opt for a local installation, the rest of the dependencies can be installed as follows:
```commandline
python3 -m pip install -r requirements.txt
```

## How to get started
The best way to learn how to use the classes from this repository is to run the example Jupyter notebooks. I created
one each for pricing different kinds of options and put ample comments and explanations in them. To use the notebooks,
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

And another example notebook for pricing an option in Euro on AEX (a capitalization-weighted index of 25 largest Dutch companies):
```commandline
jupyter notebook euro-equity-index-options-pricing-example.ipynb
```
A full run of this notebook can be seen [here for Euro Equity Index Options Pricing](https://github.com/ilchen/options-pricing/blob/main/euro-equity-index-options-pricing-example.ipynb).

### Pricing currency options
I prepared one example notebook for pricing an option on EURUSD:
```commandline
jupyter notebook fx-options-pricing-example.ipynb
```
A full run of this notebook can be seen [here for Currency Options Pricing](https://github.com/ilchen/options-pricing/blob/main/fx-options-pricing-example.ipynb).

## How to run in Google cloud
You can also run these notebooks in Google cloud. This way you don't need to install anything locally. This takes just a few seconds:
1. Go to [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) in your browser
2. In the modal window that appears select `GitHub`
3. Enter the URL of this repository's notebook, e.g.: `https://github.com/ilchen/options-pricing/blob/main/equity-options-pricing-exmple.ipynb`
4. Click the search icon
5. As you open the notebook in Google Colaboratory, please don't forget to uncomment the commands in the first cell
of the notebook and run them.
6. Enjoy.  
  In all the notebooks I make use of python code I developed as part of this project or dependencies that are not by
  default provisioned in Google Colaboratory. When running these notebooks in Colaboratory, it's important to clone this 
  repository and cd to it. I crated a commented out cell at the beginning of each notebook to make it easier.
  Please don't forget to uncomment its content and run it first. E.g. here's one from fx-options-pricing-example.ipynb:
  ```
  # Uncomment if running in Google Colaboratory, otherwise the import of the curves module in the cell below will fail
#!git clone -l -s https://github.com/ilchen/options-pricing.git cloned-repo
#%cd cloned-repo

# Install the latest version of pandas-datareader, yfinance, and pandas-market-calendars
# !pip install pandas-datareader -U
# !pip install sdmx1 -U
# !pip install yfinance -U
  ```
