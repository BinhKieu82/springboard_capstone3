import numpy as np
import pandas as pd
# from zipline.assets._assets import Equity  # Required for USEquityPricing
# from zipline.pipeline.data import USEquityPricing
# from zipline.pipeline.classifiers import Classifier
# from zipline.pipeline.engine import SimplePipelineEngine
# from zipline.pipeline.loaders import USEquityPricingLoader
# from zipline.utils.numpy_utils import int64_dtype


def get_high_lows_lookback(high, low, lookback_days):
    """
    Get the highs and lows in a lookback window.
    
    Parameters
    ----------
    high : DataFrame
        High price for each ticker and date
    low : DataFrame
        Low price for each ticker and date
    lookback_days : int
        The number of days to look back
    
    Returns
    -------
    lookback_high : DataFrame
        Lookback high price for each ticker and date
    lookback_low : DataFrame
        Lookback low price for each ticker and date
    """
    #TODO: Implement function
    lookback_high = high.shift(1).rolling(lookback_days).max()
    lookback_low = low.shift(1).rolling(lookback_days).min()
    return lookback_high, lookback_low

def bucketing_factors(factor,quantiles):
    """
    Calculates quantiles
    """
    buckets = pd.qcut(factor, q=quantiles, labels=False)
    return buckets

def resample_prices(prices, freq='M'):
    """
    Resample close prices for each ticker at specified frequency.
    """
    return prices.resample(freq).last()


def compute_log_returns(prices):
    """
    Compute log returns for each ticker.
    """
    log_prices = np.log(prices)
    return log_prices - log_prices.shift(1)

def shift_returns(returns, shift_n):
    """
    Generate shifted returns
   """
    return returns.shift(shift_n)


def build_pipeline_engine(bundle_data, trading_calendar):
    pricing_loader = PricingLoader(bundle_data)

    engine = SimplePipelineEngine(
        get_loader=pricing_loader.get_loader,
        calendar=trading_calendar.all_sessions,
        asset_finder=bundle_data.asset_finder)

    return engine