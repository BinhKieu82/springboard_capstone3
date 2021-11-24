import numpy as np
import pandas as pd

def bucketing_factors(factor,quantiles):
    """
    Calculates quantiles
    """
    buckets = pd.qcut(factor, q=quantiles, labels=False)
    return buckets