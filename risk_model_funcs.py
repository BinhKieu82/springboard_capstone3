#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

#%%
def fit_pca(returns, num_factor_exposures, svd_solver):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_factor_exposures : int
        Number of factors for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """
    mod = PCA(n_components=num_factor_exposures,svd_solver=svd_solver)
    return mod.fit(returns)

#%%
def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """
    Get the factor betas from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    factor_beta_indices : 1 dimensional Ndarray
        Factor beta indices
    factor_beta_columns : 1 dimensional Ndarray
        Factor beta columns

    Returns
    -------
    factor_betas : DataFrame
        Factor betas
    """
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1
    
    factor_betas = pd.DataFrame( 
            pca.components_.T, #a greater absolute weight "pull" the PC more to that feature's direction
            index = factor_beta_indices,
            columns = factor_beta_columns
        )
    return factor_betas

#%%
def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Get the factor returns from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    returns : DataFrame
        Returns for each ticker and date
    factor_return_indices : 1 dimensional Ndarray
        Factor return indices
    factor_return_columns : 1 dimensional Ndarray
        Factor return columns

    Returns
    -------
    factor_returns : DataFrame
        Factor returns
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1

    factor_returns = pd.DataFrame(
            pca.transform(returns), # factor returns estimated by regression (pca.transform), using 20 components
            index = factor_return_indices,
            columns = factor_return_columns
        )
    return factor_returns

#%%
def factor_cov_matrix(factor_returns, ann_factor):
    """
    Get the factor covariance matrix

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns
    ann_factor : int
        Annualization factor

    Returns
    -------
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    """
    factor_cov_matrix = np.diag(factor_returns.var(axis=0, ddof=1)*ann_factor) # size 20x20 
    
    return factor_cov_matrix

#%%
def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Get the idiosyncratic variance matrix

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        Factor returns
    factor_betas : DataFrame
        Factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    """
    common_returns = pd.DataFrame(
            data=np.dot(factor_returns,factor_betas.T),
            index=returns.index,
            columns=returns.columns
        )        
    residuals = returns - common_returns
    idio_var_matrix = pd.DataFrame(
            data=np.diag(np.var(residuals))*ann_factor,
            index=returns.columns,
            columns=returns.columns
        )
    return idio_var_matrix

#%%
def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Get the idiosyncratic variance vector

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Idiosyncratic variance Vector
    """
    idio_var_vector = pd.DataFrame(
            data=np.diag(idiosyncratic_var_matrix.values),
            index=returns.columns
        )    
    return idio_var_vector

#%%
def predict_portfolio_risk(factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights):
    """
    Get the predicted portfolio risk
    
    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        Factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2

    portfolio_cov_matrix = (np.dot(np.dot(factor_betas,factor_cov_matrix),factor_betas.T) + idiosyncratic_var_matrix)
    predict_portfolio_risk = np.sqrt(np.dot(np.dot(weights.T,portfolio_cov_matrix),weights))
    
    return float(predict_portfolio_risk)






# %%
