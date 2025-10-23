import numpy as np
import torch
from scipy import stats

def set_seed(seed):
    """
    Set the seed for reproducibility for numpy and torch

     Args:
            seed: int, seed for reproducibility
     
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

def is_one_hot(arr):
    """
    Check if an array is one-hot encoded.
    
    Args:
        arr (np.ndarray): Array of shape (n, k) to check
        
    Returns:
        bool: True if array is one-hot encoded, False otherwise
    """
    # Check if array contains only 0s and 1s
    if not np.all(np.logical_or(arr == 0, arr == 1)):
        print("Array contains values other than 0 and 1.")
        return False
    
    
        
    # Check if each row has exactly one 1
    return np.all(np.sum(arr, axis=1) == 1)
    
def str_to_bool(s):
    """
    Convert string to boolean

     Args:
          s: str, string to convert to boolean
    """
    if s.lower() == 'true':
         return True
    elif s.lower() == 'false':
         return False
    else:
         raise ValueError("Cannot convert string to boolean.")


def get_index_of_last_no_padding_token(tokens, pad_token_id=0):
     """
     Get the index of the last non-padding token in the tokens list.
     
     Args:
          tokens: torch.Tensor, list of tokens
          pad_token_id: int, padding
     
     Returns:
          last_non_pad_token_index: torch.Tensor, index of the last non-padding token
     """
     # get all cases where the token is not the pad_token_id
     non_pad_tokens = (tokens != pad_token_id).nonzero()

     # get the last index of the non_pad_tokens
     last_non_pad_token_index = non_pad_tokens[-1, :]  # Get the last non-pad token

     return last_non_pad_token_index


def est_Cov(X, y):
     """
     Estimate the covariance between X and y

     Args:
          X: np.array, features
          y: np.array, target
     Returns:
          Cov: np.array, covariance between X and y
     """
     # Center both X and y
     X_mean = np.mean(X, axis=0)
     y_mean = np.mean(y, axis=0)
     X_demeaned = X - X_mean
     y_demeaned = y - y_mean
     
     # Calculate the covariance
     n = X.shape[0]
     Cov = np.matmul(X_demeaned.T, y_demeaned)/(n - 1)

     return Cov
 
def est_Cov_torch(X, y):
    """
    Estimate the covariance between X and y using PyTorch tensors.

    Args:
        X: torch.Tensor, features
        y: torch.Tensor, target

    Returns:
        torch.Tensor: Covariance matrix
    """
    # Center both X and y
    X_mean = X.mean(dim=0)
    y_mean = y.mean(dim=0)
    X_demeaned = X - X_mean
    y_demeaned = y - y_mean

    # Calculate the covariance
    n = X.shape[0]
    Cov = (X_demeaned.T @ y_demeaned) / (n - 1)
    
    # reshape to ensure it is a 2D tensor
    if len(Cov.shape) == 1:
        Cov = Cov.unsqueeze(1)

    return Cov

def get_torch_dtype(dtype_str):
    """
    Convert string dtype to torch dtype.
    
    Args:
        dtype_str: str, dtype string representation
    
    Returns:
        torch dtype object
    """
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def get_model_id(model_name):
    """
    Get a simplified model ID from the full model name.
    
    Args:
        model_name: str, full model name/path
        
    Returns:
        str, simplified model identifier
    """
    if "llama" in model_name.lower():
        # Extract version and size
        if "llama-2" in model_name.lower():
            size = model_name.split("-")[-1].replace("hf", "")
            return f"llama2-{size}"
        else:
            return model_name.split("/")[-1]
    else:
        return model_name.split("/")[-1]

def get_layers_to_process(layers, model, model_type):

     # Parse layers to process
    layers_to_process = []

    # gets the last layer
    if layers == "last":
        if model_type == "llama":
            num_layers = len(model.base_model.layers)
            layers_to_process = [num_layers - 1]
        elif model_type == "bert":
            num_layers = len(model.bert.encoder.layer)
            layers_to_process = [num_layers - 1]
    # gets all layers
    elif layers == "all":
        if model_type == "llama":
            layers_to_process = list(range(len(model.base_model.layers)))
        elif model_type == "bert":
            layers_to_process = list(range(len(model.bert.encoder.layer)))
    # gets the lm_head
    elif layers == "lm_head":
        layers_to_process = ["lm_head"]
    
    # Handle "last_x" format where x is a number
    elif layers.startswith("last_") and layers[5:].isdigit():
        # Handle "last_x" format where x is a number
        x = int(layers[5:])
        if model_type == "llama":
            num_layers = len(model.base_model.layers) 
            start_layer = max(0, num_layers - x)
            layers_to_process = list(range(start_layer, num_layers))
        elif model_type == "bert":
            num_layers = len(model.bert.encoder.layer) 
            start_layer = max(0, num_layers - x)
            layers_to_process = list(range(start_layer+1, num_layers))
            
    else:
        # we now select a specific layer
        if not layers.isdigit():
            raise ValueError(f"Invalid layer specification: {layers}. Should be 'last', 'all', 'lm_head', 'last_x', 'x'  where x is a number")
        else:
            layers_to_process = [int(layers)]
          

    if layers == 'all' or layers.startswith('last_'):
        
       # remove the last layer and add the lm_head
        if model_type == "llama":
            layers_to_process = layers_to_process[:-1]
            layers_to_process.append("lm_head")
        
    print(f"Layers to process: {layers_to_process}")
    
    return layers_to_process
      


def one_sided_ttest(coefficient, std_error, null_hypothesis=0, alternative='greater'):
    """
    Conducts a one-sided t-test for a coefficient against a null hypothesis.
    
    Parameters:
    -----------
    coefficient : float
        The estimated coefficient value
    std_error : float
        The standard error of the coefficient
    null_hypothesis : float, default=0
        The null hypothesis value to test against
    alternative : str, default='greater'
        The alternative hypothesis direction, either 'greater' or 'less'
        'greater': H1: coefficient > null_hypothesis
        'less': H1: coefficient < null_hypothesis
        
    Returns:
    --------
    dict
        A dictionary containing:
        - t_stat: The t-statistic
        - p_value: The p-value for the test
        - conclusion: Text explaining the result at 0.05 significance
    """
    if std_error <= 0:
        raise ValueError("Standard error must be positive")
    
    # Calculate t-statistic
    t_stat = (coefficient - null_hypothesis) / std_error
    
    # Calculate p-value based on alternative hypothesis
    if alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_stat, df=np.inf)  # Using infinite df for normal approximation
    elif alternative == 'less':
        p_value = stats.t.cdf(t_stat, df=np.inf)
    else:
        raise ValueError("Alternative must be either 'greater' or 'less'")
    
    # Generate conclusion
    alpha = 0.05
    if p_value < alpha:
        if alternative == 'greater':
            conclusion = f"Reject null hypothesis. Evidence suggests coefficient > {null_hypothesis} (p={p_value:.4f})"
        else:
            conclusion = f"Reject null hypothesis. Evidence suggests coefficient < {null_hypothesis} (p={p_value:.4f})"
    else:
        if alternative == 'greater':
            conclusion = f"Fail to reject null hypothesis. Insufficient evidence that coefficient > {null_hypothesis} (p={p_value:.4f})"
        else:
            conclusion = f"Fail to reject null hypothesis. Insufficient evidence that coefficient < {null_hypothesis} (p={p_value:.4f})"
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'conclusion': conclusion
    }


def get_se_coef(model, X, y):
    
    """_summary_

    Returns:
        _type_: _description_
    """

   
    # get pred
    y_pred = model.predict(X)

    # get residuals
    residuals = y - y_pred

    # get n, p
    n, p = X.shape

    # get var of residuals
    residual_var = np.sum(residuals**2)/(n - p - 1)

    # add intercept to X
    X_design = np.concatenate([np.ones((X.shape[0], 1)),X], axis=1)

    # get var of X
    Sigma = np.linalg.inv(np.dot(X_design.T, X_design)) * residual_var
    se = np.sqrt(np.diag(Sigma))

    return se
