# SPLICE implementation

A Python implementation for projection matrices which, when applied to _X_
1. Achieve linear guardedness with regard to an attribute _z_
2. Preserve covariance with regard to a variable of interest _y_



## Quick  Example

Here's a minimal working example demonstrating the SPLICE projection:

```python
import numpy as np
from proj import proj

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
z = np.random.binomial(1, 0.5, 1000)  # binary protected attribute
y = np.random.binomial(1, 0.5, 1000)  # binary target attribute

# Initialize projector
projector = proj()

# Fit the projection
projector.fit(X, z, y, method='SPLINCE')

# Apply the projection to get novel embeddings
X_proj = projector.apply_projection(X)

# The covariance between X_proj and z is 0
Cov_X_proj_z = projector.est_Cov(X_proj, z)
norm_Cov_X_proj_z = np.linalg.norm(Cov_X_proj_z)
print(f"Norm of Cov(X_proj, z) = {norm_Cov_X_proj_z}")

# The covariance between X_proj and y is the same as the covariance between X and y
Cov_X_proj_y = projector.est_Cov(X_proj, y)
Cov_X_y = projector.est_Cov(X, y)
norm_Cov_X_proj_y = np.linalg.norm(Cov_X_proj_y - Cov_X_y)
print(f"Norm of Cov(X_proj, y) - Cov(X, y) = {norm_Cov_X_proj_y}")

```

## Applying the projection to a model

Here's a minimal working example demonstrating how to add the 

```python
import numpy as np
from proj import proj
from model import ProjectionLayer, ModelWithProj


# assume that the embeddings are given for the specific layer, y and z are given, we initialize the projector object and fit 
layer_id = 'lm_head'
projector = proj()

# note; make sure to fit the projection to the embeddings of the specific layer
projector.fit(embeddings_lm_head, z, y, method='SPLINCE')

# We create a special projectionLayer object - this takes in the previously fitted P, b. 
device = 'cuda'
torch_dtype = torch.float16
projection = ProjectionLayer(projector.P, projector.b, device, torch_dtype)

# To register the projection onto a layer, we first create a ModelWithProj object
# So for instance, if `model' here is an Llama 2 7B model, then we create a new `model_obj' as
model_obj = ModelWithProj(model)

# In the code below,  we register the projection onto the language modelling head, and it is applied to all tokens
model_obj.register_projection_hook(layer_id=layer_id, projection=projection, apply_strategy= 'all')



```