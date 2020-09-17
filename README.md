# 4D_Automark
change point detection and image segmentation for time series of astrophysical images.

## Installation
```
pip install git+https://github.com/kevinxucong/4D_Automark.git#egg=Automark_4D
```

## Example
```python
import numpy as np
from Automark_4D import *

# generate random dataset
m = 5
n = 5
T = 20
num_band = 3
change_points_true = [6, 12, 16]
beta_true = np.ones((num_band, len(change_points_true)+1,m,n))
for i in range(2):
    for j in range(1,3):
        beta_true[:,0, i,j] = 20
for i in range(1,3):
    for j in range(1,3):
        beta_true[:, 1,i,j] = 25
for i in range(2):
    for j in range(2):
        beta_true[:,2,i,j] = 20
for i in range(2):
    for j in range(2):
        beta_true[:,3,i,j] = 40
change_points_true_modified = [0]+change_points_true+[T]
temp = np.zeros((num_band, T,m,n))
for l in range(len(change_points_true)+1):
    for t in range(change_points_true_modified[l], change_points_true_modified[l+1]):
        temp[:,t] = beta_true[:,l]
data = np.random.poisson(temp)

# fit a homogeneouse model for data[:,0:6]
bg = [1e-12 for _ in range(num_band)]
exptimes = [1.0 for _ in range(T)]
result_1 = get_fitted(data[:,0:6], bg, exptimes[0:6], init_seed_im=None, n_grid=3, par_median_smooth=3, par_local_maximum_size=5, par_local_maximum_threshold=1)

# backward elimination to get the fitted model with change points
init_breaks = [2*i for i in range(1,10)]
result_2 = backward_elimination(data, bg, exptimes, init_breaks)

```