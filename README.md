# 4D_Automark

This package implemnents a method for modeling photon counts collected from astronomical sources. Extended from [\[Automark\]](https://github.com/astrostat/Automark), photon counts are binned as 4D grids of voxels (time, energy band and x-y coordinates), and viewed as a time series of non-homogeneous Poisson images. The method aims to detect the location of change points and to estimate the image segmentation for images simultaneously. In the underlying methodology, it is assumed that at each time point, the corresponding 3D image is an unknown 3D piecewise constant function corrupted by Poisson noise. It also assumes that all 3D images between any two adjacent change points (in time domain) share the same unknown 3D piecewise constant function.


## Installation
```
pip install git+https://github.com/kevinxucong/4D_Automark.git#egg=Automark_4D
```

## Example: Proxima Centauri

Data preparation.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits


# load Proxima Centauri dataset
filename = "0307_0049350101_EPN_S003_ImagingEvts.ds"
d = fits.getdata(filename)
df = pd.DataFrame(np.array(d).byteswap().newbyteorder())

# binning the list of events
# get data (np.array of size (number of bands, T, m, n)) and exptimes (list of length T)

m = 64
n = 64
T = 60
num_band = 3

xmin = 25500
xmax = 27500
ymin = 26500
ymax = 28500
tmin = 113977025
tmax = 114043049

e0 = 200
e1 = 1000
e2 = 3000
e3 = 10000
dx = float(xmax - xmin) / n
dy = float(ymax - ymin) / m
dt = float(tmax - tmin) / T

df['x_n'] = df.X.apply(lambda x: int((x-xmin)//dx))
df['y_n'] = df.Y.apply(lambda y: int((y-ymin)//dy))
df['e_n'] = df.PI.apply(lambda energy: (energy > e0) * 1 + (energy > e1) * 1 + (energy > e2) * 1 + (energy > e3) * 1 - 1)
df['t_n'] = df.TIME.apply(lambda t: int((t-tmin)//dt))
df = df[(df.x_n >= 0) & (df.x_n < n)]
df = df[(df.y_n >= 0) & (df.y_n < m)]
df = df[(df.e_n >= 0) & (df.e_n < num_band)]
df = df[(df.t_n >= 0) & (df.t_n < T)]
df_group = df.groupby(by = ['e_n','t_n', 'y_n', 'x_n'], as_index=False).agg({'TIME': 'count'})

data = np.zeros((num_band, T, m, n))
for i in range(df_group.shape[0]):
    temp = df_group.iloc[i]
    data[temp.e_n, temp.t_n, temp.y_n, temp.x_n] += temp.TIME
exptimes = [dt]*T
```

Now we fit a homogeneouse model for time interval [0, 6). First we need to settle down bg (list of small elements with length num_band) for numerical stability. And then we need a list of initial seed (init_seed_im: list of length m*n). If we do not specify the initial seeds and keep init_seed_im=None, seeds will be allocated automatically.

```python
bg = [1e-12 for _ in range(num_band)]
mdl, label = get_fitted(data[:,0:6], exptimes[0:6], bg, init_seed_im=None, n_grid=3, par_median_smooth=3, par_local_maximum_size=5, par_local_maximum_threshold=1)
```

Plot the fitted images. 
```python
images = get_fitted_image_from_label(data[:,0:6], label, exptimes[0:6])
for nb in range(num_band):
    plt.subplot(2,2,nb+1)
    plt.imshow(images[nb], vmin=0)
    plt.colorbar()
    plt.title('band={}'.format(nb))
```
![Image of fitted images](https://github.com/kevinxucong/4D_Automark/blob/master/readme/plot_git_1.png)

Now we apply backward elimination to get the fitted model with change points. Initial breaks are set to be every possible location.

```python
init_breaks = [i for i in range(1,T)]
final_break_ls, final_label_ls, min_mdl, merge_mdl_curve, K, break_list, label_list, mdl_list = backward_elimination(data, exptimes, bg, init_breaks)
```

Plot the light curves as well as the change points.
```python
light = np.sum(data, axis=(2,3))

color_ls = ['r', 'y', 'b']
for nb in range(num_band):
    plt.semilogy(light[nb], label='band={}'.format(nb), color=color_ls[nb])
plt.legend(loc='upper left')

for _,t in break_ls[:-1]:
    plt.plot([t,t],[0, 50000], 'k')
plt.xlabel('time point')
plt.ylabel('photon count')
```
![Image of change points](https://github.com/kevinxucong/4D_Automark/blob/master/readme/plot_git_2.png)


Supose we want to highlight the key pixels for the 0th band the change point between the 6th and the 7th intervals. (significance level = 0.01)
```python
label1 = final_label_ls[6]
label2 = final_label_ls[7]
data1 = data[:, final_break_ls[6][0]: final_break_ls[6][1]]
data2 = data[:, final_break_ls[7][0]: final_break_ls[7][1]]
temp_p, temp_increase, temp_decrease = highlight(data1, data2, label1, label2, nb = 0, q=1e-2, method = 2)

plt.imshow(temp_increase,vmin=-1,vmax=1,cmap='RdBu')
plt.colorbar()
```

![Image of highlighted pixels](https://github.com/kevinxucong/4D_Automark/blob/master/readme/plot_git_3.png)


## References
* Cong Xu, Hans Moritz Günther, Vinay L. Kashyap, Thomas C. M. Lee and Andreas Zezas
Change point detection and image segmentation for time series of astrophysical images. [\[arXiv\]](https://arxiv.org/abs/2101.11202)
