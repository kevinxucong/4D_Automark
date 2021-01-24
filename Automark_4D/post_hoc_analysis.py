from scipy.stats import norm

from .mdlsegi_poisson_exptime import *


def get_fitted_image_from_label(data, label, exptimes):
    num_band, T1, m, n = data.shape
    observe_im = data.reshape((num_band, T1, m*n))
    bg = [1e-9]*num_band
    graph = TLGRAPH(observe_im, label, m, n, exptimes, bg)

    fitted = np.zeros((num_band ,m,n))
    for r1, tlregion1 in graph.region_ls.items():
        c = np.sum(tlregion1.spix, axis=1)/(tlregion1.npix*np.sum(graph.exptimes))
        for nb in range(num_band):
            fitted[nb, (np.array(label).reshape(m,n) == r1)] = c[nb]
            
    return fitted


def get_average_image_from_label(data, label):
    num_band, T1, m, n = data.shape
    average_image = np.zeros((num_band ,m,n))

    observe_im = data.reshape((num_band, T1, m*n))
    for target in np.unique(label):
        s = (np.array(label) == target)
        for nb in range(num_band):
            c = np.mean(observe_im[nb, :, s])
            average_image[nb, s.reshape((m,n))] = c

    return average_image


def get_sqrt_average_image_from_label(data, label):
    num_band, T1, m, n = data.shape
    sqrt_average_images = np.zeros((num_band ,m, n))

    observe_im = data.reshape((num_band, T1, m*n))
    for target in np.unique(label):
        s = (np.array(label) == target)
        for nb in range(num_band):
            c = np.mean(np.sqrt(observe_im[nb, :, s]))
            sqrt_average_images[nb, s.reshape((m,n))] = c
            
    return sqrt_average_images


def get_sqrt_var_image_from_label(data, label):
    num_band, T1, m, n = data.shape
    sqrt_var_images = np.zeros((num_band ,m, n))

    observe_im = data.reshape((num_band, T1, m*n))
    for target in np.unique(label):
        s = (np.array(label) == target)
        for nb in range(num_band):
            c = np.var(np.sqrt(observe_im[nb, :, s]), ddof=1)
            sqrt_var_images[nb, s.reshape((m,n))] = c
            
    return sqrt_var_images


def mad(data):
    """
    data: np.array
    return: MAD
    """
    return np.median(np.abs(data - np.median(data)))


def highlight(data1, data2, label1, label2, nb, q, method):
    """
    Two methods of highlighting key pixels
     """
    num_band, _, m, n = data1.shape
    if method == 1:
        average_images = np.zeros((num_band, 2, m, n))
        average_images[:,0] = get_average_image_from_label(data1, label1)
        average_images[:,1] = get_average_image_from_label(data2, label2)

        label_images = np.zeros((2, m, n))
        label_images[0] = np.array(label1).reshape(m,n)
        label_images[1] = np.array(label2).reshape(m,n)

        raw1 = np.sqrt(average_images[nb, 0])
        raw2 = np.sqrt(average_images[nb, 1])
    #     res = highlight_key_pixels.key_regions(raw1, raw2, p=1-q, tau=1, tail="both")

        res = np.zeros((m, n))
        diff = raw2 - raw1
        mad_diff = mad(diff)
        if mad_diff > 0:
            res = (diff - np.mean(diff)) / (mad_diff / norm.ppf(3 / 4))
        elif np.std(diff) > 0:
            res = (diff - np.mean(diff)) / np.std(diff)
    
    elif method == 2:
        sqrt_average_images = np.zeros((num_band, 2, m, n))
        sqrt_average_images[:,0] = get_sqrt_average_image_from_label(data1, label1)
        sqrt_average_images[:,1] = get_sqrt_average_image_from_label(data2, label2)

        sqrt_var_images = np.zeros((num_band, 2, m, n))
        sqrt_var_images[:,0] = get_sqrt_var_image_from_label(data1, label1)
        sqrt_var_images[:,1] = get_sqrt_var_image_from_label(data2, label2)

        label_images = np.zeros((2, m, n))
        label_images[0] = np.array(label1).reshape(m,n)
        label_images[1] = np.array(label2).reshape(m,n)

        M = np.max(label_images)+1
        res = np.zeros((m, n))
        for target in np.unique((label_images[1]*M + label_images[0])):
            s = (label_images[1]*M + label_images[0] == target)
            numer = sqrt_average_images[nb, 1, s][0] - sqrt_average_images[nb, 0, s][0]
            denom = np.sqrt(sqrt_var_images[nb, 0, s][0] + sqrt_var_images[nb, 1, s][0])
            res[s] = numer / denom
        
    p_value = norm.cdf(res)
    res_increase = np.isin(label_images[1], np.unique(label_images[1][p_value > 1-0.5*q]))
    res_decrease = np.isin(label_images[0], np.unique(label_images[0][p_value < 0.5*q]))
    return (p_value, res_increase, res_decrease)
