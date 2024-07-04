import numpy as np
import pandas as pd
from scipy import interpolate
import pywt
from skimage.metrics import structural_similarity as ssim


def interpolate_array(src_array, target_shape):
    src_shape = src_array.shape
    x = np.linspace(0, src_shape[0], src_shape[0])
    y = np.linspace(0, src_shape[1], src_shape[1])
    new_x = np.linspace(0, src_shape[0], target_shape[0])
    new_y = np.linspace(0, src_shape[1], target_shape[1])
    interpolator = interpolate.RectBivariateSpline(x, y, src_array)
    return interpolator(new_x, new_y)

def intp_wavelet(df_meta_signal, dict_left_signal, dict_right_signal):
    dict_left_wav = {}
    dict_right_wav = {}
    dict_ssim = {}
    
    wavelet = 'gaus1'
    scale = np.arange(1, 64)

    for i, row in df_meta_signal.iterrows():
        rate = row['Rate']
        ts = 1.0 / rate
        
        left = np.array(dict_left_signal[i])
        right = np.array(dict_right_signal[i]) 

        coefficients_left, frequencies_left = pywt.cwt(left, scale, wavelet, sampling_period=ts)
        coefficients_right, frequencies_right = pywt.cwt(right, scale, wavelet, sampling_period=ts)

        if coefficients_left.size < coefficients_right.size:
            dict_left_wav[i] = interpolate_array(coefficients_left, coefficients_right.shape)
            dict_right_wav[i] = coefficients_right
        elif coefficients_left.size > coefficients_right.size:
            dict_left_wav[i] = coefficients_left
            dict_right_wav[i] = interpolate_array(coefficients_right, coefficients_left.shape)
        else:
            dict_left_wav[i] = coefficients_left
            dict_right_wav[i] = coefficients_right

        dict_ssim[i] = ssim(np.array(dict_left_wav[i]), np.array(dict_right_wav[i]), 
                                      data_range=np.array(dict_right_wav[i]).max() - np.array(dict_right_wav[i]).min())
    
    df_ssim = pd.DataFrame.from_dict(dict_ssim, orient='index', columns=['SSIM Left-Right'])
    df_meta_signal = pd.concat([df_meta_signal, df_ssim], axis=1)

    return df_meta_signal