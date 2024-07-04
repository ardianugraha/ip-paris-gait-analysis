import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
import pywt

def compute_fft(x_left, x_right, n_left, n_right, ts):
    x_left_fft = fft.fft(x_left)
    x_right_fft = fft.fft(x_right)
    f_left_fft = fft.fftfreq(n_left, ts)
    f_right_fft = fft.fftfreq(n_right, ts)

    return x_left_fft, x_right_fft, f_left_fft, f_right_fft

def plot_sig_fft_wave(df_patient, dict_left, dict_right):
    for index, row in df_patient.iterrows():
        id_patient = index
        disease = row['Disease Group']
        rate = row['Rate']

        ts = 1.0 / rate

        x_left = np.array(dict_left[id_patient])
        x_right = np.array(dict_right[id_patient]) 

        n_left = len(x_left)
        n_right = len(x_right)

        t_right = np.arange(n_right)
        t_left = np.arange(n_left)

        # Create a 2x2 subplot grid
        fig, axs = plt.subplots(2, 3, figsize=(15, 6))

        # Plot time domain for left leg
        axs[0, 0].plot(t_left, x_left)
        axs[0, 0].set_title(f'Original: Left - {id_patient} {disease}')
        axs[0, 0].set_xlabel('Time (sample)')
        axs[0, 0].set_ylabel('Knee angle (deg)')
        axs[0, 0].spines[['right', 'top']].set_visible(False)

        # Plot time domain for right leg
        axs[1, 0].plot(t_right, x_right)
        axs[1, 0].set_title(f'Original: Right - {id_patient} {disease}')
        axs[1, 0].set_xlabel('Time (sample)')
        axs[1, 0].set_ylabel('Knee angle (deg)')
        axs[1, 0].spines[['right', 'top']].set_visible(False)

        # Compute FFT
        x_left_fft, x_right_fft, f_left_fft, f_right_fft = compute_fft(x_left=x_left,
                                                                       x_right=x_right,
                                                                       n_left=n_left,
                                                                       n_right=n_right,
                                                                       ts=ts)

        # Plot FFT for left leg
        axs[0, 1].stem(f_left_fft[:n_left//2],
                       np.abs(x_left_fft)[:n_left//2],
                       linefmt=':',
                       markerfmt='o',
                       basefmt=' ')
        axs[0, 1].set_title(f'FFT: Left - {id_patient} - {disease}')
        axs[0, 1].set_xlabel('Frequency (Hz)')
        axs[0, 1].set_ylabel('Magnitude')
        axs[0, 1].set_xlim(left=-0.5, right=5.5, auto=True)
        axs[0, 1].spines[['right', 'top']].set_visible(False)

        # Plot FFT for right leg
        axs[1, 1].stem(f_right_fft[:n_right//2],
                       np.abs(x_right_fft)[:n_right//2],
                       linefmt=':',
                       markerfmt='o',
                       basefmt=' ')
        axs[1, 1].set_title(f'FFT: Right - {id_patient} {disease}')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].set_xlim(left=-0.5, right=5.5, auto=True)
        axs[1, 1].spines[['right', 'top']].set_visible(False)

        # Compute CWT
        wavelet = 'gaus1'
        scale = np.arange(1, 64)
        coefficients_left, frequencies_left = pywt.cwt(x_left, scale, wavelet, sampling_period=ts)
        coefficients_right, frequencies_right = pywt.cwt(x_right, scale, wavelet, sampling_period=ts)

        # Plot scalogram for left leg
        im = axs[0, 2].imshow(np.abs(coefficients_left) ** 2,
                         aspect='auto',
                         extent=[0, n_left, frequencies_left[-1], frequencies_left[0]],
                         cmap='jet')
        axs[0, 2].set_title(f'Wavelet: Left - {id_patient} {disease}')
        axs[0, 2].set_xlabel('Time (sample)')
        axs[0, 2].set_ylabel('Frequency (Hz)')
        # axs[0, 2].set_ylim(bottom=0.15)
        axs[0, 2].spines[['top']].set_visible(False)
        plt.colorbar(im, ax=axs[0, 2], label='Power')

        # Plot scalogram for right leg
        im = axs[1, 2].imshow(np.abs(coefficients_right) ** 2,
                         aspect='auto',
                         extent=[0, n_right, frequencies_right[-1], frequencies_right[0]],
                         cmap='jet')
        axs[1, 2].set_title(f'Wavelet: Right - {id_patient} {disease}')
        axs[1, 2].set_xlabel('Time (sample)')
        axs[1, 2].set_ylabel('Frequency (Hz)')
        # axs[1, 2].set_ylim(bottom=0.15)
        axs[1, 2].spines[['top']].set_visible(False)
        plt.colorbar(im, ax=axs[1, 2], label='Power')        

        # # Spectogram
        # f, t, spectr = signal.stft(x_left, fs=rate, window='hann', nperseg=100, noverlap=50)
        # cycle = t * rate
        # im = axs[0, 3].pcolormesh(cycle, f, np.abs(spectr), shading='gouraud')
        # axs[0, 3].set_title(f'STFT: Left - {id_patient} {disease}')
        # axs[0, 3].set_xlabel('Time (sample)')
        # axs[0, 3].set_ylabel('Frequency (Hz)')
        # axs[0, 3].set_ylim(bottom=0, top=3)
        # axs[0, 3].spines[['right', 'top']].set_visible(False)
        # plt.colorbar(im, ax=axs[0, 3], label='Power/Frequency (dB/Hz)')

        # f, t, spectr = signal.stft(x_right, fs=rate, window='hann', nperseg=100, noverlap=50)
        # cycle = t * rate
        # im = axs[1, 3].pcolormesh(cycle, f, np.abs(spectr), shading='gouraud')
        # axs[1, 3].set_title(f'STFT: Right - {id_patient} {disease}')
        # axs[1, 3].set_xlabel('Time (sample))')
        # axs[1, 3].set_ylabel('Frequency (Hz)')
        # axs[1, 3].set_ylim(bottom=0, top=3)
        # axs[1, 3].spines[['right', 'top']].set_visible(False)
        # plt.colorbar(im, ax=axs[1, 3], label='Power/Frequency (dB/Hz)')

        plt.tight_layout()
        plt.show()