import os
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

def export_img(df_patient, dict_left, dict_right):
    for index, row in df_patient.iterrows():
        id_patient = index
        disease = row['Disease Group']
        rate = row['Rate']
        dir = f"dataset/{id_patient}"
        os.makedirs(dir, exist_ok=True)
        
        ts = 1.0 / rate
        x_left = np.array(dict_left[id_patient])
        x_right = np.array(dict_right[id_patient]) 

        n_left = len(x_left)
        n_right = len(x_right)

        t_right = np.arange(n_right)
        t_left = np.arange(n_left)

        # Plot time domain for left leg
        fig1, ax1 = plt.subplots()
        ax1.plot(t_left, x_left)
        ax1.set_title(f'Original: Left - {id_patient} - {disease}')
        ax1.set_xlabel('Time (sample)')
        ax1.set_ylabel('Knee angle (deg)')
        ax1.spines[['right', 'top']].set_visible(False)
        fig1.savefig(dir+'/'+f'ori-left-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig1)
        # Plot time domain for right leg
        fig2, ax2 = plt.subplots()
        ax2.plot(t_right, x_right)
        ax2.set_title(f'Original: Right - {id_patient} - {disease}')
        ax2.set_xlabel('Time (sample)')
        ax2.set_ylabel('Knee angle (deg)')
        ax2.spines[['right', 'top']].set_visible(False)
        fig2.savefig(dir+'/'+f'ori-right-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig2)

        # Compute FFT
        x_left_fft, x_right_fft, f_left_fft, f_right_fft = compute_fft(x_left=x_left,
                                                                       x_right=x_right,
                                                                       n_left=n_left,
                                                                       n_right=n_right,
                                                                       ts=ts)

        # Plot frequency domain for left leg
        fig3, ax3 = plt.subplots()
        ax3.stem(f_left_fft[:n_left//2],
                       np.abs(x_left_fft)[:n_left//2],
                       linefmt=':',
                       markerfmt='o',
                       basefmt=' ')
        ax3.set_title(f'FFT: Left - {id_patient} - {disease}')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Magnitude')
        ax3.set_xlim(left=-0.5, right=5.5, auto=True)
        ax3.spines[['right', 'top']].set_visible(False)
        fig3.savefig(dir+'/'+f'fft-left-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig3)
        
        # Plot frequency domain for right leg
        fig4, ax4 = plt.subplots()
        ax4.stem(f_right_fft[:n_right//2],
                       np.abs(x_right_fft)[:n_right//2],
                       linefmt=':',
                       markerfmt='o',
                       basefmt=' ')
        ax4.set_title(f'FFT: Right - {id_patient} - {disease}')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.set_xlim(left=-0.5, right=5.5, auto=True)
        ax4.spines[['right', 'top']].set_visible(False)
        fig4.savefig(dir+'/'+f'fft-right-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig4)

        # Compute CWT
        wavelet = 'gaus1'
        scale = np.arange(1, 64)
        coefficients_left, frequencies_left = pywt.cwt(x_left, scale, wavelet, sampling_period=ts)
        coefficients_right, frequencies_right = pywt.cwt(x_right, scale, wavelet, sampling_period=ts)

        # # Plot scalogram for left leg
        fig5, ax5 = plt.subplots()
        im = ax5.imshow(np.abs(coefficients_left) ** 2,
                         aspect='auto',
                         extent=[0, n_left, frequencies_left[-1], frequencies_left[0]],
                         cmap='jet')
        ax5.set_title(f'Wavelet: Left - {id_patient} - {disease}')
        ax5.set_xlabel('Time (sample)')
        ax5.set_ylabel('Frequency (Hz)')
        # ax5.set_ylim(bottom=0.15)
        ax5.spines[['top']].set_visible(False)
        plt.colorbar(im, ax=ax5, label='Power')
        fig5.savefig(dir+'/'+f'wvlt-left-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig5)
        
        # Plot scalogram for right leg
        fig6, ax6 = plt.subplots()
        im = ax6.imshow(np.abs(coefficients_right) ** 2,
                         aspect='auto',
                         extent=[0, n_right, frequencies_right[-1], frequencies_right[0]],
                         cmap='jet')
        ax6.set_title(f'Wavelet: Right - {id_patient} - {disease}')
        ax6.set_xlabel('Time (sample)')
        ax6.set_ylabel('Frequency (Hz)')
        # ax6.set_ylim(bottom=0.15)
        ax6.spines[['top']].set_visible(False)
        plt.colorbar(im, ax=ax6, label='Power')        
        fig6.savefig(dir+'/'+f'wvlt-right-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig6)

        ## Spectogram
        f, t, spectr = signal.stft(x_left, fs=rate, window='hann', nperseg=100, noverlap=50)
        cycle = t * rate
        fig7, ax7 = plt.subplots()
        im = ax7.pcolormesh(cycle, f, np.abs(spectr), shading='gouraud')
        ax7.set_title(f'STFT: Left - {id_patient} - {disease}')
        ax7.set_xlabel('Time (sample)')
        ax7.set_ylabel('Frequency (Hz)')
        ax7.set_ylim(bottom=0, top=3)
        ax7.spines[['right', 'top']].set_visible(False)
        plt.colorbar(im, ax=ax7, label='Power/Frequency (dB/Hz)')
        fig7.savefig(dir+'/'+f'stft-left-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig7)

        f, t, spectr = signal.stft(x_right, fs=rate, window='hann', nperseg=100, noverlap=50)
        cycle = t * rate
        fig8, ax8 = plt.subplots()
        im = ax8.pcolormesh(cycle, f, np.abs(spectr), shading='gouraud')
        ax8.set_title(f'STFT: Right - {id_patient} - {disease}')
        ax8.set_xlabel('Time (sample))')
        ax8.set_ylabel('Frequency (Hz)')
        ax8.set_ylim(bottom=0, top=3)
        ax8.spines[['right', 'top']].set_visible(False)
        plt.colorbar(im, ax=ax8, label='Power/Frequency (dB/Hz)')
        fig8.savefig(dir+'/'+f'stft-right-{id_patient}-{disease}.jpg', format='jpg')
        plt.close(fig8)