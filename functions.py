import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, fft
import dataset.data_preprocessing as dp
import pywt

def get_data(directory, disabled):
    df_meta = dp.get_metadata_df(directory=directory, disabled=disabled)
    left, right, rate = dp.get_data_raw_cycles(directory=directory,
                                               planes=['sagittal'],
                                               list_joints=['Knee'],
                                               disabled=disabled)
    df_rate = pd.DataFrame(rate)
    df_meta2 = pd.concat([df_meta, df_rate], axis=1).rename(columns={0:'Rate'})

    df_lc_length = df_meta2.groupby(['id_patient'])['Left cycle length'].sum()
    df_lc_length.name = 'Left cycle length SUM'
    df_rc_length = df_meta2.groupby(['id_patient'])['Right cycle length'].sum()
    df_rc_length.name = 'Right cycle length SUM'

    df_lc_length_avg = df_meta2.groupby(['id_patient'])['Left cycle length'].mean()
    df_lc_length_avg.name = 'Left cycle length AVG'

    df_rc_length_avg = df_meta2.groupby(['id_patient'])['Right cycle length'].mean()
    df_rc_length_avg.name = 'Right cycle length AVG'

    df_disease = df_meta2.groupby(['id_patient'])['Disease'].first()
    df_rate = df_meta2.groupby(['id_patient'])['Rate'].first()

    df_meta_summary = pd.concat([df_disease,
                                 df_lc_length,
                                 df_rc_length,
                                 df_lc_length_avg,
                                 df_rc_length_avg,
                                 df_rate],
                                 axis=1)

    left_dict = {}
    for i in range(len(left)):
        id_patient = df_meta2['id_patient'][i]
        if id_patient not in left_dict:
            left_dict[id_patient] = []
        for j in range(len(left[i])):
            left_dict[id_patient].append(left[i][j][1])

    right_dict = {}
    for i in range(len(right)):
        id_patient = df_meta2['id_patient'][i]
        if id_patient not in right_dict:
            right_dict[id_patient] = []
        for j in range(len(right[i])):
            right_dict[id_patient].append(right[i][j][1])

    return df_meta2, df_meta_summary, left_dict, right_dict
    

def plot(df_patient, dict_left, dict_right):
    # dict_left = dict_left_HC
    # dict_right = dict_right_HC

    for index, row in df_patient.iterrows():
        id_patient = index
        left_length = row['Left cycle length SUM']
        right_length = row['Right cycle length SUM']
        rate = row['Rate']

       # print(f"id_patient: {id_patient}, Left cycle length SUM: {left_length}, Right cycle length SUM: {right_length}")

        x_left = np.array(dict_left[id_patient])
        x_right = np.array(dict_right[id_patient])      
        t_right = np.arange(left_length)
        t_left = np.arange(right_length)

        # print("left: ", len(x_left))
        # print("right: ", len(x_right))
        # print(len(t_left))
        # print(len(t_right))

        # Create a 2x2 subplot grid
        fig, axs = plt.subplots(2, 4, figsize=(20, 6))

        # Plot time domain for left leg
        axs[0, 0].plot(t_left, x_left)
        axs[0, 0].set_title(f'Original signal: Left leg patient {id_patient}')
        axs[0, 0].set_xlabel('Cycle')
        axs[0, 0].set_ylabel('Knee angle (deg)')

        # Plot time domain for right leg
        axs[1, 0].plot(t_right, x_right)
        axs[1, 0].set_title(f'Original signal: Right leg patient {id_patient}')
        axs[1, 0].set_xlabel('Cycle')
        axs[1, 0].set_ylabel('Knee angle (deg)')

        # Compute FFT
        x_left_fft = fft.fft(x_left)
        x_right_fft = fft.fft(x_right)
        n_left = len(x_left)
        n_right = len(x_right)
        ts = 1.0 / rate
        f_left_fft = fft.fftfreq(n_left, ts)
        f_right_fft = fft.fftfreq(n_right, ts)

        # Plot frequency domain for left leg
        axs[0, 1].stem(f_left_fft[:n_left//2],
                       np.abs(x_left_fft)[:n_left//2],
                       linefmt=':',
                       markerfmt='o',
                       basefmt=' ')
        axs[0, 1].set_title(f'FFT: Left leg patient {id_patient}')
        axs[0, 1].set_xlabel('Frequency (Hz)')
        axs[0, 1].set_ylabel('Magnitude')
        axs[0, 1].set_xlim([-0.5, 5])
        axs[0, 1].set_xticks(np.arange(0, 5, 0.5))
        # axs[0, 1].set_ylim([500, ])

        # Plot frequency domain for right leg
        axs[1, 1].stem(f_right_fft[:n_right//2],
                       np.abs(x_right_fft)[:n_right//2],
                       linefmt=':',
                       markerfmt='o',
                       basefmt=' ')
        axs[1, 1].set_title(f'FFT: Right leg patient {id_patient}')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].set_xlim([-0.5, 5])
        axs[1, 1].set_xticks(np.arange(0, 5, 0.5))
        # axs[1, 1].set_ylim([500, ])
     
        # Compute CWT
        scale = np.arange(1, 64)  # Adjust the range of scales as needed
        coefficients_left, frequencies_left = pywt.cwt(x_left, scale, 'gaus1')
        coefficients_right, frequencies_right = pywt.cwt(x_right, scale, 'gaus1')

        # Plot scalogram for left leg
        axs[0, 2].imshow(np.abs(coefficients_left),
                         aspect='auto',
                         extent=[0, left_length, frequencies_left[-1], frequencies_left[0]],
                         cmap='jet')
        axs[0, 2].set_title(f'Wavelet transform: Left leg patient {id_patient}')
        axs[0, 2].set_xlabel('Cycle')
        axs[0, 2].set_ylabel('Frequency')

        # Plot scalogram for right leg
        axs[1, 2].imshow(np.abs(coefficients_right),
                         spect='auto',
                         extent=[0, right_length, frequencies_right[-1], frequencies_right[0]],
                         cmap='jet')
        axs[1, 2].set_title(f'Wavelet transform: Right leg patient {id_patient}')
        axs[1, 2].set_xlabel('Cycle')
        axs[1, 2].set_ylabel('Frequency')

        f, t, spectr = signal.spectrogram(x_left, fs=rate)

        axs[0, 3].pcolormesh(t, f, np.abs(spectr), shading='gouraud', cmap='jet')
        axs[0, 3].set_title(f'Spectrogram: Left leg patient {id_patient}')
        axs[0, 3].set_xlabel('Time (s)')
        axs[0, 3].set_ylabel('Frequency (Hz)')
        axs[0, 3].set_ylim([0, 5])
        # plt.colorbar(label='Power/Frequency (dB/Hz)')

        f, t, spectr = signal.spectrogram(x_right, fs=rate)

        axs[1, 3].pcolormesh(t, f, np.abs(spectr), shading='gouraud', cmap='jet')
        axs[1, 3].set_title(f'Spectrogram: Right leg patient {id_patient}')
        axs[1, 3].set_xlabel('Time (s)')
        axs[1, 3].set_ylabel('Frequency (Hz)')
        axs[1, 3].set_ylim([0, 5])
        # plt.colorbar(label='Power/Frequency (dB/Hz)')

        # Adjust layout
        plt.tight_layout()
        plt.show()

def trend(df_patient, dict_left, dict_right, top):
    top_freq_df = pd.DataFrame(columns=['id_patient',
                                        'top freq left',
                                        'top mag left',
                                        'top freq right',
                                        'top mag right'])

    for index, row in df_patient.iterrows():
        id_patient = index
        rate = row['Rate']

        x_left = np.array(dict_left[id_patient])
        x_right = np.array(dict_right[id_patient])

        # Compute FFT
        x_left_fft = fft.fft(x_left)
        x_right_fft = fft.fft(x_right)
        n_left = len(x_left)
        n_right = len(x_right)
        ts = 1.0 / rate
        f_left_fft = fft.fftfreq(n_left, ts)
        f_right_fft = fft.fftfreq(n_right, ts)

        # Find the top 10 frequencies and magnitudes for left leg
        top_indices_left = np.argsort(np.abs(x_left_fft[:n_left//2]))[::-1][:top]
        top_frequencies_left = f_left_fft[top_indices_left]
        top_magnitudes_left = np.abs(x_left_fft[top_indices_left])

        # Find the top 10 frequencies and magnitudes for right leg
        top_indices_right = np.argsort(np.abs(x_right_fft[:n_right//2]))[::-1][:top]
        top_frequencies_right = f_right_fft[top_indices_right]
        top_magnitudes_right = np.abs(x_right_fft[top_indices_right])
       
        df = pd.DataFrame({'id_patient': id_patient,
                           'top freq left': top_frequencies_left,
                           'top mag left': top_magnitudes_left,
                           'top freq right': top_frequencies_right,
                           'top mag right': top_magnitudes_right})
        top_freq_df = pd.concat([top_freq_df, df],axis=0)

    # print(top_freq_df)
    melted_df_left = top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'frequency', 'top mag left': 'magnitude'})
    melted_df_right = top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'frequency', 'top mag right': 'magnitude'})
    melted_df_left['side'] = 'left'
    melted_df_right['side'] = 'right'
    melted_df = pd.concat([melted_df_left, melted_df_right])
    print(melted_df)

    # Create the FacetGrid
    grid = sns.FacetGrid(melted_df, col='side', hue='side', palette="viridis")

    # Map the scatter plot onto the grid
    grid.map(plt.scatter, 'frequency', 'magnitude', alpha=0.7)

    # Add legend
    grid.add_legend()

    # Show the plot
    plt.show()

    return melted_df
