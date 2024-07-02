import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats


class FrequencyMagnitudeAnalyzer:
    def __init__(self, df_patient, dict_left, dict_right):
        self.df_patient = df_patient
        self.dict_left = dict_left
        self.dict_right = dict_right
        self.top_freq_df = pd.DataFrame(columns=['id_patient', 'top freq left', 'top mag left', 'top freq right', 'top mag right'])
        self.top_freq_impacted_df = pd.DataFrame(columns=['id_patient', 'side', 'top freq', 'top mag'])

    def compute_fft(self, x_left, x_right, rate):
        n_left = len(x_left)
        n_right = len(x_right)
        ts = 1.0 / rate

        x_left_fft = np.fft.fft(x_left)
        x_right_fft = np.fft.fft(x_right)
        f_left_fft = np.fft.fftfreq(n_left, ts)
        f_right_fft = np.fft.fftfreq(n_right, ts)

        return x_left_fft, x_right_fft, f_left_fft, f_right_fft

    def find_top_frequencies_and_magnitudes(self, x_fft, f_fft, top):
        top_indices = np.argsort(np.abs(x_fft[:len(x_fft)//2]))[::-1][:top]
        top_frequencies = f_fft[top_indices]
        top_magnitudes = np.abs(x_fft[top_indices])
        return top_frequencies, top_magnitudes

    def analyze(self, disease, top):
        for index, row in self.df_patient.iterrows():
            id_patient = index
            rate = row['Rate']
            if disease == 'PT':
                impacted_side = row['Impacted Side'].lower()

            x_left = np.array(self.dict_left[id_patient])
            x_right = np.array(self.dict_right[id_patient])

            x_left_fft, x_right_fft, f_left_fft, f_right_fft = self.compute_fft(x_left, x_right, rate)

            top_frequencies_left, top_magnitudes_left = self.find_top_frequencies_and_magnitudes(x_left_fft, f_left_fft, top)
            top_frequencies_right, top_magnitudes_right = self.find_top_frequencies_and_magnitudes(x_right_fft, f_right_fft, top)

            if disease == 'HC':
                df = pd.DataFrame({
                    'id_patient': [id_patient] * top,
                    'top freq left': top_frequencies_left,
                    'top mag left': top_magnitudes_left,
                    'top freq right': top_frequencies_right,
                    'top mag right': top_magnitudes_right
                })
                self.top_freq_df = pd.concat([self.top_freq_df, df], axis=0)
            elif disease == 'PT' and impacted_side == 'both':
                for freq, mag in zip(top_frequencies_left, top_magnitudes_left):
                    side = 'impacted'
                    self.top_freq_impacted_df = pd.concat([self.top_freq_impacted_df, pd.DataFrame({'id_patient': [id_patient], 'side': [side], 'top freq': [freq], 'top mag': [mag]})], axis=0)
                for freq, mag in zip(top_frequencies_right, top_magnitudes_right):
                    side = 'impacted'
                    self.top_freq_impacted_df = pd.concat([self.top_freq_impacted_df, pd.DataFrame({'id_patient': [id_patient], 'side': [side], 'top freq': [freq], 'top mag': [mag]})], axis=0)
            else:
                for freq, mag in zip(top_frequencies_left, top_magnitudes_left):
                    side = 'impacted' if impacted_side == 'left' else 'non-impacted'
                    self.top_freq_impacted_df = pd.concat([self.top_freq_impacted_df, pd.DataFrame({'id_patient': [id_patient], 'side': [side], 'top freq': [freq], 'top mag': [mag]})], axis=0)
                for freq, mag in zip(top_frequencies_right, top_magnitudes_right):
                    side = 'impacted' if impacted_side == 'right' else 'non-impacted'
                    self.top_freq_impacted_df = pd.concat([self.top_freq_impacted_df, pd.DataFrame({'id_patient': [id_patient], 'side': [side], 'top freq': [freq], 'top mag': [mag]})], axis=0)
        if disease == 'HC':
            return self.top_freq_df
        else:
            return self.top_freq_impacted_df
    
    def plot_histograms(self, disease, bins=30):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
            custom_palette = {'left': 'blue', 'right': 'red'}
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})
            custom_palette = {'red'}

        freq_grid = sns.FacetGrid(melted_df, col='side', hue='side', palette=custom_palette, height=4, aspect=1)
        freq_grid.map(sns.histplot, 'Frequency (Hz)', bins=bins, alpha=0.5).add_legend()
        freq_grid.fig.suptitle('Frequency', y=1.02)
        
        # Create facet grid for magnitudes
        mag_grid = sns.FacetGrid(melted_df, col='side', hue='side', palette=custom_palette, height=4, aspect=1)
        mag_grid.map(sns.histplot, 'Magnitude', bins=bins, alpha=0.5).add_legend()
        mag_grid.fig.suptitle('Magnitude', y=1.02)

        plt.show()

    def plot_gmm(self, disease, bins=30, max_n_components=10):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
            custom_palette = {'left': 'blue', 'right': 'red'}
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})
            custom_palette = {'non-impacted': 'blue', 'impacted': 'red'}
            
        def find_best_gmm(data, max_n_components):
            bics = []
            models = []

            for n in range(1, max_n_components + 1):
                gmm = GaussianMixture(n_components=n, random_state=0).fit(data)
                bics.append(gmm.bic(data))
                models.append(gmm)
            
            best_bic_index = np.argmin(bics)
            return models[best_bic_index], bics

        def plot_gmm_on_grid(ax, data, best_gmm, color, bins):
            x = np.linspace(data.min(), data.max(), 1000)
            logprob = best_gmm.score_samples(x.reshape(-1, 1))
            responsibilities = best_gmm.predict_proba(x.reshape(-1, 1))
            pdf = np.exp(logprob)
            pdf_individual = responsibilities * pdf[:, np.newaxis]

            ax.hist(data, bins=bins, density=True, alpha=0.5, color=color)
            ax.plot(x, pdf, '-k', label='GMM')
            for i in range(best_gmm.n_components):
                ax.plot(x, pdf_individual[:, i], '--', label=f'Component {i+1}')

        # Create facet grid for frequencies
        freq_grid = sns.FacetGrid(melted_df, col='side', hue='side', palette=custom_palette, height=4, aspect=1)
        freq_grid.map(sns.histplot, 'Frequency (Hz)', bins=bins, stat='density', alpha=0.5).add_legend()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        freq_bics = {}

        # Overlay GMM plots on frequency grids
        for ax, side in zip(freq_grid.axes.flat, melted_df['side'].unique()):
            side_data = melted_df[melted_df['side'] == side]
            freq_data = side_data[['Frequency (Hz)']].values
            best_gmm, bics = find_best_gmm(freq_data, max_n_components)
            plot_gmm_on_grid(ax, freq_data, best_gmm, custom_palette[side], bins)
            freq_bics[side] = bics

        freq_grid.fig.suptitle('Frequency', y=1.02)
        plt.show()

        # Create facet grid for magnitudes
        mag_grid = sns.FacetGrid(melted_df, col='side', hue='side', palette=custom_palette, height=4, aspect=1)
        mag_grid.map(sns.histplot, 'Magnitude', bins=bins, stat='density', alpha=0.5).add_legend()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        mag_bics = {}
        # Overlay GMM plots on magnitude grids
        for ax, side in zip(mag_grid.axes.flat, melted_df['side'].unique()):
            side_data = melted_df[melted_df['side'] == side]
            mag_data = side_data[['Magnitude']].values
            best_gmm, bics = find_best_gmm(mag_data, max_n_components)
            plot_gmm_on_grid(ax, mag_data, best_gmm, custom_palette[side], bins)
            mag_bics[side] = bics

        mag_grid.fig.suptitle('Magnitude', y=1.02)
        plt.show()
        
        return freq_bics, mag_bics

    def plot_bic_scores(self, freq_bics, mag_bics):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for side, bics in freq_bics.items():
            axes[0].plot(range(1, len(bics) + 1), bics, marker='o', label=f'{side} side')
        axes[0].set_title('BIC vs. Number of Components for Frequency')
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('BIC')
        axes[0].legend()
        
        for side, bics in mag_bics.items():
            axes[1].plot(range(1, len(bics) + 1), bics, marker='o', label=f'{side} side')
        axes[1].set_title('BIC vs. Number of Components for Magnitude')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('BIC')
        axes[1].legend()

        plt.show()

    def plot_scatter(self, disease):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
            custom_palette = {'left': 'blue', 'right': 'red'}
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})
            custom_palette = {'red'}

        # Create the FacetGrid
        grid = sns.FacetGrid(melted_df, col='side', hue='side', palette=custom_palette, height=4, aspect=1)

        # Map the scatter plot onto the grid
        grid.map(plt.scatter, 'Frequency (Hz)', 'Magnitude', alpha=0.5)
        grid.fig.suptitle('Magnitude vs. Frequency', y=1.02)

        # Add legend
        grid.add_legend()

        # Show the plot
        plt.show()

    def plot_3d_surface(self, disease, freq_bin_size=10, mag_bin_size=10):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            # melted_df = pd.concat([melted_df_left, melted_df_right])
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})
            melted_df_left = melted_df[melted_df['side']== 'non-impacted']
            melted_df_right = melted_df[melted_df['side']== 'impacted']

        # Create a grid of frequencies and magnitudes
        freq_bins_left = np.linspace(melted_df_left['Frequency (Hz)'].min(), melted_df_left['Frequency (Hz)'].max(), freq_bin_size)
        mag_bins_left = np.linspace(melted_df_left['Magnitude'].min(), melted_df_left['Magnitude'].max(), mag_bin_size)
        freq_grid_left, mag_grid_left = np.meshgrid(freq_bins_left[:-1], mag_bins_left[:-1])

        # Create a 2D histogram (frequency-magnitude count matrix)
        count_matrix_left, _, _ = np.histogram2d(melted_df_left['Frequency (Hz)'], melted_df_left['Magnitude'], bins=[freq_bins_left, mag_bins_left])
        
        # Create a grid of frequencies and magnitudes
        freq_bins_right = np.linspace(melted_df_right['Frequency (Hz)'].min(), melted_df_right['Frequency (Hz)'].max(), freq_bin_size)
        mag_bins_right = np.linspace(melted_df_right['Magnitude'].min(), melted_df_right['Magnitude'].max(), mag_bin_size)
        freq_grid_right, mag_grid_right = np.meshgrid(freq_bins_right[:-1], mag_bins_right[:-1])

        # Create a 2D histogram (frequency-magnitude count matrix)
        count_matrix_right, _, _ = np.histogram2d(melted_df_right['Frequency (Hz)'], melted_df_right['Magnitude'], bins=[freq_bins_right, mag_bins_right])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot left side
        surf_left = ax.plot_surface(freq_grid_left, mag_grid_left, count_matrix_left.T, cmap='Blues', alpha=0.2)
        surf_right = ax.plot_surface(freq_grid_right, mag_grid_right, count_matrix_right.T, cmap='Reds', alpha=0.2)

        # Add wireframes for better visualization
        ax.plot_wireframe(freq_grid_left, mag_grid_left, count_matrix_left.T, color='blue', linewidth=0.5, alpha=0.3)
        ax.plot_wireframe(freq_grid_right, mag_grid_right, count_matrix_right.T, color='red', linewidth=0.5, alpha=0.3)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_zlabel('Count')
        ax.set_title('3D Surface Plot of Frequency, Magnitude, and Count')

        # Add color bars for both surfaces
        left_cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        # right_cbar_ax = fig.add_axes([1.05, 0.15, 0.02, 0.7]) # [left, bottom, width, height]

        if disease == 'HC':
            fig.colorbar(surf_left, cax=left_cbar_ax, orientation='vertical', label='Count (Left)')
            fig.colorbar(surf_right, cax=right_cbar_ax, orientation='vertical', label='Count (Right)')
        else:
            # fig.colorbar(surf_left, cax=right_cbar_ax, orientation='vertical', label='Count (Non-Impacted)')
            fig.colorbar(surf_right, cax=left_cbar_ax, orientation='vertical', label='Count (Impacted)')

        plt.show()