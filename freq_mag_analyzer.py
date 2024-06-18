import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


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
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})

        # Create facet grid for frequencies
        freq_grid = sns.FacetGrid(melted_df, col='side', hue='side', palette="viridis", height=4, aspect=1)
        freq_grid.map(sns.histplot, 'Frequency (Hz)', bins=bins, kde=True).add_legend()
        freq_grid.fig.suptitle('Frequency', y=1.02)
        
        # Create facet grid for magnitudes
        mag_grid = sns.FacetGrid(melted_df, col='side', hue='side', palette="viridis", height=4, aspect=1)
        mag_grid.map(sns.histplot, 'Magnitude', bins=bins, kde=True).add_legend()
        mag_grid.fig.suptitle('Magnitude', y=1.02)

        plt.show()

    def plot_scatter(self, disease):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})

        # Create the FacetGrid
        grid = sns.FacetGrid(melted_df, col='side', hue='side', palette="viridis", height=4, aspect=1)

        # Map the scatter plot onto the grid
        grid.map(plt.scatter, 'Frequency (Hz)', 'Magnitude', alpha=0.7)
        grid.fig.suptitle('Magnitude vs. Frequency', y=1.02)

        # Add legend
        grid.add_legend()

        # Show the plot
        plt.show()


    def kmeans_cluster(self, disease, n_clusters):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})

        # Perform K-means clustering
        X = melted_df[['Frequency (Hz)', 'Magnitude']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        melted_df['Cluster'] = kmeans.labels_

        # Plot clustered scatter plot
        clustered_grid = sns.FacetGrid(melted_df, col='side', hue='Cluster', height=4, aspect=1)
        clustered_grid.map(plt.scatter, 'Frequency (Hz)', 'Magnitude', alpha=0.7)
        clustered_grid.fig.suptitle('K-means Cluster', y=1.02)

        # Add legend
        clustered_grid.add_legend()

        # Show the plot
        plt.show()
    
    def agglo_cluster(self, disease, n_clusters, linkage):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})
        
        X = melted_df[['Frequency (Hz)', 'Magnitude']].values
        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X)
        melted_df['Cluster'] = agglo.labels_

        # Plot clustered scatter plot
        clustered_grid = sns.FacetGrid(melted_df, col='side', hue='Cluster', height=4, aspect=1)
        clustered_grid.map(plt.scatter, 'Frequency (Hz)', 'Magnitude', alpha=0.7)
        clustered_grid.fig.suptitle('Agglomerative Cluster', y=1.02)

        # Add legend
        clustered_grid.add_legend()

        # Show the plot
        plt.show()

    # def plot_gmm(self, disease, n_components=5, type='freq'):
    #     if disease == 'HC': 
    #         melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
    #         melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
    #         melted_df_left['side'] = 'left'
    #         melted_df_right['side'] = 'right'
    #         melted_df = pd.concat([melted_df_left, melted_df_right])
    #     else:
    #         melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})
        
    #     if type == 'freq':
    #         X = melted_df[['Frequency (Hz)']].values
    #     elif type == 'mag':
    #         X = melted_df[['Magnitude']].values
    #     else:
    #         X = melted_df[['Frequency (Hz)', 'Magnitude']].values

    #     gmm = GaussianMixture(n_components=n_components, random_state=0)
    #     gmm.fit(X)
    #     labels = gmm.predict(X)

    #     melted_df['Cluster'] = labels

    #     # Plot clustered scatter plot with GMM
    #     grid = sns.FacetGrid(melted_df, col='side', hue='Cluster', palette="viridis", height=4, aspect=1)
    #     # grid.map(plt.scatter, col, alpha=0.7)
    #     if type == 'freq':
    #         grid.map(plt.scatter, 'Frequency (Hz)', alpha=0.7)
    #     elif type == 'mag':
    #         grid.map(plt.scatter, 'Magnitude', alpha=0.7)
    #     else:
    #         grid.map(plt.scatter, 'Frequency (Hz)', 'Magnitude', alpha=0.7)
    #     grid.fig.suptitle('GMM', y=1.02)

    #     # Add legend
    #     grid.add_legend()

    #     # Show the plot
    #     plt.show()
    
    def plot_3d_scatter(self, disease):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})

        # Count occurrences
        melted_df['Count'] = melted_df.groupby(['Frequency (Hz)', 'Magnitude'])['Frequency (Hz)'].transform('count')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for side in melted_df['side'].unique():
            side_df = melted_df[melted_df['side'] == side]
            ax.scatter(side_df['Frequency (Hz)'], side_df['Magnitude'], side_df['Count'], label=side)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_zlabel('Count')
        ax.set_title('3D Scatter Plot of Frequency, Magnitude, and Count')
        ax.legend()

        plt.show()

    def plot_3d_surface(self, disease, freq_bin_size=10, mag_bin_size=10):
        if disease == 'HC': 
            melted_df_left = self.top_freq_df[['id_patient', 'top freq left', 'top mag left']].rename(columns={'top freq left': 'Frequency (Hz)', 'top mag left': 'Magnitude'})
            melted_df_right = self.top_freq_df[['id_patient', 'top freq right', 'top mag right']].rename(columns={'top freq right': 'Frequency (Hz)', 'top mag right': 'Magnitude'})
            melted_df_left['side'] = 'left'
            melted_df_right['side'] = 'right'
            melted_df = pd.concat([melted_df_left, melted_df_right])
        else:
            melted_df  = self.top_freq_impacted_df[['id_patient', 'side', 'top freq', 'top mag']].rename(columns={'top freq': 'Frequency (Hz)', 'top mag': 'Magnitude'})
        
        if melted_df.empty:
            print("No data available for plotting.")
            return

        print(f"Data for plotting ({disease}):")
        print(melted_df.head())
        print(melted_df.describe())

        # Create a grid of frequencies and magnitudes
        freq_bins = np.linspace(melted_df['Frequency (Hz)'].min(), melted_df['Frequency (Hz)'].max(), freq_bin_size)
        mag_bins = np.linspace(melted_df['Magnitude'].min(), melted_df['Magnitude'].max(), mag_bin_size)
        freq_grid, mag_grid = np.meshgrid(freq_bins[:-1], mag_bins[:-1])

        # Create a 2D histogram (frequency-magnitude count matrix)
        count_matrix, _, _ = np.histogram2d(melted_df['Frequency (Hz)'], melted_df['Magnitude'], bins=[freq_bins, mag_bins])
        
        if not np.any(count_matrix):
            print("No data available for creating the 3D surface plot.")
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(freq_grid, mag_grid, count_matrix.T, cmap='viridis')  # Transpose count_matrix

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_zlabel('Count')
        ax.set_title('3D Surface Plot of Frequency, Magnitude, and Count')

        plt.show()
