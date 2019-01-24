import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
import matplotlib.pyplot as plt


class DataReader:

    def __init__(self, covariates_data_path=None, covariates_data_filename=None, reference_data_path=None):
        self.covariates_data_path = covariates_data_path
        self.X = np.genfromtxt(os.path.join(covariates_data_path, covariates_data_filename), delimiter=',', )

        if reference_data_path is not None:
            self.y = np.genfromtxt(covariates_data_path, delimiter=',')
            assert self.X.shape[0] == self.y.shape[0], ('The number of samples in covariate data and reference data is '
                                                        'not the same. Check the paths and relevant files.')

    def save_result(self, output_filename=None, data_to_save=None):
        output_directory = os.path.join(self.covariates_data_path, 'Decomposition')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        np.savetxt(os.path.join(output_directory, output_filename), data_to_save, delimiter=',')


class PcaWithScaling(DataReader):

    def __init__(self, dataset_path=None, dataset_filename=None, number_of_components=None, scale_with_std=False):
        self.dataset_path = dataset_path
        self.dataset_filename = dataset_filename
        self.number_of_components = number_of_components
        self.scale_with_std = scale_with_std
        super().__init__(dataset_path, dataset_filename)

        self.transformed_X = None
        self.explained_variance = None
        self.normalized_explained_variance = None
        self.cumulative_variance = None
        self.components = None
        self.mean = None
        self.mode_number = 0
        self.number_of_std = 1
        self.decompose()
        self.extremes = np.zeros((self.number_of_components*2, self.components.shape[1]))

    def decompose(self):

        scaler = StandardScaler()
        pca = PCA()

        scaled_pca = Pipeline([
            ('scale', scaler),
            ('pca', pca)])
        scaled_pca.set_params(pca__n_components=self.number_of_components
                              , scale__with_std=self.scale_with_std
                              )
        self.transformed_X = scaled_pca.fit_transform(self.X)
        self.explained_variance = scaled_pca.named_steps['pca'].explained_variance_
        self.number_of_components = self.explained_variance.shape[0] - 1  # The last component is irrelevant
        self.normalized_explained_variance = scaled_pca.named_steps['pca'].explained_variance_ratio_
        self.cumulative_variance = [np.sum(self.normalized_explained_variance[:i+1]) for i in
                                    range(len(self.normalized_explained_variance))]
        self.components = scaled_pca.named_steps['pca'].components_
        self.mean = scaled_pca.named_steps['pca'].mean_

    def get_weighted_mode(self, weight=np.array(1)):
        return weight * self.components[self.mode_number, :]

    def get_extremes_of_mode(self, mode_number=0):
        self.mode_number = mode_number
        weight = self.number_of_std * np.std(self.transformed_X[self.mode_number, :])
        positive_extreme = self.get_weighted_mode(weight).reshape((1, -1))
        negative_extreme = -positive_extreme
        return np.concatenate((positive_extreme, negative_extreme), axis=0)

    def get_all_extremes(self, number_of_std=1):
        self.number_of_std = number_of_std
        for component in range(self.number_of_components):
            self.extremes[2*component:2*component+2, :] = self.get_extremes_of_mode(component)

    def save_transformed_data(self):
        self.save_result('modes.csv', self.transformed_X)

    def save_extremes(self):
        self.save_result('extreme_momenta.csv', self.extremes)

    def save_eigenvectors(self):
        self.save_result('eigenvectors.csv', self.components)

    def save_explained_variance(self):
        self.save_result('explained_variance.csv', self.explained_variance)
        self.save_result('normalized_explained_variance.csv', self.normalized_explained_variance)

    def save_all_decomposition_results(self):
        self.save_eigenvectors()
        self.save_explained_variance()
        self.save_extremes()
        self.save_transformed_data()


if __name__ == "__main__":

    path_to_data = os.path.join(str(Path.home()), 'Deformetrica', 'deterministic_atlas_ct',
                                'output_separate_tmp10_def10_prttpe8_aligned')
    data_filename = 'DeterministicAtlas__EstimatedParameters__Momenta_Table.csv'
    momenta_pca = PcaWithScaling(path_to_data, data_filename)
    momenta_pca.get_all_extremes(3)
    momenta_pca.save_all_decomposition_results()
    print('components shape: {}'.format(momenta_pca.components.shape))
    print('mean Components : {}'.format(np.mean(momenta_pca.components[17, :])))
    print('std components: {}'.format(np.std(momenta_pca.transformed_X, axis=1)))
    print('Cumulative variance: {}'.format(momenta_pca.cumulative_variance[:8]))
    print('Mean max and mix: {}  {}'.format(max(momenta_pca.mean), min(momenta_pca.mean)))

    # plt.bar([*range(len(momenta_pca.normalized_explained_variance))], momenta_pca.normalized_explained_variance,
    #         alpha=0.7)
    # plt.plot(momenta_pca.cumulative_variance, 'r-.')
    # plt.axvline(5)
    # plt.axvline(6)
    # plt.axvline(7)
    # plt.axhline(0.8)
    # plt.show()

    # plt.scatter(momenta_pca.transformed_X[:,0], momenta_pca.transformed_X[:,1])
    # plt.xlabel('Mode 1')
    # plt.ylabel('Mode 2')
    # plt.show()
