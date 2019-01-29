import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
import os
from pathlib import Path
import matplotlib.pyplot as plt


class DataHandler:

    def __init__(self, covariates_data_path=None, covariates_data_filename=None, reference_data_path=None, X=None,
                 y=None):
        self.covariates_data_path = covariates_data_path
        if X is not None:
            self.X = X
        else:
            self.X = np.genfromtxt(os.path.join(covariates_data_path, covariates_data_filename), delimiter=',', )

        if y is not None:
            self.y = y
            assert self.X.shape[0] == self.y.shape[0], ('The number of samples in covariate data and reference data is '
                                                        'not the same. Check the input data.')
        elif reference_data_path is not None:
            print(reference_data_path)
            # TODO: Allow for response data to be packed in the same file as the covariates
            self.y = np.genfromtxt(covariates_data_path, delimiter=',')
            assert self.X.shape[0] == self.y.shape[0], ('The number of samples in covariate data and reference data is '
                                                        'not the same. Check the input data.')

    def save_result(self, output_filename=None, data_to_save=None):
        output_directory = os.path.join(self.covariates_data_path, 'Decomposition')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        np.savetxt(os.path.join(output_directory, output_filename), data_to_save, delimiter=',')


class PcaWithScaling(DataHandler):

    def __init__(self, dataset_path=None, dataset_filename=None, number_of_components=None, scale_with_std=False):
        super().__init__(dataset_path, dataset_filename)
        self.dataset_path = dataset_path
        self.dataset_filename = dataset_filename
        self.scale_with_std = scale_with_std
        self.number_of_components = number_of_components if number_of_components is not None else min(self.X.shape)
        self.transformed_X = None
        self.explained_variance = None
        self.normalized_explained_variance = None
        self.cumulative_variance = None
        self.components = None
        self.mean = None
        self.mode_number = 0
        self.number_of_std = 1
        self.extremes = np.zeros((self.number_of_components*2, self.components.shape[1]))

    def decompose_with_pca(self):
        """
        Principal component analysis run on centered data. Assumes that self.X is in a shape established for
        scikit-learn - a 2D data frame with samples as rows and features as columns.
        """
        scaler = StandardScaler()
        pca = PCA()

        scaled_pca = Pipeline([
            ('scale', scaler),
            ('pca', pca)])
        scaled_pca.set_params(pca__n_components=self.number_of_components
                              , scale__with_std=self.scale_with_std  # False for shape mode visualization, but should be
                              )                                      # true for discriminant or regression analysis.
        self.transformed_X = scaled_pca.fit_transform(self.X)
        self.explained_variance = scaled_pca.named_steps['pca'].explained_variance_
        self.number_of_components = self.explained_variance.shape[0] - 1  # -1 because the last component is irrelevant
        self.normalized_explained_variance = scaled_pca.named_steps['pca'].explained_variance_ratio_
        self.cumulative_variance = [np.sum(self.normalized_explained_variance[:i+1]) for i in
                                    range(len(self.normalized_explained_variance))]
        self.components = scaled_pca.named_steps['pca'].components_
        self.mean = scaled_pca.named_steps['pca'].mean_

    def get_weighted_mode(self, weight=np.array(1)):
        return weight * self.components[self.mode_number, :]

    def get_extremes_of_mode(self, mode_number=0):
        """
        Calculates extreme loadings along a given mode. Useful for statistical shape analysis.

        :param mode_number: The number of the mode of interest.

        :return: 2D numpy array with positive [0, :] and negative [1, :] extreme calculated according to the number of
        standard deviations in the provided mode.
        """
        self.mode_number = mode_number
        weight = self.number_of_std * np.std(self.transformed_X[self.mode_number, :])
        positive_extreme = self.get_weighted_mode(weight).reshape((1, -1))
        negative_extreme = -positive_extreme
        return np.concatenate((positive_extreme, negative_extreme), axis=0)

    def get_all_extremes(self, number_of_std=1):
        self.number_of_std = number_of_std
        for component in range(self.number_of_components):
            self.extremes[2*component:2*component+2, :] = self.get_extremes_of_mode(component)

    # TODO: These funcions could be transformed to be generic for other types of decomposition
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


class PLSBinaryClassification(DataHandler):

    def __init__(self, dataset_path=None, dataset_filename=None, number_of_components=None, X=None, y=None):
        super().__init__(dataset_path, dataset_filename, X=X, y=y)
        self.dataset_path = dataset_path
        self.dataset_filename = dataset_filename
        self.number_of_components = number_of_components if number_of_components is not None else min(self.X.shape)
        self.count_balance = 0
        print('Number of components: {}'.format(self.number_of_components))
        # PLS-DAfactors
        self.X_centered = np.zeros(self.X.shape)
        self.W = np.zeros((self.X.shape[1], self.number_of_components))
        self.T = np.zeros((self.X.shape[0], self.number_of_components))
        self.P = np.zeros((self.number_of_components, self.X.shape[1]))
        self.q = np.zeros((self.number_of_components, 1))
        self.E, self.f, self.b = (None,) * 3
        self.orthogonal_remodelling_component_number = 0

    @ staticmethod
    def get_pls_factors_vectors_binary(covariates, response):
        X_current = covariates  # could be original or residual
        y_current = response    # could be original or residual

        w = (X_current.T @ y_current).reshape(-1, 1)  # weight vector
        w /= np.linalg.norm(w)  # scaled weight vector
        t = X_current @ w   # score vector
        t_squared_norm = np.sum(np.square(t))
        p = t.T @ X_current / t_squared_norm  # X loadings
        q = y_current.T @ t / t_squared_norm  # y loading (scalar)

        X_resid = X_current - t @ p  # residual data matrix
        y_resid = y_current - t @ q  # residual response vector

        return w, t, p, q, X_resid, y_resid

    def get_pls_factors_binary(self, _x_centered, _y):
        X_current = _x_centered
        y_current = _y

        for component in range(self.number_of_components):
            _w, _t, _p, _q, X_current, y_current = self.get_pls_factors_vectors_binary(X_current, y_current)

            self.W[:, component] = _w.squeeze()  # weights
            self.T[:, component] = _t.squeeze()  # scores
            self.P[component, :] = _p.squeeze()  # X loadings
            self.q[component] = _q  # y loadings

        self.E = X_current  # Residual X
        self.f = y_current  # residual y
        self.b = self.W @ np.linalg.inv(self.P @ self.W) @ self.q + self.f  # relationship between X and y

    def get_predictions_binary(self, samples):
        estimation = []
        for sample in samples:
            estimation.append(1 if sample @ self.b >= 0 else -1)
        return estimation

    def assign_proper_labels_binary(self, class_mask):
        pos = self.y == class_mask[0]
        neg = self.y == class_mask[1]
        self.y[pos], self.y[neg] = 1, -1

    def get_class_balance_binary(self):
        classes, counts = np.unique(self.y, return_counts=True)
        assert len(classes) == 2, 'There must be 2 classes in this decomposition. Check your inputs.'
        # No matter how classes are encoded, they are turned to 1 and -1
        self.assign_proper_labels_binary(class_mask=classes)
        # To ensure proper boundary, centering must we weighted according to the balance of the classes
        counts_ratio = counts[0] / counts[1]
        self.count_balance = (counts_ratio - 1) / (counts_ratio + 1)

    def decompose_with_pls(self, method='da'):
        self.get_class_balance_binary()  # check the balance between classes
        # Centering the covariates
        X_mu = (np.mean(self.X[self.y == 1, :], axis=0) + np.mean(self.X[self.y == -1, :], axis=0)) / 2
        self.X_centered = self.X - X_mu
        assert np.all(self.count_balance * np.mean(self.X_centered[self.y == 1, :], axis=0) -
                      np.mean(self.X_centered, axis=0) <= 10.0e-9), \
            'Classes are not centered properly. Check X centering rules.\n {}' \
            .format(self.count_balance * np.mean(self.X_centered[self.y == 1, :], axis=0) -
                    np.mean(self.X_centered, axis=0))

        if method == 'da':
            self.get_pls_factors_binary(self.X_centered, self.y)
        elif method == 'scikit':
            plsr = PLSRegression(self.number_of_components, scale=False)
            plsr.fit(pls.X_centered, pls.y)

    def get_weighted_component(self, weight=np.array(1)):
        return weight * self.W[self.orthogonal_remodelling_component_number, :]  # weight comes from score distribution

    




if __name__ == "__main__":

    # -----PCA testing-------------------------------------------------------------------------------------------------
    # path_to_data = os.path.join(str(Path.home()), 'Deformetrica', 'deterministic_atlas_ct',
    #                             'output_separate_tmp10_def10_prttpe8_aligned')
    # data_filename = 'DeterministicAtlas__EstimatedParameters__Momenta_Table.csv'
    # momenta_pca = PcaWithScaling(path_to_data, data_filename)
    # momenta_pca.decompose_with_pca()
    # momenta_pca.get_all_extremes(3)
    # momenta_pca.save_all_decomposition_results()
    # print('components shape: {}'.format(momenta_pca.components.shape))
    # print('mean Components : {}'.format(np.mean(momenta_pca.components[17, :])))
    # print('std components: {}'.format(np.std(momenta_pca.transformed_X, axis=1)))
    # print('Cumulative variance: {}'.format(momenta_pca.cumulative_variance[:8]))
    # print('Mean max and mix: {}  {}'.format(max(momenta_pca.mean), min(momenta_pca.mean)))

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
    # ------------------------------------------------------------------------------------------------------------------

    # -----PLS testing--------------------------------------------------------------------------------------------------
    data, target = load_iris(return_X_y=True)
    data = data[0:100, 0:4]
    target = target[0:100]
    pls = PLSBinaryClassification(X=data, y=target)
    pls.decompose_with_pls(method='da')

    plsr = PLSRegression(4, scale=False)
    x_plsr, y_plsr = plsr.fit_transform(pls.X_centered, pls.y)

    plt.scatter(plsr.x_scores_[pls.y == 1, 0], plsr.x_scores_[pls.y == 1, 1], c='red', marker='d')
    plt.scatter(plsr.x_scores_[pls.y == -1, 0], plsr.x_scores_[pls.y == -1, 1], c='blue', marker='x')
    x = np.linspace(-2, 2, 100)

    # print('W:\n {}'.format(pls.W))
    # print('xw:\n {}'.format(plsr.x_weights_))
    # print('T:\n {}'.format(pls.T))
    # print('Xload:\n {}'.format(plsr.x_loadings_))
    # print('P:\n {}'.format(pls.P))
    # print('q:\n {}'.format(pls.q))
    # print('----------------')
    #
    # print('yload:\n {}'.format(plsr.y_loadings_))
    #
    # print('yw:\n {}'.format(plsr.y_weights_))
    # print('x_scores:\n {}'.format(plsr.x_scores_))
    # print('y_scores:\n {}'.format(plsr.y_scores_))
    # print('xr:\n {}'.format(plsr.x_rotations_))
    # print('yr:\n {}'.format(plsr.y_rotations_))
    for i in range(4):
        print('T:\n min: {}, median: {}, max: {}'.format(np.min(pls.T[:, i]), np.median(pls.T[:, i]), np.max(pls.T[:, i])))


    plt.plot(x, np.mean(pls.b[2])*x, 'k.-', linewidth=4)
    plt.plot(x, plsr.coef_[2]*x, 'y.-')
    # plt.show()
