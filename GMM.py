import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class DataTransformer:
    """
    Transformer class responsible for processing data to train models with mixed data types.
    
    Methods:
    - __init__(): Initializes the transformer and computes metadata for the input dataset.
    - fit(): Fits Bayesian Gaussian Mixture models for numeric and mixed columns.
    - transform(): Transforms data into a format suitable for model training.
    - inverse_transform(): Reverts transformed data back to its original form.
    """

    def __init__(self, train_data, categorical_list=[], mixed_dict={}, n_clusters=10, eps=0.005):
        self.train_data = train_data
        self.categorical_columns = categorical_list
        self.mixed_columns = mixed_dict
        self.n_clusters = n_clusters
        self.eps = eps

        self.meta = None
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.ordering = []
        self.filter_arr = []
        self.model = []

        logging.info("Initializing DataTransformer.")
        self.meta = self.get_metadata()

    def get_metadata(self):
        """Extracts metadata from the dataset to determine column types and their properties."""
        logging.info("Extracting metadata from the dataset.")
        meta = []
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:, index]
            if index in self.categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({"name": index, "type": "categorical",
                            "size": len(mapper), "i2s": mapper})
            elif index in self.mixed_columns.keys():
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index]
                })
            else:
                meta.append({"name": index, "type": "continuous",
                            "min": column.min(), "max": column.max()})
        return meta

    def fit(self):
        """Fits Bayesian Gaussian Mixture models to numeric and mixed columns."""
        logging.info("Fitting Bayesian Gaussian Mixture models to the data.")
        data = self.train_data.values
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                gm = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    random_state=42
                )
                gm.fit(data[:, id_].reshape(-1, 1))
                self.model.append(gm)

                valid_modes = gm.weights_ > self.eps
                self.components.append(valid_modes)
                self.output_info += [(1, 'tanh'),
                                     (np.sum(valid_modes), 'softmax')]
                self.output_dim += 1 + np.sum(valid_modes)
            elif info['type'] == "mixed":
                # Mixed column handling (skipping implementation for brevity)
                pass
            else:
                self.model.append(None)
                self.output_info += [(info['size'], 'softmax')]
                self.output_dim += info['size']
        logging.info("Fitting completed.")

    def transform(self, data):
        """Transforms the data into a format suitable for model training."""
        logging.info("Transforming data.")
        values = []
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                current = data[:, id_].reshape(-1, 1)
                gm = self.model[id_]
                means = gm.means_.reshape((1, self.n_clusters))
                stds = np.sqrt(gm.covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (4 * stds)
                features = np.clip(features, -0.99, 0.99)
                values.append(features)
            else:
                # Skipping other types for brevity
                pass
        logging.info("Transformation completed.")
        return np.concatenate(values, axis=1)

    def inverse_transform(self, data):
        """Reverts the transformed data back to its original form."""
        logging.info("Performing inverse transformation.")
        data_t = np.zeros((len(data), len(self.meta)))
        st = 0
        for id_, info in enumerate(self.meta):
            if info['type'] == "continuous":
                u = data[:, st]
                gm = self.model[id_]
                means = gm.means_.reshape(-1)
                stds = np.sqrt(gm.covariances_).reshape(-1)
                data_t[:, id_] = u * 4 * stds[np.argmax(gm.predict_proba(
                    data[:, st:st+1]), axis=1)] + means[np.argmax(gm.predict_proba(data[:, st:st+1]), axis=1)]
                st += 1
            else:
                # Skipping other types for brevity
                pass
        logging.info("Inverse transformation completed.")
        return data_t


# Example Usage
if __name__ == "__main__":
    logging.info("Loading data.")
    train_data = pd.read_csv("Credit.csv")[["Amount"]]
    logging.info("Initializing and fitting DataTransformer.")
    transformer = DataTransformer(train_data=train_data)
    transformer.fit()
    transformed_train_data = transformer.transform(train_data.values)

    logging.info("Transformed Data Example:")
    logging.info(transformed_train_data[0])

    logging.info("Performing inverse transformation.")
    inverse_transformed_train_data = transformer.inverse_transform(
        transformed_train_data)

    logging.info("Inverse Transformed Data Example:")
    logging.info(pd.DataFrame(
        inverse_transformed_train_data, columns=["Amount"]).head())
