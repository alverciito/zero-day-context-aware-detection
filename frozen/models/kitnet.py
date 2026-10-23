from .__special__ import logging
import numpy as np
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from .oopsie import __param_check__, __check_input__, NETWORK_TYPES
from .cluster import CorClust, KMeans, DBSCAN, create_feature_map


class KitNET:
    def __init__(self,
                 n_features,
                 feature_map=None,
                 autoencoder_size=5,
                 hidden_ratio=0.75,
                 ae_type='original',
                 output_ae_type=None,
                 sequence_length=500,
                 learning_rate=0.002,
                 fm_grace_period=None,
                 ad_grace_period=10_000,
                 execution_window=10_000,
                 clustering='corr',
                 ar=False,
                 monitor=False):
        """
        KitNET constructor.
        :param n_features: The input dimension space (number of columns in data).
        :param autoencoder_size: Number of autoencoders.
        :param fm_grace_period: The number of instances to train the feature mapper before execute mode.
        :param ad_grace_period: The number of instances to train the model before execute mode.
        :param learning_rate: The default SGD learning rate for all autoencoders in the KitNET instance.
        :param hidden_ratio: Compression factor used to determine the number of nodes in the autoencoder hidden layer.
        :param feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
                where the i-th entry contains a list of the feature indices to be assigned to the i-th autoencoder in
                the ensemble. For example, [[2,5,3],[4,0,1],[6,7]]
        :param execution_window: The number of instances to process before updating the ensemble layer.
        :param sequence_length: The length of the input window.
        :param ae_type: The type of autoencoder to use. The default is 'original'.
        :param clustering: The clustering model to use. The default is 'corr'.
        :param ar: If True, the model will use an autoregressive windowed dataset.
        :param output_ae_type: The type of autoencoder to use for the output layer. The default is None.
        :param monitor: If True, the model will print the progress of the training.
        """
        fm_grace_period = __param_check__(n_features, autoencoder_size, hidden_ratio, ae_type, output_ae_type,
                        sequence_length, learning_rate, fm_grace_period, ad_grace_period, execution_window, feature_map)

        if output_ae_type is None:
            output_ae_type = ae_type

        self.autoencoder_config = {
            'type': ae_type,
            'output_type': output_ae_type,
            'hidden_ratio': hidden_ratio,
            'learning_rate': learning_rate,
            'sequence_length': sequence_length,
            'autoencoder_size': autoencoder_size,
            'is_ar': ar
        }

        self.execution_config = {
            'execution_window': execution_window,
            'fm_grace_period': fm_grace_period,
            'ad_grace_period': ad_grace_period
        }

        self.tracking = {
            'exec_loss': list(),
        }

        self.n_features: int = n_features
        self.feature_map: list[list[int]] = feature_map

        if clustering == 'corr':
            self.clustering_model = CorClust(n_features)
        elif clustering == 'kmeans':
            self.clustering_model = KMeans(n_features)
        elif clustering == 'dbscan':
            self.clustering_model = DBSCAN(n_features)
        elif clustering == 'random':
            self.clustering_model = None
            self.feature_map = create_feature_map(n_features, autoencoder_size)
            self.execution_config['ad_grace_period'] += self.execution_config['fm_grace_period']
            self.execution_config['fm_grace_period'] = 0
        else:
            raise ValueError(f"Clustering model {clustering} is not supported.")
        self.clustering_model_name = clustering

        self.autoencoders = list()
        self.output_autoencoder = None

        self.trained_instances = 0
        self.train_stack = list()
        self.execution_stack = list()
        self.execution_stack_count = 0

        self.current_error = list()

        if self.feature_map is not None:
            self.__build_daes__()
        self.monitor = monitor


    def process(self, x: np.ndarray, is_last=False):
        """
        Learning or execute mode of KitNET.
        :param x: A numpy array of length M x n.
        :param is_last: If True, the last instance of the dataset is being processed.
        :return: The anomaly score of x during training (do not use for alerting).
        """
        x = __check_input__(x, self.n_features).reshape(1, -1)
        train_period = self.execution_config['fm_grace_period'] + self.execution_config['ad_grace_period']

        if self.trained_instances >= train_period:
            self.execute(x, is_last)
        elif self.trained_instances >= self.execution_config['fm_grace_period']:
            self.train_dae(x)
            self.trained_instances += x.shape[0]
        else:
            self.train_map(x)
            self.trained_instances += x.shape[0]

    def train_map(self, x: np.ndarray):
        """
        This method trains the feature mapper.
        :param x: A numpy array of length M x n.
        :return: Nothing.
        """
        for _x in x:
            self.clustering_model.update(_x)

        if self.trained_instances + x.shape[0] >= self.execution_config['fm_grace_period']:
            if self.clustering_model_name == 'corr':
                self.feature_map = self.clustering_model.cluster(self.n_features)
            elif self.clustering_model_name == 'kmeans' or self.clustering_model_name == 'dbscan':
                self.feature_map = self.clustering_model.cluster(self.autoencoder_config['autoencoder_size'])
            self.__build_daes__()

        self.current_error.extend([0.0] * x.shape[0])


    def train_dae(self, x: np.ndarray):
        """
        This method trains the denoising autoencoders.
        :param x: A numpy array of length M x n.
        :return: Nothing.
        """
        self.train_stack.append(x)
        train_period = self.execution_config['ad_grace_period'] + self.execution_config['fm_grace_period']

        if self.trained_instances + x.shape[0] >= train_period:
            logging.info(f'[KN] Training {len(self.autoencoders)} AEs...')
            train_data = np.concatenate(self.train_stack, axis=0)
            error_data = np.zeros((train_data.shape[0], len(self.autoencoders)))
            for idx, (ae, fmap) in enumerate(zip(self.autoencoders, self.feature_map)):
                error = ae.train(train_data[:, fmap])
                error_data[:, idx] = error
                _ext_error = np.concatenate([[0.] * self.execution_config['fm_grace_period'],
                                             error], axis=0)
                self.tracking['exec_loss'].append(_ext_error)
            train_error = self.output_autoencoder.train(error_data)

            self.current_error.extend(train_error)

    def execute(self, x, is_last=False):
        """
        This method executes the KitNET model.
        :param x: A numpy array of length M x n.
        :param is_last: If True, the last instance of the dataset is being processed.
        :return: Nothing.
        """
        self.execution_stack.append(x)
        self.execution_stack_count += x.shape[0]

        if self.execution_stack_count >= self.execution_config['execution_window'] or is_last:
            _x = np.concatenate(self.execution_stack)
            error_data = np.zeros((_x.shape[0], len(self.autoencoders)))
            for idx, (ae, fmap) in enumerate(zip(self.autoencoders, self.feature_map)):
                error = ae.execute(_x[:, fmap])
                error_data[:, idx] = error
                self.tracking['exec_loss'][idx] = np.concatenate([self.tracking['exec_loss'][idx], error], axis=0)
            exec_error = self.output_autoencoder.execute(error_data)
            self.execution_stack = list()
            self.execution_stack_count = 0

            self.current_error.extend(exec_error)

    def save(self, path):
        """
        Save the model to a file.
        :param path: The path to the file.
        :return: Nothing.
        """
        if self.autoencoder_config['type'] == 'original':
            with open(f'{path}_model.pkl', 'wb') as f:
                pickle.dump(self, f)
        else:
            if not os.path.exists(path):
                os.mkdir(path)
            for idx, ae in enumerate(self.autoencoders):
                save_path = f'{path}/ae{idx}/'
                ae.model.save(save_path + 'model')
                ae.encoder.save(save_path + 'encoder')
                ae.decoder.save(save_path + 'decoder')
                aux = (ae.model, ae.encoder, ae.decoder)
                ae.model = None
                ae.encoder = None
                ae.decoder = None
                with open(save_path + 'class.pkl', 'wb') as f:
                    pickle.dump(ae, f)
                ae.model, ae.encoder, ae.decoder = aux

            with open(f'{path}/output.pkl', 'wb') as f:
                pickle.dump(self.output_autoencoder, f)

            prev_aes = self.autoencoders
            prev_output = self.output_autoencoder
            self.autoencoders = list()
            self.output_autoencoder = None

            with open(f'{path}/model.pkl', 'wb') as f:
                pickle.dump(self, f)

            self.autoencoders = prev_aes
            self.output_autoencoder = prev_output

    @staticmethod
    def load(path, is_original=False):
        """
        Load the model from a file.
        :param path: The path to the file.
        :param is_original: If True, the model is an original KitNet model.
        :return: The KitNet model.
        """
        if is_original:
            with open(f'{path}_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            with open(f'{path}/model.pkl', 'rb') as f:
                self_model = pickle.load(f)

            with open(f'{path}/output.pkl', 'rb') as f:
                self_model.output_autoencoder = pickle.load(f)

            ae_paths = [f'{path}/{p}' for p in os.listdir(path) if p.startswith('ae')]
            for ae_path in ae_paths:
                ae_model = tf.keras.models.load_model(f'{ae_path}/model')
                ae_encoder = tf.keras.models.load_model(f'{ae_path}/encoder')
                ae_decoder = tf.keras.models.load_model(f'{ae_path}/decoder')
                with open(f'{ae_path}/class.pkl', 'rb') as f:
                    model = pickle.load(f)
                    model.model = ae_model
                    model.encoder = ae_encoder
                    model.decoder = ae_decoder
                self_model.autoencoders.append(model)

            return self_model


    def __build_daes__(self):
        self.autoencoders = list()
        network_type_ae = NETWORK_TYPES[self.autoencoder_config['type']]
        network_type = NETWORK_TYPES[self.autoencoder_config['output_type']]
        logging.info(f'[KN] Building {len(self.feature_map)} {network_type_ae.__name__} AEs '
                     f'& {network_type.__name__}...')
        for ix, fmap in enumerate(self.feature_map):
            ae = network_type_ae(n_visible=len(fmap),
                              sequence_length=self.autoencoder_config['sequence_length'],
                              lr=self.autoencoder_config['learning_rate'],
                              hidden_ratio=self.autoencoder_config['hidden_ratio'],
                              haste=False,
                              ar=self.autoencoder_config['is_ar'],
                              name=f'Autoencoder [{ix}]')
            self.autoencoders.append(ae)

        self.output_autoencoder = network_type(len(self.feature_map),
                                                lr=self.autoencoder_config['learning_rate'],
                                                sequence_length=self.autoencoder_config['sequence_length'],
                                                ar=self.autoencoder_config['is_ar'],
                                                haste=False,
                                                hidden_ratio=self.autoencoder_config['hidden_ratio'],
                                                name='Output AE')


    def show_stats(self, y: np.ndarray = None, save: str = None):
        """
        Show the statistics of the model.
        :param y: The labels of the dataset if provided.
        :param save: The path to save the plot.
        :return: Nothing.
        """
        mse = np.array(self.current_error)
        mse = np.log10(mse + 1)
        grace_period_ad = self.execution_config['ad_grace_period'] + self.execution_config['fm_grace_period']
        grace_period_fm = self.execution_config['fm_grace_period']
        plt.figure(figsize=(10, 5))
        if y is not None:
            plt.fill_between(np.arange(len(mse)), 0, max(mse) * y, color='gray', alpha=0.1, label='Anomaly')
        plt.plot(mse, label="RMSE", alpha=0.8, linewidth=0., color='black', marker='o', markersize=1.)
        plt.axvline(grace_period_ad, color='black', alpha=0.5, linestyle='--', label='Stop AD training')
        plt.axvline(grace_period_fm, color='black', alpha=0.5, linestyle='--', label='Stop FM training')
        plt.xlim(0, len(mse))
        plt.ylim(0, max(mse) * 1.01)
        plt.title(f'Log error for "{self.autoencoder_config["type"].capitalize()}" model, '
                  f'"{self.clustering_model_name.capitalize()}" clustering, {len(self.autoencoders)} AEs, '
                  f'{self.autoencoder_config["hidden_ratio"]} hidden ratio '
                  f'{"(Autoregressive)" if self.autoencoder_config["is_ar"] else ""}.')
        plt.xlabel('Packet Number')
        plt.ylabel('$\log_{10}({RMSE})$')
        try:
            if save is not None:
                np.save(save + '.npy', mse)
                plt.savefig(save + '.eps')
                plt.savefig(save + '.png')
        except Exception as e:
            logging.error(f'Error saving the plot.')
            logging.error(e)
        finally:
            plt.show()

    def show_ae_stats(self, y: np.ndarray = None, save: str = None):
        """
        Show the statistics of the model.
        :param y: The labels of the dataset if provided.
        :param save: The path to save the plot.
        :return: Nothing.
        """
        mse = np.array(self.tracking['exec_loss'])
        mse = np.log10(mse + 1)
        grace_period_ad = self.execution_config['ad_grace_period'] + self.execution_config['fm_grace_period']
        grace_period_fm = self.execution_config['fm_grace_period']
        plt.figure(figsize=(10, 5))
        if y is not None:
            plt.fill_between(np.arange(mse.shape[-1]), 0, np.max(mse) * y, color='gray', alpha=0.1, label='Anomaly')
        for idx, _mse in enumerate(mse):
            plt.plot(_mse, label=f"RMSE AE{idx}", alpha=0.8, linewidth=0., marker='o', markersize=1.)
        plt.axvline(grace_period_ad, color='black', alpha=0.5, linestyle='--', label='Stop AD training')
        plt.axvline(grace_period_fm, color='black', alpha=0.5, linestyle='--', label='Stop FM training')
        plt.xlim(0, mse.shape[-1])
        plt.ylim(0, np.max(mse) * 1.01)
        plt.title(f'Log error for "{self.autoencoder_config["type"].capitalize()}" model, '
                  f'"{self.clustering_model_name.capitalize()}" clustering, {len(self.autoencoders)} AEs, '
                  f'{self.autoencoder_config["hidden_ratio"]} hidden ratio '
                  f'{"(Autoregressive)" if self.autoencoder_config["is_ar"] else ""}.')
        plt.xlabel('Packet Number')
        plt.ylabel('$\log_{10}({RMSE})$')
        plt.legend(loc='upper left')
        try:
            if save is not None:
                np.save(save + '.npy', mse)
                plt.savefig(save + '.eps')
                plt.savefig(save + '.png')
        except Exception as e:
            logging.error(f'Error saving the plot.')
            logging.error(e)
        finally:
            plt.show()

    @property
    def exec_error(self):
        return self.tracking['exec_loss']
