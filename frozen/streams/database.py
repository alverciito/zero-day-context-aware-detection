import numpy as np
import pandas as pd
KITSUNE_DATABASE_PATHS = [
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/Active Wiretap",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/ARP MitM",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/Fuzzing",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/Mirai Botnet",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/OS Scan",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/SSDP Flood",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/SSL Renegotiation",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/SYN DoS",
    "/mnt/8A3A82BA3A82A335/database/KITSUNE/Video Injection"
]
EPSILON = 1e-6
class KitsuneDatabase:
    def __init__(self, regularized: bool = True):
        self.db_paths = KITSUNE_DATABASE_PATHS
        self.db_names = [_.split("/")[-1].replace(" ", "_") for _ in KITSUNE_DATABASE_PATHS]
        self.sub_path = [(or_pth + f"/{pp_path}_dataset.csv", or_pth + f"/{pp_path}_labels.csv")
                         for or_pth, pp_path in zip(KITSUNE_DATABASE_PATHS, self.db_names)]
        self.regularized = regularized

    def get(self, name: str | int) -> tuple:
        """
        This method returns the dataset and labels of the given name.
        :param name: A string with the name of the dataset.
        :return: A tuple with the dataset and labels.
        """
        if name not in self.db_names:
            raise ValueError(f"The name {name} is not in the database.")
        if isinstance(name, int):
            index = name
        else:
            name = name.replace(" ", "_")
            index = self.db_names.index(name)
        dataset_path, labels_path = self.sub_path[index]
        x_df = pd.read_csv(dataset_path)
        y_df = pd.read_csv(labels_path)
        x = x_df.to_numpy()
        y = y_df.to_numpy()[:, -1]
        if self.regularized:
            x = self.regularize(x)
        return x, y

    @staticmethod
    def regularize(x: np.ndarray) -> tuple:
        """
        This method balances the data. Subs mean and divides by std.
        :param x: The data in np.ndarray
        :return: The regularized data.
        """
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + EPSILON)
        return x

    @staticmethod
    def get_db_info(x: np.ndarray, y: np.ndarray) -> dict:
        """
        The database information.
        :param x: The data in np.ndarray
        :param y: The labels in np.ndarray
        :return: A dictionary with database information.
        """
        n_rows = len(x)
        n_dims = x.shape[-1]

        n_ones = np.sum(len(np.where(y == 1)[0]))
        n_zero = np.sum(len(np.where(y == 0)[0]))
        balance = n_ones / n_rows

        dissimilarity = np.mean(np.abs(np.diff(x, axis=0)), axis=1)
        similar_data = np.sum(dissimilarity <= 0.1) / n_rows

        info = {
            'rows': n_rows,
            'cols': n_dims,
            'type_0': int(n_zero),
            'type_1': int(n_ones),
            'balance': 100 * balance,
            'similarity_ratio': similar_data
        }
        return info

    def __iter__(self):
        self.__iter_index = 0
        return self

    def __next__(self):
        if self.__iter_index < len(self.db_names):
            name = self.db_names[self.__iter_index]
            self.__iter_index += 1
            return self.get(name)
        raise StopIteration

    def __len__(self):
        return len(self.db_names)

    def __repr__(self):
        return f"KitsuneDatabase({self.db_paths})"
