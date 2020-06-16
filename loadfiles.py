def load_data_to_dataframe(path, delimiter=',', header=None):
    """Load a file, whether it is a CSV, txt or XLS file to the data frame format
        Args:
            path (string): The path of the file to be loaded.
            delimiter: Alias for sep
        Returns:
           The file loaded in the data frame format.
    """
    from pandas import read_csv
    return read_csv(path, delimiter= delimiter, header=header)


def load_data_to_numpy_from_gentxt(path):
    """Load a file, whether it is a CSV, txt or XLS file to the Numpy format using a Numpy function.
        Args:
            path (string): The path of the file to be loaded.
        Returns:
           The file loaded in the Numpy format.
    """
    from numpy import genfromtxt
    return genfromtxt(path, delimiter=',', names=True, autostrip=True)


def load_data_to_numpy(path, delimiter=',', header=None):
    """Load a file, whether it is a CSV, txt or XLS file to the Numpy format using a Pandas function.
        Args:
            path (string): The path of the file to be loaded.
            delimiter: Alias for sep
        Returns:
           The file loaded in the Numpy format.
    """
    from pandas import read_csv

    return read_csv(path, delimiter=delimiter, header = header).values
