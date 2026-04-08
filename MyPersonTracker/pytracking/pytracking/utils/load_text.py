import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from io import StringIO


def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        # Fallback for mixed-delimiter annotation files (e.g. comma + space lines).
        try:
            with open(path, 'r') as f:
                text = f.read().replace(',', ' ').replace('\t', ' ')
            ground_truth_rect = np.loadtxt(StringIO(text), delimiter=None, dtype=dtype)
            return ground_truth_rect
        except:
            raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    if pd is None:
        return load_text_numpy(path, delimiter, dtype)
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)
