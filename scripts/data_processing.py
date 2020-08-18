import datetime

import numpy as np
from eofs.standard import Eof

from data_loader import load_data


def fill_outliers(arr):
    arr = np.copy(arr)
    threshold = np.percentile(arr, 99)
    arr[arr >= threshold] = threshold
    return arr


def fill_nan(arr):
    res = np.copy(arr)
    for i, y in enumerate(arr):
        mask = np.isnan(y)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = y[np.arange(idx.shape[0])[:, None], idx]
        res[i] = out
    return res


def reconstruct_data(arr, neofs=16):
    solver = Eof(arr, center=False)
    reconstructed = solver.reconstructedField(neofs=neofs)
    pcs = solver.pcs(npcs=neofs)
    eofs = solver.eofs(neofs=neofs)
    return reconstructed, pcs, eofs


def timestamp_to_features(timestamps):
    x = []
    for t in timestamps:
        date = datetime.datetime.fromtimestamp(t)
        doy = np.sin(date.timetuple().tm_yday * 2 * np.pi / 365)
        tod = np.sin((date.hour + date.minute * 60) * 2 * np.pi / 24 / 60)
        x.append([doy, tod])
    return x


if __name__ == '__main__':
    x, y = load_data('var_tec_reshape.npz')
    x = timestamp_to_features(x)
    y = fill_nan(y)
    y = fill_outliers(y)
    np.savez('data/raw_maps', x=x, y=y)
    for neofs in range(1, 30):
        reconstructed, pcs, eofs = reconstruct_data(y, neofs=neofs)
        np.savez('data/reconstructed(neofs=%i)' % neofs, x=x, y=y)
        np.savez('data/eofs_coefficient(neofs=%i)' % neofs, x=x, y=pcs, eofs=eofs)
