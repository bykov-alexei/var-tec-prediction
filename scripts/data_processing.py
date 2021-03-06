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
    if type(neofs) == int:
        neofs = [neofs]

    solver = Eof(arr, center=False)
    for n in neofs: 
        reconstructed = solver.reconstructedField(neofs=n)
        pcs = solver.pcs(npcs=n)
        eofs = solver.eofs(neofs=n)
        yield reconstructed, pcs, eofs


def timestamp_to_features(timestamps):
    x = []
    for t in timestamps:
        date = datetime.datetime.fromtimestamp(t)
        sdoy = np.sin(date.timetuple().tm_yday * 2 * np.pi / 365)
        stod = np.sin((date.hour * 60 + date.minute) * 2 * np.pi / 24 / 60)
        cdoy = np.cos(date.timetuple().tm_yday * 2 * np.pi / 365)
        ctod = np.cos((date.hour * 60 + date.minute) * 2 * np.pi / 24 / 60)
        x.append([sdoy, cdoy, stod, ctod])
    return x


if __name__ == '__main__':
    x, y = load_data('var_tec_reshape.npz')
    x = timestamp_to_features(x)
    y = fill_nan(y)
    y = fill_outliers(y)
    np.savez('data/raw_maps', x=x, y=y)
    neofs = np.arange(1, 31)
    for neofs, (reconstructed, pcs, eofs,) in zip(neofs, reconstruct_data(y, neofs)):
        np.savez('data/reconstructed_maps(neofs=%i)' % neofs, x=x, y=reconstructed, pcs=pcs, eofs=eofs)
