from __future__ import annotations

import h5py
import yaw
from rail.core.data import DataHandle


class CorrFuncHandle(DataHandle):
    """Class to act as a handle for yaw.CorrFunc HDF5 data.

    Parameters
    ----------
    tag : str
        The tag under which this data handle can be found in the store
    data : yaw.CorrrFunc or None
        The associated data
    path : str or None
        The path to the associated file
    creator : str or None
        The name of the stage that created this data handle
    """

    data: yaw.CorrFunc | None

    @classmethod
    def _open(cls, path: str, **kwargs) -> h5py.File:
        return h5py.File(path, **kwargs)

    @classmethod
    def _read(cls, path: str, **kwargs) -> yaw.CorrFunc:
        return yaw.CorrFunc.from_file(path)

    @classmethod
    def _write(cls, data: yaw.CorrFunc, path: str, **kwargs) -> None:
        data.to_file(path)
