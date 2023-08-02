from __future__ import annotations

import yaw
from rail.core.data import Hdf5Handle, QPHandle, TableHandle
from rail.core.stage import RailStage

from yaw_rail.config import config_to_stageparams, default_resampling


class YAWEstimator(RailStage):
    name = "YAWEstimator"
    config_options = RailStage.config_options.copy()
    config_options.update(config_to_stageparams(default_resampling))
    inputs = [
        ("cross_corr", Hdf5Handle),
        ("ref_corr", Hdf5Handle),
        ("unk_corr", Hdf5Handle),
    ]
    outputs = [("nz_estimate", QPHandle)]

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)
        # set up data management
        self.setup = yaw.ResamplingConfig.from_dict(self.config_options)

    def estimate(
        self,
        cross_corr: TableHandle,
        ref_corr: TableHandle | None = None,
        unk_corr: TableHandle | None = None,
    ) -> Hdf5Handle:
        # collect the inputs
        corrfuncs = {}
        corrfuncs["cross_corr"] = self.set_data("cross_corr", cross_corr)
        if ref_corr is not None:
            corrfuncs["ref_corr"] = self.set_data("ref_corr", ref_corr)
        if unk_corr is not None:
            corrfuncs["unk_corr"] = self.set_data("unk_corr", cross_corr)
        # run the computation
        nz_cc = yaw.RedshiftData.from_corrfuncs(**corrfuncs, config=self.setup)
        self.set_data("nz_estimate", nz_cc)
        return self.get_handle("nz_estimate")
