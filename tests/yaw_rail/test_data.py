from pytest import fixture
from rail.core.stage import RailStage

from yaw_rail.correlators import YAWCrossCorr
from yaw_rail.data import CorrFuncHandle


@fixture
def stage():
    return YAWCrossCorr.make_stage(
        rmin=100.0,
        rmax=1000.0,
        zmin=0.01,
        zmax=3.0,
        zbin_num=30,
        ra_name="ra",
        dec_name="dec",
        patch_name="patch",
        redshift_name="z",
    )


@fixture
def corrfunc():
    from yaw.examples import w_ss

    return w_ss


class TestCorrFuncHandle:
    def test_io(self, stage, corrfunc, tmp_path):
        DS = RailStage.data_store

        # put into datastore
        stage.finalize(corrfunc, linkage=dict(dummy=None))
        # fetch from data store
        handle: CorrFuncHandle = DS.get("corrfunc")
        # write to file
        handle.path = str(tmp_path / "corrfunc.hdf5")
        handle.write()
        # check that written file can be loaded and is identical to the input
        assert handle.data == corrfunc

        DS.clear()
