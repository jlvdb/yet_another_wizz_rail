import numpy.testing as npt
import yaw
from numpy.random import default_rng
from pytest import fixture
from rail.core.data import PqHandle
from rail.core.stage import RailStage
from yaw import UniformRandoms

from yaw_rail.correlators import YAWAutoCorr


@fixture
def seed():
    return 12345


@fixture
def n_draw():
    return 10_000


@fixture
def generator(seed):
    return UniformRandoms(-5, 5, -5, 5, seed=seed)


@fixture
def np_rng(seed):
    return default_rng(seed)


@fixture
def data(generator, np_rng, n_draw):
    z = np_rng.uniform(0.01, 1.0, size=n_draw)
    return generator.generate(n_draw, draw_from=dict(z=z))


@fixture
def rand(generator, np_rng, n_draw):
    z = np_rng.uniform(0.01, 1.0, size=n_draw)
    return generator.generate(n_draw, draw_from=dict(z=z))


class TestYAWCorrelatorBase:
    def test_make_stage(self):
        DS = RailStage.data_store
        YAWAutoCorr.make_stage(
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
        DS.clear()

    def test_build_catalog(self, data, tmp_path):
        DS = RailStage.data_store

        # create input files with random points
        path = str(tmp_path / "data.pq")
        data["patch"] = 0
        data.to_parquet(path)
        data.to_parquet(path + "t")

        # add files to datastore
        data_handle = DS.read_file("data", PqHandle, path)

        stage = YAWAutoCorr.make_stage(
            rmin=100.0,
            rmax=1000.0,
            zmin=0.01,
            zmax=3.0,
            zbin_num=30,
            ra_name="ra",
            dec_name="dec",
            patch_name="patch",
            redshift_name="z",
            cache_path=str(tmp_path),
        )

        assert stage.build_catalog("any_value", None) is None

        ref_cat = yaw.NewCatalog().from_file(
            path + "t",
            ra="ra",
            dec="dec",
            redshift="z",
            patches="patch",
        )

        stage_cat = stage.build_catalog("data", data_handle)

        for attr in ("ra", "dec", "redshifts", "patch"):
            npt.assert_array_equal(getattr(ref_cat, attr), getattr(stage_cat, attr))

        DS.clear()
