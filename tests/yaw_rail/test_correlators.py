from numpy.random import default_rng
from pytest import fixture
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


"""
from rail.core.data import PqHandle

        # create input files with random points
        path = str(tmp_path / f"data.pqt")
        data.to_parquet(path)
        path = str(tmp_path / f"rand.pqt")
        rand.to_parquet(path)

        # add files to datastore
        data_handle = DS.read_file("data", PqHandle, "data.pqt")
        rand_handle = DS.read_file("rand", PqHandle, "rand.pqt")
"""
