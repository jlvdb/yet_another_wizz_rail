from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaw
from rail.core.data import DataHandle, ModelHandle, TableHandle
from rail.core.stage import RailStage
from yaw.catalogs import BaseCatalog, PatchLinkage

from yaw_rail.config import SetupConfig, config_to_stageparams, default_setup
from yaw_rail.data import CorrFuncHandle

# TODO: choose good default cache location


class YAWCorrelatorBase(ABC, RailStage):
    name = "YAWCorrelatorBase"
    config_options = RailStage.config_options.copy()
    config_options.update(config_to_stageparams(default_setup))
    outputs = [
        ("corrfunc", CorrFuncHandle),
        ("linkage", ModelHandle),
    ]

    @property
    @abstractmethod
    def inputs(self) -> list[tuple[str, DataHandle]]:
        pass

    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)
        # set up data management
        self.factory = yaw.NewCatalog()  # TODO: implement backend options
        params = set(self.config_options) - set(RailStage.config_options)
        kwargs = {
            param: value for param, value in self.config.items() if param in params
        }
        self.setup = SetupConfig.create(**kwargs)
        self._caches: set[Path] = set()

    def build_linkage(self, catalogs: Iterable[BaseCatalog]) -> PatchLinkage:
        cats = list(catalogs)
        longest = cats[0]
        for cat in cats[1:]:
            if len(cat) > len(longest):
                longest = cat
        return PatchLinkage.from_setup(self.setup.analysis, longest)

    def build_catalog(self, tag: str, handle: TableHandle | None) -> BaseCatalog | None:
        if handle is None:
            return None
        # put the data in the datastore
        data = self.set_data(tag, handle)
        if not isinstance(data, pd.DataFrame):
            data: pd.DataFrame = data.to_pandas()
        # build the catalog
        cache_directory = self.setup.data.get_cache_path() / f"cache_{tag}"
        cache_directory.mkdir()
        self._caches.add(cache_directory)
        return self.factory.from_dataframe(
            data, **self.setup.data.get_colnames(), cache_directory=str(cache_directory)
        )

    def finalize(
        self, corrfunc: yaw.CorrFunc | dict[str, yaw.CorrFunc], linkage: PatchLinkage
    ) -> None:
        try:
            for scale, cf in corrfunc.items():
                self.set_data(f"corrfunc@{scale}", cf)
        except AttributeError:
            self.set_data("corrfunc", corrfunc)
        self.set_data("linkage", linkage)

    def drop_cache(self) -> None:
        while len(self._caches) > 0:
            shutil.rmtree(self._caches.pop())

    @abstractmethod
    def correlate(
        self,
        linkage: ModelHandle | None = None,
        **samples: TableHandle | None,
    ) -> tuple[CorrFuncHandle, ModelHandle]:
        pass


class YAWCrossCorr(YAWCorrelatorBase):
    name = "YAWCrossCorr"
    inputs = [
        ("reference", TableHandle),
        ("ref_rand", TableHandle),
        ("unknown", TableHandle),
        ("unk_rand", TableHandle),
        ("linkage", ModelHandle),
    ]

    def correlate(
        self,
        reference: TableHandle,
        unknown: TableHandle,
        ref_rand: TableHandle | None = None,
        unk_rand: TableHandle | None = None,
        linkage: ModelHandle | None = None,
    ) -> tuple[CorrFuncHandle, ModelHandle]:
        try:
            # collect the inputs and build catalog instances
            catalogs = dict(
                reference=self.build_catalog("reference", reference),
                unknown=self.build_catalog("unknown", unknown),
                ref_rand=self.build_catalog("ref_rand", ref_rand),
                unk_rand=self.build_catalog("unk_rand", unk_rand),
            )
            if linkage is None:
                links = self.build_linkage(catalogs.values())
            else:
                links = self.set_data("linkage", linkage)
            # run the computation
            corrfunc = yaw.crosscorrelate(
                self.setup.analysis, linkage=links, **catalogs
            )
        finally:
            self.drop_cache()
        # retrieve the results
        self.finalize(corrfunc, links)
        return self.get_handle("corrfunc"), self.get_handle("linkage")


class YAWAutoCorr(YAWCorrelatorBase):
    name = "YAWAutoCorr"
    inputs = [
        ("reference", TableHandle),
        ("random", TableHandle),
        ("linkage", ModelHandle),
    ]

    def correlate(
        self,
        reference: TableHandle,
        random: TableHandle,
        linkage: ModelHandle | None = None,
    ) -> tuple[CorrFuncHandle, ModelHandle]:
        try:
            # collect the inputs and build catalog instances
            catalogs = dict(
                reference=self.build_catalog(reference),
                unkown=self.build_catalog(random),
            )
            if linkage is None:
                links = self.build_linkage(catalogs.values())
            else:
                links = self.set_data("linkage", linkage)
            # run the computation
            corrfunc = yaw.autocorrelate(self.setup.analysis, linkage=links, **catalogs)
        finally:
            self.drop_cache()
        # retrieve the results
        self.finalize(corrfunc, links)
        return self.get_handle("corrfunc"), self.get_handle("linkage")
