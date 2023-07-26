from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import yaw
from ceci.config import StageParameter
from numpy.typing import ArrayLike, NDArray
from yaw.config import default as DEFAULT
from yaw.config.utils import parse_section_error
from yaw.core.abc import DictRepresentation
from yaw.core.cosmology import TypeCosmology
from yaw.core.docs import Parameter


@dataclass(frozen=True)
class ColumnsConfig(DictRepresentation):
    """Configuration column names in the input data.

    The data in the right ascension and declincation column must be given in
    degrees, the patch assignment expects an integer index (starting from 0)
    that assigns each object to one of the spatial patches.

    Args:
        ra_name (:obj:`str`):
            Right ascension column name.
        dec_name (:obj:`str`):
            Declination column name.
        patch_name (:obj:`str`):
            Patch index column name.
        redshift_name (:obj:`str`):
            Redshifts column name.
        weight_name (:obj:`str`):
            Weights column name.
    """

    ra_name: str = field(metadata=Parameter(help="right ascension column name"))
    """Right ascension column name."""
    dec_name: str = field(metadata=Parameter(help="declination column name"))
    """Declination column name."""
    patch_name: str = field(metadata=Parameter(help="patch index column name"))
    """Patch index column name."""
    redshift_name: str | None = field(
        default=None, metadata=Parameter(help="redshifts column name")
    )
    """Redshifts column name."""
    weight_name: str | None = field(
        default=None, metadata=Parameter(help="weights column name")
    )
    """Weights column name."""


@dataclass(frozen=True)
class SetupConfig(yaw.Configuration):
    columns: ColumnsConfig
    cache_path: str | None = field(
        default=None, metadata=Parameter(help="path to directory where data is cached")
    )

    def get_cache_path(self) -> pathlib.Path:
        # TODO: resolve cache path
        return pathlib.Path()

    @classmethod
    def from_yaw_config(
        cls,
        yaw_config: yaw.Configuration,
        columns: ColumnsConfig,
        cache_path: str | None = None,
    ) -> SetupConfig:
        return cls(
            scales=yaw_config.scales,
            binning=yaw_config.binning,
            backend=yaw_config.backend,
            cosmology=yaw_config.cosmology,
            columns=columns,
            cache_path=cache_path,
        )

    def get_yaw_config(self) -> yaw.Configuration:
        return yaw.Configuration(
            scales=self.scales,
            binning=self.binning,
            backend=self.backend,
            cosmology=self.cosmology,
        )

    @classmethod
    def create(
        cls,
        *,
        cosmology: TypeCosmology | str | None = DEFAULT.Configuration.cosmology,
        # ScalesConfig
        rmin: ArrayLike,
        rmax: ArrayLike,
        rweight: float | None = DEFAULT.Configuration.scales.rweight,
        rbin_num: int = DEFAULT.Configuration.scales.rbin_num,
        # AutoBinningConfig / ManualBinningConfig
        zmin: ArrayLike = None,
        zmax: ArrayLike = None,
        zbin_num: int | None = DEFAULT.Configuration.binning.zbin_num,
        method: str = DEFAULT.Configuration.binning.method,
        zbins: NDArray[np.float_] | None = None,
        # BackendConfig
        thread_num: int | None = DEFAULT.Configuration.backend.thread_num,
        crosspatch: bool = DEFAULT.Configuration.backend.crosspatch,
        rbin_slop: float = DEFAULT.Configuration.backend.rbin_slop,
        # ColumnsConfig
        ra_name: str,
        dec_name: str,
        patch_name: str,
        redshift_name: str | None,
        weight_name: str | None,
        # extra parameters
        cache_path: str | None,
    ) -> SetupConfig:
        yaw_conf = yaw.Configuration.create(
            cosmology=cosmology,
            rmin=rmin,
            rmax=rmax,
            rweight=rweight,
            rbin_num=rbin_num,
            zmin=zmin,
            zmax=zmax,
            zbin_num=zbin_num,
            method=method,
            zbins=zbins,
            thread_num=thread_num,
            crosspatch=crosspatch,
            rbin_slop=rbin_slop,
        )
        return cls.from_yaw_config(
            yaw_conf,
            ColumnsConfig(
                ra_name=ra_name,
                dec_name=dec_name,
                patch_name=patch_name,
                redshift_name=redshift_name,
                weight_name=weight_name,
            ),
            cache_path,
        )

    def modify(
        self,
        *,
        cosmology: TypeCosmology | str | None = DEFAULT.NotSet,
        # ScalesConfig
        rmin: ArrayLike | None = DEFAULT.NotSet,
        rmax: ArrayLike | None = DEFAULT.NotSet,
        rweight: float | None = DEFAULT.NotSet,
        rbin_num: int | None = DEFAULT.NotSet,
        # AutoBinningConfig / ManualBinningConfig
        zmin: float | None = DEFAULT.NotSet,
        zmax: float | None = DEFAULT.NotSet,
        zbin_num: int | None = DEFAULT.NotSet,
        method: str | None = DEFAULT.NotSet,
        zbins: NDArray[np.float_] | None = DEFAULT.NotSet,
        # BackendConfig
        thread_num: int | None = DEFAULT.NotSet,
        crosspatch: bool | None = DEFAULT.NotSet,
        rbin_slop: float | None = DEFAULT.NotSet,
        # ColumnsConfig
        ra_name: str = DEFAULT.NotSet,
        dec_name: str = DEFAULT.NotSet,
        patch_name: str = DEFAULT.NotSet,
        redshift_name: str | None = DEFAULT.NotSet,
        weight_name: str | None = DEFAULT.NotSet,
        # extra parameters
        cache_path: str | None = DEFAULT.NotSet,
    ) -> SetupConfig:
        yaw_config = self.get_yaw_config().modify(
            cosmology=cosmology,
            rmin=rmin,
            rmax=rmax,
            rweight=rweight,
            rbin_num=rbin_num,
            zmin=zmin,
            zmax=zmax,
            zbin_num=zbin_num,
            method=method,
            zbins=zbins,
            thread_num=thread_num,
            crosspatch=crosspatch,
            rbin_slop=rbin_slop,
        )
        conf_dict = yaw_config.to_dict()
        # ColumnsConfig
        conf_dict["columns"] = self.columns.to_dict()
        if ra_name is not DEFAULT.NotSet:
            conf_dict["columns"]["ra_name"] = ra_name
        if dec_name is not DEFAULT.NotSet:
            conf_dict["columns"]["dec_name"] = dec_name
        if patch_name is not DEFAULT.NotSet:
            conf_dict["columns"]["patch_name"] = patch_name
        if redshift_name is not DEFAULT.NotSet:
            conf_dict["columns"]["redshift_name"] = redshift_name
        if weight_name is not DEFAULT.NotSet:
            conf_dict["columns"]["weight_name"] = weight_name
        # extra parameters
        if cache_path is not DEFAULT.NotSet:
            conf_dict["cache_path"] = cache_path
        return self.__class__.from_dict(conf_dict)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> SetupConfig:
        config = {k: v for k, v in the_dict.items()}
        # remove all keys that are not part of yaw.Configuration
        try:
            columns = ColumnsConfig.from_dict(config.pop("columns"))
        except (TypeError, KeyError) as e:
            parse_section_error(e, "columns")
        cache_path = the_dict.get("cache_path")
        # parse remaining config
        yaw_conf = yaw.Configuration.from_dict(config)
        return cls.from_yaw_config(yaw_conf, columns, cache_path)


def config_to_stageparams(config: Any) -> dict[str, StageParameter]:
    flat_dict = dict()
    for dfield in fields(config):
        try:
            flat_dict.update(config_to_stageparams(dfield.type))
        except TypeError:
            meta = dfield.metadata
            if not isinstance(meta, Parameter):
                raise TypeError(f"missing metadata for parameter '{dfield.name}'")
            meta: Parameter
            flat_dict[dfield.name] = StageParameter(
                default=dfield.default,
                dtype=meta.type,
                required=meta.required,
                msg=meta.help,
            )
    return flat_dict


default_config = SetupConfig.create(
    rmin=100.0,
    rmax=1000.0,
    zmin=0.01,
    zmax=3.0,
    zbin_num=30,
    ra_name="ra",
    dec_name="dec",
    patch_name="patch",
    redshift_name="redshift",
    weight_name="weight",
)
