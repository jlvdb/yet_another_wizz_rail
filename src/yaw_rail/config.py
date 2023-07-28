from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import yaw
from ceci.config import StageParameter
from numpy.typing import ArrayLike, NDArray
from yaw.config import default as DEFAULT
from yaw.config.abc import BaseConfig
from yaw.config.utils import ConfigError, parse_section_error
from yaw.core.cosmology import TypeCosmology
from yaw.core.docs import Parameter


@dataclass(frozen=True)
class DataConfig(BaseConfig):
    """Configuration of the input data.

    Configures the columns in the input data and cache paths. The data in the
    right ascension and declincation column must be given in degrees, the patch
    assignment expects an integer index (starting from 0) that assigns each
    object to one of the spatial patches.

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
        cache_path (:obj:`str`):
            Path to directory where data is cached.
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
    cache_path: str | None = field(
        default=None, metadata=Parameter(help="path to directory where data is cached")
    )
    """Path to directory where data is cached."""

    def __post_init__(self) -> None:
        if self.cache_path is None:
            object.__setattr__(self, "cache_path", ".")

    def modify(
        self,
        ra_name: str = DEFAULT.NotSet,
        dec_name: str = DEFAULT.NotSet,
        patch_name: str = DEFAULT.NotSet,
        redshift_name: str | None = DEFAULT.NotSet,
        weight_name: str | None = DEFAULT.NotSet,
        cache_path: str | None = DEFAULT.NotSet,
    ) -> DataConfig:
        return super().modify(
            ra_name=ra_name,
            dec_name=dec_name,
            patch_name=patch_name,
            redshift_name=redshift_name,
            weight_name=weight_name,
            cache_path=cache_path,
        )

    def get_colnames(self) -> dict[str, str]:
        ignore = {"cache_path"}
        return {k: v for k, v in self.to_dict() if k not in ignore}

    def get_cache_path(self) -> pathlib.Path:
        return pathlib.Path(self.cache_path)


@dataclass(frozen=True)
class SetupConfig(BaseConfig):
    analysis: yaw.Configuration
    data: DataConfig

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
        # BinningConfig
        zmin: ArrayLike = None,
        zmax: ArrayLike = None,
        zbin_num: int | None = DEFAULT.Configuration.binning.zbin_num,
        method: str = DEFAULT.Configuration.binning.method,
        zbins: NDArray[np.float_] | None = None,
        # BackendConfig
        thread_num: int | None = DEFAULT.Configuration.backend.thread_num,
        crosspatch: bool = DEFAULT.Configuration.backend.crosspatch,
        rbin_slop: float = DEFAULT.Configuration.backend.rbin_slop,
        # DataConfig
        ra_name: str,
        dec_name: str,
        patch_name: str,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        cache_path: str | None = None,
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
        data_conf = DataConfig.create(
            ra_name=ra_name,
            dec_name=dec_name,
            patch_name=patch_name,
            redshift_name=redshift_name,
            weight_name=weight_name,
            cache_path=cache_path,
        )
        return cls(yaw_conf, data_conf)

    def modify(
        self,
        *,
        cosmology: TypeCosmology | str | None = DEFAULT.NotSet,
        # ScalesConfig
        rmin: ArrayLike | None = DEFAULT.NotSet,
        rmax: ArrayLike | None = DEFAULT.NotSet,
        rweight: float | None = DEFAULT.NotSet,
        rbin_num: int | None = DEFAULT.NotSet,
        # BinningConfig
        zmin: float | None = DEFAULT.NotSet,
        zmax: float | None = DEFAULT.NotSet,
        zbin_num: int | None = DEFAULT.NotSet,
        method: str | None = DEFAULT.NotSet,
        zbins: NDArray[np.float_] | None = DEFAULT.NotSet,
        # BackendConfig
        thread_num: int | None = DEFAULT.NotSet,
        crosspatch: bool | None = DEFAULT.NotSet,
        rbin_slop: float | None = DEFAULT.NotSet,
        # DataConfig
        ra_name: str = DEFAULT.NotSet,
        dec_name: str = DEFAULT.NotSet,
        patch_name: str = DEFAULT.NotSet,
        redshift_name: str | None = DEFAULT.NotSet,
        weight_name: str | None = DEFAULT.NotSet,
        cache_path: str | None = DEFAULT.NotSet,
    ) -> SetupConfig:
        analysis = self.analysis.modify(
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
        data = self.data.modify(
            ra_name=ra_name,
            dec_name=dec_name,
            patch_name=patch_name,
            redshift_name=redshift_name,
            weight_name=weight_name,
            cache_path=cache_path,
        )
        return self.__class__(analysis=analysis, data=data)

    @classmethod
    def from_dict(cls, the_dict: dict[str, Any], **kwargs) -> SetupConfig:
        config = {k: v for k, v in the_dict.items()}
        # remove all keys that are not part of yaw.Configuration
        try:
            analysis_dict = config.pop("analysis")
            analysis = yaw.Configuration.from_dict(analysis_dict)
        except (TypeError, KeyError) as e:
            parse_section_error(e, "analysis")
        try:
            data_dict = config.pop("data")
            data = yaw.Configuration.from_dict(data_dict)
        except (TypeError, KeyError) as e:
            parse_section_error(e, "data")
        # check that there are no entries left
        if len(config) > 0:
            key = next(iter(config.keys()))
            raise ConfigError(f"encountered unknown section '{key}'")
        return cls(analysis=analysis, data=data)


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
