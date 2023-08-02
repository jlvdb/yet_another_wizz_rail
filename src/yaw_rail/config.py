from __future__ import annotations

import pathlib
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import numpy as np
from ceci.config import StageParameter
from numpy.typing import ArrayLike, NDArray
from yaw.config import Configuration, ResamplingConfig
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

    ra_name: str = field(
        metadata=Parameter(type=str, required=True, help="right ascension column name")
    )
    """Right ascension column name."""
    dec_name: str = field(
        metadata=Parameter(type=str, required=True, help="declination column name")
    )
    """Declination column name."""
    patch_name: str = field(
        metadata=Parameter(type=str, help="patch index column name")
    )
    """Patch index column name."""
    redshift_name: str | None = field(
        default=None, metadata=Parameter(type=str, help="redshifts column name")
    )
    """Redshifts column name."""
    weight_name: str | None = field(
        default=None, metadata=Parameter(type=str, help="weights column name")
    )
    """Weights column name."""
    cache_path: str | None = field(
        default=None,
        metadata=Parameter(type=str, help="path to directory where data is cached"),
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
        return {k: v for k, v in self.to_dict().items() if k not in ignore}

    def get_cache_path(self) -> pathlib.Path:
        return pathlib.Path(self.cache_path)


@dataclass(frozen=True)
class SetupConfig(BaseConfig):
    analysis: Configuration
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
        yaw_conf = Configuration.create(
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
        # remove all keys that are not part of Configuration
        try:
            analysis_dict = config.pop("analysis")
            analysis = Configuration.from_dict(analysis_dict)
        except (TypeError, KeyError) as e:
            parse_section_error(e, "analysis")
        try:
            data_dict = config.pop("data")
            data = Configuration.from_dict(data_dict)
        except (TypeError, KeyError) as e:
            parse_section_error(e, "data")
        # check that there are no entries left
        if len(config) > 0:
            key = next(iter(config.keys()))
            raise ConfigError(f"encountered unknown section '{key}'")
        return cls(analysis=analysis, data=data)


def config_to_stageparams(config: BaseConfig) -> dict[str, StageParameter]:
    if not isinstance(config, BaseConfig):
        raise TypeError("'config' must be an instance of 'BaseConfig'")
    stageparams = dict()
    for cfield in fields(config):
        try:
            parameter = Parameter.from_field(cfield)
            kwargs = dict(
                dtype=parameter.type,
                required=parameter.required,
                msg=parameter.help,
            )
            if not isinstance(cfield.default, type(MISSING)):
                kwargs["default"] = cfield.default
            stageparams[cfield.name] = StageParameter(**kwargs)
        except TypeError:
            attr = getattr(config, cfield.name)
            try:
                stageparams.update(config_to_stageparams(attr))
            except TypeError:
                continue
    return stageparams


default_setup = SetupConfig.create(
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
default_resampling = ResamplingConfig.create()
