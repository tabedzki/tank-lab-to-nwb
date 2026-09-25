"""Guards for dependency pins that keep neuroconv importable.

neuroconv (through 0.10.2) calls the zarr 2 API at import time
(``zarr.codec_registry`` in ``tools/nwb_helpers/_configuration_models/_zarr_dataset_io.py``)
but does not bound zarr itself. hdmf-zarr 0.14.0 requires ``zarr>=3.4``, so an
unconstrained install resolves zarr 3 and ``import neuroconv.tools.nwb_helpers``
fails. We pin ``zarr<3`` until neuroconv supports zarr 3
(catalystneuro/neuroconv#1749; stopgap pin catalystneuro/neuroconv#2063).
"""

import importlib.metadata


def _major(version: str) -> int:
    return int(version.split(".")[0])


def test_major_parses_release_and_prerelease_versions():
    assert _major("2.18.7") == 2
    assert _major("3.0.0rc1") == 3
    assert _major("0.14.0") == 0


def test_installed_zarr_is_below_3():
    version = importlib.metadata.version("zarr")
    assert _major(version) < 3, (
        f"zarr {version} is installed; neuroconv needs zarr<3 until it supports zarr 3 "
        "(catalystneuro/neuroconv#1749)."
    )


def test_neuroconv_nwb_helpers_imports():
    import neuroconv.tools.nwb_helpers  # noqa: F401
