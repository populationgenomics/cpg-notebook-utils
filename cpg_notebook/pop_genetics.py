#!/usr/bin/env python3
"""
cpg_notebook.pop_genetics — install population genetics tools on CPG COS notebook VMs.

From a notebook Python cell:

    from cpg_notebook import pop_genetics as npg
    npg.install_all()

Or install selectively:

    npg.install_eigensoft()
    npg.install_flare()
    npg.install_admixture()

General genomics tools (htslib, samtools, bcftools, shapeit5, glimpse2, beagle,
king, plink2, plink19, hapibd, ibdends, fraposa) are in the companion module:

    from cpg_notebook import genomics as ng
    ng.install_all()
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from cpg_notebook.genomics import (
    BUILD_DIR,
    INSTALL_DIR,
    _apt,
    _download,
    _ensure_dirs,
    _install_archived_binary,
    _install_jar_tool,
    _link_to_system,
    _prefix,
    _run,
    _uninstall,
    _untar,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version constants — override by passing version= to individual install fns
# ---------------------------------------------------------------------------
EIGENSOFT_VERSION = '8.0.0'
FLARE_VERSION     = 'latest'   # no versioned release URL — downloads current jar from faculty page
ADMIXTURE_VERSION = '1.3.1'


# ---------------------------------------------------------------------------
# Population genetics tools
# ---------------------------------------------------------------------------

def install_eigensoft(
    version: str = EIGENSOFT_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Build and install EIGENSOFT from source. Returns the bin/ directory."""
    install_dir = install_dir or INSTALL_DIR
    prefix  = _prefix('eigensoft', version, install_dir)
    bin_dir = prefix / 'bin'

    if not force and (bin_dir / 'smartpca').exists():
        log.info('eigensoft %s already installed, re-linking', version)
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    _apt('make', 'gcc', 'g++', 'libgsl-dev', 'liblapack-dev', 'liblapacke-dev', 'libopenblas-dev')

    tarball = build_dir / f'EIG-{version}.tar.gz'
    src_dir = build_dir / f'EIG-{version}'

    if not tarball.exists():
        _download(
            f'https://github.com/DReichLab/EIG/archive/refs/tags/v{version}.tar.gz',
            tarball,
        )
    _untar(tarball, src_dir)

    # Without this patch the build fails with:
    #   undefined reference to `LAPACKE_dsyevd'  (and other LAPACKE_ symbols)
    # Two things are required together:
    #   1. liblapacke-dev must be installed (it is in the _apt() call above) —
    #      Debian/Ubuntu split lapacke into a separate package from liblapack-dev.
    #   2. The Makefile must link against it — upstream ships the correct line but
    #      leaves it commented out ("# override LDLIBS += -llapacke") with a note
    #      that it is needed on systems where lapacke is a separate library.
    #      We uncomment it here.
    makefile = src_dir / 'src' / 'Makefile'
    text = makefile.read_text()
    text = text.replace('# override LDLIBS += -llapacke', 'override LDLIBS += -llapacke')
    makefile.write_text(text)

    # make install moves binaries to ../bin relative to src/ — create it first
    (src_dir / 'bin').mkdir(parents=True, exist_ok=True)
    _run(['make', '-j4'], cwd=src_dir / 'src')
    _run(['make', 'install'], cwd=src_dir / 'src')

    # Copy from the EIG build's bin/ into our versioned prefix/bin/
    bin_dir.mkdir(parents=True, exist_ok=True)
    for f in (src_dir / 'bin').iterdir():
        if f.is_file() and os.access(f, os.X_OK):
            shutil.copy2(f, bin_dir / f.name)

    _link_to_system(bin_dir)
    log.info('eigensoft %s → %s', version, prefix)
    return bin_dir


def install_flare(
    version: str = FLARE_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download flare JAR and write a shell wrapper. Returns the bin/ directory."""
    return _install_jar_tool(
        'flare', version,
        'https://faculty.washington.edu/browning/flare.jar',
        'flare.jar',
        install_dir or INSTALL_DIR, force,
    )


def install_admixture(
    version: str = ADMIXTURE_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download ADMIXTURE pre-compiled binary (C++ by David Alexander). Returns the bin/ directory.

    Note: this is ADMIXTURE (C++ binary, dalexander/admixture), not ADAMIXTURE
    (PyTorch Python package). See cpg_notebook.genomics.install_adamixture
    for the latter.
    """
    return _install_archived_binary(
        'admixture', version,
        f'https://dalexander.github.io/admixture/binaries/admixture_linux-{version}.tar.gz',
        'admixture',
        install_dir or INSTALL_DIR, build_dir, force,
    )


# ---------------------------------------------------------------------------
# Public tool registry + install_all
# ---------------------------------------------------------------------------

TOOL_FUNCS: dict[str, callable] = {
    'eigensoft': install_eigensoft,
    'flare':     install_flare,
    'admixture': install_admixture,
}


def install_all(
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> None:
    """Install all population genetics tools."""
    install_dir = install_dir or INSTALL_DIR
    kwargs = dict(install_dir=install_dir, build_dir=build_dir, force=force)
    tools = list(TOOL_FUNCS.items())
    for i, (name, fn) in enumerate(tools, 1):
        print(f'\n{"─" * 52}\n  [{i}/{len(tools)}]  {name}\n{"─" * 52}', flush=True)
        fn(**kwargs)


# ---------------------------------------------------------------------------
# Uninstall functions
# ---------------------------------------------------------------------------

def uninstall_eigensoft(version: str = EIGENSOFT_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('eigensoft', version, install_dir or INSTALL_DIR)


def uninstall_flare(version: str = FLARE_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('flare', version, install_dir or INSTALL_DIR)


def uninstall_admixture(version: str = ADMIXTURE_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('admixture', version, install_dir or INSTALL_DIR)


UNINSTALL_FUNCS: dict[str, callable] = {
    'eigensoft': uninstall_eigensoft,
    'flare':     uninstall_flare,
    'admixture': uninstall_admixture,
}


def uninstall_all(install_dir: Path | None = None) -> None:
    """Uninstall all population genetics tools."""
    install_dir = install_dir or INSTALL_DIR
    for name, fn in UNINSTALL_FUNCS.items():
        log.info('=== uninstall %s ===', name)
        fn(install_dir=install_dir)


# Binary tools: (dir_name, version, relative_check_path)
_INSTALL_CHECK = {
    'eigensoft': ('eigensoft', EIGENSOFT_VERSION, 'bin/smartpca'),
    'flare':     ('flare',     FLARE_VERSION,     'bin/flare'),
    'admixture': ('admixture', ADMIXTURE_VERSION, 'bin/admixture'),
}


def list_tools(install_dir: Path | None = None) -> None:
    """Print all installable tools and whether each is currently installed."""
    install_dir = install_dir or INSTALL_DIR
    print(f'{"tool":<14} {"version":<20} {"status"}')
    print('-' * 46)
    for name, (tool, version, check) in sorted(_INSTALL_CHECK.items()):
        installed = (_prefix(tool, version, install_dir) / check).exists()
        status = 'installed' if installed else 'not installed'
        print(f'{name:<14} {version:<20} {status}')


