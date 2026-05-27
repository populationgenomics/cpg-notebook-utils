#!/usr/bin/env python3
"""
cpg_notebook.genomics — install bioinformatics tools on CPG COS notebook VMs.

From a notebook Python cell:

    from cpg_notebook import genomics
    genomics.install_all()

Or install selectively:

    genomics.install_htslib()
    genomics.install_samtools()
    genomics.install_bcftools()

Population genetics tools (EIGENSOFT, FLARE, ADMIXTURE) are in the companion module:

    from cpg_notebook import pop_genetics as npg
    npg.install_all()

BCFTOOLS_PLUGINS caveat: /etc/environment is not re-read by %%bash subprocesses
mid-session. After setup_path() runs, either:
    - Add 'export BCFTOOLS_PLUGINS=...' to ~/.bashrc once, or
    - Set it inline in any %%bash cell that uses plugins:
        export BCFTOOLS_PLUGINS=/content/tools/bcftools-1.23.1/lib/bcftools
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Version constants — override by passing version= to individual install fns
# ---------------------------------------------------------------------------
HTSLIB_VERSION    = '1.23.1'
SAMTOOLS_VERSION  = '1.23.1'
BCFTOOLS_VERSION  = '1.23.1'
SHAPEIT5_VERSION  = '5.1.1'
BEAGLE_VERSION    = '27Feb25.75f'
GLIMPSE2_VERSION  = '2.0.0'
KING_VERSION      = '2.3.2'
PLINK2_VERSION    = '20260311'
PLINK19_VERSION   = '20250819'
HAPIBD_VERSION    = 'latest'   # no versioned release URL — downloads current jar from faculty page
IBDENDS_VERSION   = 'latest'   # no pinned release — downloads current jar from faculty page

# /content is local SSD on COS — writable, fast, 2.9T, but does not persist across VM shutdown
WORKDIR     = Path(os.environ.get('WORKDIR', '/content'))
INSTALL_DIR = WORKDIR / 'tools'
BUILD_DIR   = Path('/tmp/build')

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, cwd: Path | None = None, silent: bool = False) -> str:
    """Run a command; stream stdout to terminal unless silent=True. Always captures stderr."""
    log.debug('+ %s', ' '.join(str(c) for c in cmd))
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE if silent else None,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return (result.stdout or '').strip()
    except subprocess.CalledProcessError as e:
        if e.stdout:
            log.error('stdout from %s:\n%s', cmd[0], e.stdout.strip())
        if e.stderr:
            log.error('stderr from %s:\n%s', cmd[0], e.stderr.strip())
        raise


def _apt(*packages: str) -> None:
    """Install apt packages, skipping any already installed."""
    missing = [
        p for p in packages
        if subprocess.run(['dpkg', '-s', p], capture_output=True).returncode != 0
    ]
    if missing:
        log.info('apt-get update then install %s', ' '.join(missing))
        _run(['sudo', 'apt-get', 'update', '-qq'])
        _run(['sudo', 'apt-get', 'install', '-y', '--no-install-recommends', *missing])


def _prefix(tool: str, version: str, install_dir: Path) -> Path:
    return install_dir / f'{tool}-{version}'


def _ensure_dirs() -> None:
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)


def _link_to_system(bin_dir: Path) -> None:
    """Symlink all executables in bin_dir into /usr/local/bin."""
    for exe in bin_dir.iterdir():
        if exe.is_file() and os.access(exe, os.X_OK):
            _run(['sudo', 'ln', '-sf', str(exe), f'/usr/local/bin/{exe.name}'])
            log.info('  linked %s → /usr/local/bin/', exe.name)


def _register_ldconfig(lib_dir: Path) -> None:
    """Register a library directory with the dynamic linker."""
    conf = '/etc/ld.so.conf.d/cpg-tools.conf'
    _run(['sudo', 'bash', '-c', f"echo '{lib_dir}' >> {conf}"])
    _run(['sudo', 'ldconfig'])


def _unlink_from_system(bin_dir: Path) -> None:
    """Remove /usr/local/bin symlinks that point into bin_dir."""
    if not bin_dir.exists():
        return
    for exe in bin_dir.iterdir():
        link = Path('/usr/local/bin') / exe.name
        if link.is_symlink() and Path(os.readlink(link)) == exe:
            _run(['sudo', 'rm', '-f', str(link)])
            log.info('  removed /usr/local/bin/%s', exe.name)


def _deregister_ldconfig() -> None:
    """Remove the cpg-tools ldconfig conf and refresh the linker cache."""
    conf = Path('/etc/ld.so.conf.d/cpg-tools.conf')
    if conf.exists():
        _run(['sudo', 'rm', '-f', str(conf)])
        _run(['sudo', 'ldconfig'])


def _write_jar_wrapper(jar: Path, wrapper: Path) -> None:
    """Write an exec-style shell wrapper for a JAR."""
    wrapper.write_text(f'#!/bin/sh\nexec java -jar "{jar}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info('downloading %s', url)

    def _progress(n, block, total):
        if total > 0:
            print(f'\r  {min(n * block * 100 // total, 100)}%', end='', flush=True)
        else:
            # Server did not send Content-Length (chunked transfer); show MB instead
            print(f'\r  {n * block / 1_048_576:.1f} MB', end='', flush=True)

    urllib.request.urlretrieve(url, str(dest), _progress)
    print()


def _untar(tarball: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    with tarfile.open(tarball) as tf:
        tf.extractall(tarball.parent)


def _unzip(zippath: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath) as zf:
        zf.extractall(dest_dir)


def _find_binary(root: Path, name: str) -> Path:
    """Search root recursively for a file named name.

    Does not check execute permission — zip archives don't preserve it,
    and callers set chmod themselves after copying.
    """
    for candidate in [root / name, *root.rglob(name)]:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f'{name} binary not found under {root}')


def _pip_installed(package: str) -> bool:
    return subprocess.run(['pip', 'show', package], capture_output=True).returncode == 0


def _run_pip(cmd: list) -> None:
    """Run a pip command, capturing combined stdout+stderr and printing it.

    Unlike _run, this always captures output so that pip's detailed error
    messages (which go to stdout, not stderr) are visible on failure.
    """
    log.debug('+ %s', ' '.join(str(c) for c in cmd))
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        if result.stdout:
            print(result.stdout, end='', flush=True)
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout, end='', flush=True)
        raise


def _pip_uninstall(package: str) -> None:
    if _pip_installed(package):
        _run_pip(['pip', 'uninstall', '-y', package])
        log.info('%s removed', package)
    else:
        log.info('%s is not installed', package)


def _install_jar_tool(
    name: str,
    version: str,
    url: str,
    jar_filename: str,
    install_dir: Path,
    force: bool,
    wrapper_name: str | None = None,
) -> Path:
    """Download a JAR and write a shell wrapper. Returns the bin/ directory."""
    wrapper_name = wrapper_name or name
    prefix   = _prefix(name, version, install_dir)
    bin_dir  = prefix / 'bin'
    jars_dir = prefix / 'jars'
    jar      = jars_dir / jar_filename
    wrapper  = bin_dir / wrapper_name

    if not force and wrapper.exists():
        log.info('%s %s already installed, re-linking', name, version)
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    jars_dir.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)

    _download(url, jar)
    _write_jar_wrapper(jar, wrapper)
    _link_to_system(bin_dir)
    log.info('%s %s → %s', name, version, prefix)
    return bin_dir


def _install_archived_binary(
    name: str,
    version: str,
    archive_url: str,
    binary_name: str,
    install_dir: Path,
    build_dir: Path,
    force: bool,
    dest_name: str | None = None,
) -> Path:
    """Download a .tar.gz or .zip archive, find binary_name, install to prefix/bin."""
    dest_name = dest_name or binary_name
    prefix  = _prefix(name, version, install_dir)
    bin_dir = prefix / 'bin'

    if not force and (bin_dir / dest_name).exists():
        log.info('%s %s already installed, re-linking', name, version)
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    bin_dir.mkdir(parents=True, exist_ok=True)

    ext = '.tar.gz' if '.tar.gz' in archive_url else '.zip'
    archive = build_dir / f'{name}-{version}{ext}'
    if not archive.exists():
        _download(archive_url, archive)

    tmp = build_dir / f'{name}-{version}-extracted'
    if ext == '.zip':
        _unzip(archive, tmp)
    else:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(tmp)

    src = _find_binary(tmp, binary_name)
    dest = bin_dir / dest_name
    shutil.copy2(src, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    _link_to_system(bin_dir)
    log.info('%s %s → %s', name, version, prefix)
    return bin_dir


# ---------------------------------------------------------------------------
# Source-compiled tools
# ---------------------------------------------------------------------------

def install_htslib(
    version: str = HTSLIB_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Build and install htslib from source. Returns the bin/ directory."""
    install_dir = install_dir or INSTALL_DIR
    prefix  = _prefix('htslib', version, install_dir)
    bin_dir = prefix / 'bin'

    if not force and (bin_dir / 'bgzip').exists():
        log.info('htslib %s already installed, re-linking', version)
        _register_ldconfig(prefix / 'lib')
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    _apt('autoconf', 'make', 'gcc', 'libcurl4-openssl-dev',
         'libbz2-dev', 'liblzma-dev', 'zlib1g-dev', 'libssl-dev')

    tarball = build_dir / f'htslib-{version}.tar.bz2'
    src_dir = build_dir / f'htslib-{version}'

    if not tarball.exists():
        _download(
            f'https://github.com/samtools/htslib/releases/download/{version}/htslib-{version}.tar.bz2',
            tarball,
        )
    _untar(tarball, src_dir)

    _run(['./configure', f'--prefix={prefix}',
          '--enable-libcurl', '--enable-gcs', '--enable-plugins'], cwd=src_dir)
    _run(['make', '-j4'], cwd=src_dir)
    _run(['make', 'install'], cwd=src_dir)

    _register_ldconfig(prefix / 'lib')
    _link_to_system(bin_dir)
    log.info('htslib %s → %s', version, prefix)
    return bin_dir


def install_samtools(
    version: str = SAMTOOLS_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
    htslib_version: str = HTSLIB_VERSION,
) -> Path:
    """Build and install samtools from source. Returns the bin/ directory."""
    install_dir = install_dir or INSTALL_DIR
    prefix  = _prefix('samtools', version, install_dir)
    bin_dir = prefix / 'bin'

    if not force and (bin_dir / 'samtools').exists():
        log.info('samtools %s already installed, re-linking', version)
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    htslib_prefix = _prefix('htslib', htslib_version, install_dir)
    if not (htslib_prefix / 'bin' / 'bgzip').exists():
        install_htslib(version=htslib_version, install_dir=install_dir,
                       build_dir=build_dir, force=force)

    _apt('autoconf', 'make', 'gcc', 'libcurl4-openssl-dev',
         'libbz2-dev', 'liblzma-dev', 'zlib1g-dev', 'libssl-dev')

    tarball = build_dir / f'samtools-{version}.tar.bz2'
    src_dir = build_dir / f'samtools-{version}'

    if not tarball.exists():
        _download(
            f'https://github.com/samtools/samtools/releases/download/{version}/samtools-{version}.tar.bz2',
            tarball,
        )
    _untar(tarball, src_dir)

    _run(['./configure', f'--prefix={prefix}', f'--with-htslib={htslib_prefix}'], cwd=src_dir)
    _run(['make', '-j4'], cwd=src_dir)
    _run(['make', 'install'], cwd=src_dir)

    _link_to_system(bin_dir)
    log.info('samtools %s → %s', version, prefix)
    return bin_dir


def install_bcftools(
    version: str = BCFTOOLS_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
    htslib_version: str = HTSLIB_VERSION,
) -> Path:
    """Build and install bcftools + gtc2vcf and score plugins from source.

    Returns the bin/ directory.
    """
    install_dir = install_dir or INSTALL_DIR
    prefix      = _prefix('bcftools', version, install_dir)
    bin_dir     = prefix / 'bin'
    plugins_dir = prefix / 'lib' / 'bcftools'

    gtc_names   = {'idat2gtc', 'gtc2vcf', 'affy2vcf', 'BAFregress'}
    score_names = {'munge', 'liftover', 'score', 'metal', 'blup', 'pgs'}

    all_plugins_present = all(
        (plugins_dir / f'{n}.so').exists() for n in gtc_names | score_names
    )
    if not force and (bin_dir / 'bcftools').exists() and all_plugins_present:
        log.info('bcftools %s already installed, re-linking', version)
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    htslib_prefix = _prefix('htslib', htslib_version, install_dir)
    if not (htslib_prefix / 'bin' / 'bgzip').exists():
        install_htslib(version=htslib_version, install_dir=install_dir,
                       build_dir=build_dir, force=force)

    _apt('autoconf', 'make', 'gcc', 'git', 'libcurl4-openssl-dev',
         'libbz2-dev', 'liblzma-dev', 'zlib1g-dev', 'libssl-dev')

    tarball = build_dir / f'bcftools-{version}.tar.bz2'
    src_dir = build_dir / f'bcftools-{version}'

    if not tarball.exists():
        _download(
            f'https://github.com/samtools/bcftools/releases/download/{version}/bcftools-{version}.tar.bz2',
            tarball,
        )

    # ── bcftools binary ────────────────────────────────────────────────────────
    if force or not (bin_dir / 'bcftools').exists():
        _untar(tarball, src_dir)
        _run(['./configure', f'--prefix={prefix}', f'--with-htslib={htslib_prefix}'], cwd=src_dir)
        _run(['make', '-j4'], cwd=src_dir)
        _run(['make', 'install'], cwd=src_dir)
        # Copy ALL standard compiled plugins
        plugins_dir.mkdir(parents=True, exist_ok=True)
        for so in (src_dir / 'plugins').glob('*.so'):
            shutil.copy2(so, plugins_dir / so.name)
        log.info('copied standard bcftools plugins to %s', plugins_dir)
    else:
        log.info('bcftools %s binary already present', version)

    # ── ensure bcftools source dir is available for plugin compilation ─────────
    # BUILD_DIR (/tmp/build) is volatile and cleared on container restart, while
    # INSTALL_DIR (/content/tools) persists. Re-extract + configure the source so
    # we can compile missing plugins without rebuilding the full binary.
    plugins_dir.mkdir(parents=True, exist_ok=True)
    if not (src_dir / 'Makefile').exists():
        _untar(tarball, src_dir)
        _run(['./configure', f'--prefix={prefix}', f'--with-htslib={htslib_prefix}'], cwd=src_dir)

    bcf_plugins_src = src_dir / 'plugins'

    # ── gtc2vcf plugins (idat2gtc, gtc2vcf, affy2vcf, BAFregress) ─────────────
    # Plugins include "bcftools.h" from the source tree (not installed to prefix),
    # so they must be compiled within the bcftools source tree via make.
    gtc_missing = force or any(not (plugins_dir / f'{n}.so').exists() for n in gtc_names)
    if gtc_missing:
        gtc2vcf_tarball = build_dir / 'gtc2vcf.tar.gz'
        gtc2vcf_stage   = build_dir / 'gtc2vcf-master'
        _download(
            'https://github.com/freeseek/gtc2vcf/archive/refs/heads/master.tar.gz',
            gtc2vcf_tarball,
        )
        _untar(gtc2vcf_tarball, gtc2vcf_stage)
        for f in gtc2vcf_stage.iterdir():
            if f.suffix in ('.c', '.h'):
                shutil.copy2(f, bcf_plugins_src / f.name)
        _run(['make', '-j4'], cwd=src_dir)
        n_built = 0
        for name in gtc_names:
            so = bcf_plugins_src / f'{name}.so'
            if so.exists():
                shutil.copy2(so, plugins_dir / so.name)
                n_built += 1
        log.info('gtc2vcf: %d/%d plugins → %s', n_built, len(gtc_names), plugins_dir)

    # ── score plugins (munge, liftover, score, metal, blup, pgs) ──────────────
    # The pgs plugin links against CHOLMOD (SuiteSparse). Debian/Ubuntu install
    # the CHOLMOD header under suitesparse/, but the code expects cholmod.h on
    # the plain include path — create a shim that re-exports it with corrected
    # #include paths. Two extra SuiteSparse headers are also downloaded into the
    # bcftools source root because pgs.mk references them from there.
    score_missing = force or any(not (plugins_dir / f'{n}.so').exists() for n in score_names)
    if score_missing:
        _apt('libsuitesparse-dev')
        cholmod_shim = Path('/usr/include/cholmod.h')
        if not cholmod_shim.exists():
            _run(['sudo', 'bash', '-c',
                  "sed "
                  "'s|^#include \"cholmod_|#include \"suitesparse/cholmod_|;"
                  "s|^#include \"SuiteSparse_|#include \"suitesparse/SuiteSparse_|' "
                  "/usr/include/suitesparse/cholmod.h | sudo tee /usr/include/cholmod.h"])
        # SuiteSparse headers expected in the bcftools source root by pgs.mk
        for gh_path, dest_name in [
            ('SuiteSparse_config/SuiteSparse_config.h', 'SuiteSparse_config.h'),
            ('CHOLMOD/Include/cholmod.h', 'cholmod.h'),
        ]:
            _download(
                f'https://raw.githubusercontent.com/DrTimothyAldenDavis/SuiteSparse/stable/{gh_path}',
                src_dir / dest_name,
            )
        # score source files + pgs.mk (pgs.mk is included by the bcftools Makefile
        # via -include plugins/*.mk and adds the CHOLMOD link flags for pgs.so)
        for fname in ['score.c', 'score.h', 'munge.c', 'liftover.c',
                      'metal.c', 'blup.c', 'pgs.c', 'pgs.mk']:
            _download(
                f'https://raw.githubusercontent.com/freeseek/score/master/{fname}',
                bcf_plugins_src / fname,
            )
        _run(['make', '-j4'], cwd=src_dir)
        n_built = 0
        for name in score_names:
            so = bcf_plugins_src / f'{name}.so'
            if so.exists():
                shutil.copy2(so, plugins_dir / so.name)
                n_built += 1
        log.info('score: %d/%d plugins → %s', n_built, len(score_names), plugins_dir)

    _link_to_system(bin_dir)
    log.info('bcftools %s → %s', version, prefix)
    return bin_dir


# ---------------------------------------------------------------------------
# Static binary tools
# ---------------------------------------------------------------------------

def install_shapeit5(
    version: str = SHAPEIT5_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download shapeit5 static binaries. Returns the bin/ directory."""
    install_dir = install_dir or INSTALL_DIR
    prefix  = _prefix('shapeit5', version, install_dir)
    bin_dir = prefix / 'bin'

    if not force and (bin_dir / 'phase_common').exists():
        log.info('shapeit5 %s already installed, re-linking', version)
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    bin_dir.mkdir(parents=True, exist_ok=True)

    base = f'https://github.com/odelaneau/shapeit5/releases/download/v{version}'
    for name in ['phase_common', 'phase_rare', 'ligate', 'switch', 'simulate', 'xcftools']:
        dest = bin_dir / name
        _download(f'{base}/{name}_static', dest)
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    _link_to_system(bin_dir)
    log.info('shapeit5 %s → %s', version, prefix)
    return bin_dir


def install_glimpse2(
    version: str = GLIMPSE2_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download GLIMPSE2 static binaries. Returns the bin/ directory."""
    install_dir = install_dir or INSTALL_DIR
    prefix  = _prefix('glimpse2', version, install_dir)
    bin_dir = prefix / 'bin'

    if not force and (bin_dir / 'GLIMPSE2_phase').exists():
        log.info('glimpse2 %s already installed, re-linking', version)
        _link_to_system(bin_dir)
        return bin_dir

    _ensure_dirs()
    bin_dir.mkdir(parents=True, exist_ok=True)

    base = f'https://github.com/odelaneau/GLIMPSE/releases/download/v{version}'
    for name in [
        'GLIMPSE2_phase',
        'GLIMPSE2_chunk',
        'GLIMPSE2_ligate',
        'GLIMPSE2_concordance',
        'GLIMPSE2_split_reference',
    ]:
        dest = bin_dir / name
        _download(f'{base}/{name}_static', dest)
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    _link_to_system(bin_dir)
    log.info('glimpse2 %s → %s', version, prefix)
    return bin_dir


# ---------------------------------------------------------------------------
# Java JAR tools (openjdk-17 assumed present)
# ---------------------------------------------------------------------------

def install_beagle(
    version: str = BEAGLE_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download beagle JAR and write a shell wrapper. Returns the bin/ directory."""
    return _install_jar_tool(
        'beagle', version,
        f'https://faculty.washington.edu/browning/beagle/beagle.{version}.jar',
        f'beagle.{version}.jar',
        install_dir or INSTALL_DIR, force,
    )


# ---------------------------------------------------------------------------
# Additional static binary / JAR / pip tools
# ---------------------------------------------------------------------------

def install_king(
    version: str = KING_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download KING pre-compiled binary. Returns the bin/ directory."""
    ver_nodot = version.replace('.', '')
    return _install_archived_binary(
        'king', version,
        f'https://www.kingrelatedness.com/executables/Linux-king{ver_nodot}.tar.gz',
        'king',
        install_dir or INSTALL_DIR, build_dir, force,
    )


def install_plink2(
    version: str = PLINK2_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download PLINK2 pre-compiled binary (AVX2). Returns the bin/ directory."""
    return _install_archived_binary(
        'plink2', version,
        f'https://s3.amazonaws.com/plink2-assets/plink2_linux_avx2_{version}.zip',
        'plink2',
        install_dir or INSTALL_DIR, build_dir, force,
    )


def install_plink19(
    version: str = PLINK19_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download PLINK 1.9 pre-compiled binary. Returns the bin/ directory."""
    return _install_archived_binary(
        'plink19', version,
        f'https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_{version}.zip',
        'plink',
        install_dir or INSTALL_DIR, build_dir, force,
    )


def install_hapibd(
    version: str = HAPIBD_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download hap-IBD JAR and write a shell wrapper. Returns the bin/ directory."""
    return _install_jar_tool(
        'hapibd', version,
        'https://faculty.washington.edu/browning/hap-ibd.jar',
        'hap-ibd.jar',
        install_dir or INSTALL_DIR, force,
        wrapper_name='hap-ibd',
    )


def install_ibdends(
    version: str = IBDENDS_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:
    """Download ibd-ends JAR and write a shell wrapper. Returns the bin/ directory."""
    return _install_jar_tool(
        'ibdends', version,
        'https://faculty.washington.edu/browning/ibd-ends.jar',
        'ibd-ends.jar',
        install_dir or INSTALL_DIR, force,
        wrapper_name='ibd-ends',
    )


def _pip_install_pypi(pkg: str, force: bool, build_deps: list[str] | None = None) -> None:
    """Install a package from PyPI."""
    if not force and _pip_installed(pkg.split('==')[0]):
        log.info('%s already installed', pkg)
        return
    if build_deps:
        _run_pip(['pip', 'install', *build_deps])
    flags = ['--no-build-isolation'] if build_deps else []
    _run_pip(['pip', 'install', *flags, pkg])
    log.info('%s installed', pkg)


def _pip_install_git(
    pkg: str,
    url: str,
    version: str | None,
    force: bool,
    build_deps: list[str] | None = None,
) -> None:
    """Install a package from a git URL via pip, optionally pinning to a ref.

    build_deps: pip packages to install first (build backends not in the
    isolated env by default, e.g. ['poetry-core'] for Poetry-based projects).
    """
    if not force and _pip_installed(pkg):
        log.info('%s already installed', pkg)
        return
    if build_deps:
        _run_pip(['pip', 'install', *build_deps])
    ref = f'@{version}' if version else ''
    _run_pip(['pip', 'install', '--no-build-isolation', f'git+{url}{ref}'])
    log.info('%s installed', pkg)


def install_fraposa(
    version: str | None = None,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> None:
    """Install fraposa_pgsc (PGS Catalog fork) from PyPI.

    All released versions pin pandas<2.0.0 and numpy<2.0.0, but the COS VM
    ships with pandas 2.x.  We install with --no-deps to bypass the pin —
    the wheel itself is pure Python and works with pandas 2.x provided the
    code avoids removed APIs (DataFrame.append removed in 2.0; get_dummies
    now returns bool dtype instead of uint8).  Other deps (numpy, matplotlib,
    scikit-learn) are assumed already present.
    Pass version= to pin e.g. version='1.0.2'.
    """
    pkg = f'fraposa-pgsc=={version}' if version else 'fraposa-pgsc'
    pkg_name = pkg.split('==')[0]
    if not force and _pip_installed(pkg_name):
        log.info('%s already installed', pkg)
        return
    # Install --no-deps to bypass the pandas<2.0.0 pin, then explicitly add
    # pyplink which fraposa imports at runtime but is not pre-installed on the
    # COS VM (confirmed missing: ModuleNotFoundError: No module named 'pyplink').
    _run_pip(['pip', 'install', '--no-deps', pkg])
    _run_pip(['pip', 'install', 'pyplink'])
    log.info('%s installed (--no-deps, pandas pin bypassed; pyplink added)', pkg)


def install_adamixture(
    version: str | None = None,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> None:
    """Install ADAMIXTURE (PyTorch population-structure model) from PyPI.

    Note: this is ADAMIXTURE (Python/PyTorch package, PyPI: adamixture), not
    ADMIXTURE (C++ binary by David Alexander). See
    cpg_notebook.pop_genetics.install_admixture for the latter.

    Available as a wheel on PyPI (v1.5.5+). Has Cython extensions compiled
    with OpenMP — Linux g++ handles this natively. Requires torch>=2.6.
    Pass version= to pin e.g. version='1.5.5'.
    """
    pkg = f'adamixture=={version}' if version else 'adamixture'
    _pip_install_pypi(pkg, force)


def install_archetypal_analysis(
    version: str | None = None,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> None:
    """Install archetypal-analysis (AI-sandbox) from git.

    Pins numpy==1.19.5, sklearn==0.24.2, scipy==1.5.4 (2020-vintage), but
    the code itself is unlikely to use removed APIs — we bypass the pins
    with --no-deps and rely on the already-installed modern versions.
    Not available on PyPI; always installed from the git HEAD.
    """
    pkg = 'archetypal-analysis-popgen'
    if not force and _pip_installed(pkg):
        log.info('%s already installed', pkg)
        return
    ref = f'@{version}' if version else ''
    _run_pip([
        'pip', 'install', '--no-deps',
        f'git+https://github.com/AI-sandbox/archetypal-analysis.git{ref}',
    ])
    log.info('%s installed (--no-deps, version pins bypassed)', pkg)


# ---------------------------------------------------------------------------
# Path registration (run once after installs complete)
# ---------------------------------------------------------------------------

def setup_path(install_dir: Path | None = None) -> None:
    """
    Register BCFTOOLS_PLUGINS in three places so every caller sees it:

    1. os.environ  — immediate effect for the current Python process and any
                     %%bash cells that run afterwards (subprocesses inherit the
                     kernel's environment). This is the primary mechanism when
                     setup_path() is called from a Python cell.

    2. ~/.bashrc   — sourced by each %%bash subprocess, covers the case where
                     the install was done via %%bash (os.environ mutation in a
                     subprocess dies with that subprocess).

    3. /etc/environment — read by PAM at login; covers gcloud ssh / docker exec
                          sessions that start after the install.

    Both file writes replace any existing BCFTOOLS_PLUGINS line (not just
    append), so stale values from previous installs can't shadow the correct path.
    """
    install_dir = install_dir or INSTALL_DIR
    plugins_dir = install_dir / f'bcftools-{BCFTOOLS_VERSION}' / 'lib' / 'bcftools'
    if not plugins_dir.exists():
        return

    # /etc/environment — strip any existing BCFTOOLS_PLUGINS line, append correct one
    env_line = f'BCFTOOLS_PLUGINS={plugins_dir}'
    env_file = Path('/etc/environment')
    existing = env_file.read_text() if env_file.exists() else ''
    filtered = '\n'.join(
        ln for ln in existing.splitlines() if not ln.startswith('BCFTOOLS_PLUGINS=')
    )
    new_content = filtered.rstrip('\n') + f'\n{env_line}\n'
    if new_content != existing:
        _run(['sudo', 'bash', '-c', f"printf '%s' '{new_content}' > /etc/environment"])
        log.info('BCFTOOLS_PLUGINS written to /etc/environment')

    # ~/.bashrc — strip any existing export BCFTOOLS_PLUGINS line, append correct one
    bashrc = Path.home() / '.bashrc'
    export_line = f'export BCFTOOLS_PLUGINS={plugins_dir}'
    bashrc_text = bashrc.read_text() if bashrc.exists() else ''
    filtered_rc = '\n'.join(
        ln for ln in bashrc_text.splitlines()
        if not ln.startswith('export BCFTOOLS_PLUGINS=')
    )
    new_rc = filtered_rc.rstrip('\n') + f'\n{export_line}\n'
    if new_rc != bashrc_text:
        bashrc.write_text(new_rc)
        log.info('BCFTOOLS_PLUGINS export written to ~/.bashrc')

    # Apply to the current process so subsequent Python/subprocess calls see it
    os.environ['BCFTOOLS_PLUGINS'] = str(plugins_dir)
    log.info('BCFTOOLS_PLUGINS=%s', plugins_dir)


# ---------------------------------------------------------------------------
# Public tool registry + install_all
# ---------------------------------------------------------------------------

TOOL_FUNCS: dict[str, callable] = {
    'htslib':    install_htslib,
    'samtools':  install_samtools,
    'bcftools':  install_bcftools,
    'shapeit5':  install_shapeit5,
    'glimpse2':  install_glimpse2,
    'beagle':    install_beagle,
    'king':      install_king,
    'plink2':    install_plink2,
    'plink19':   install_plink19,
    'hapibd':    install_hapibd,
    'ibdends':   install_ibdends,
    'fraposa':   install_fraposa,
    'adamixture': install_adamixture,
    'archetypal': install_archetypal_analysis,
}


def install_all(
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> None:
    """Install all tools then run setup_path()."""
    install_dir = install_dir or INSTALL_DIR
    kwargs = dict(install_dir=install_dir, build_dir=build_dir, force=force)
    tools = list(TOOL_FUNCS.items())
    for i, (name, fn) in enumerate(tools, 1):
        print(f'\n{"─" * 52}\n  [{i}/{len(tools)}]  {name}\n{"─" * 52}', flush=True)
        fn(**kwargs)
    setup_path(install_dir=install_dir)


# ---------------------------------------------------------------------------
# Uninstall functions
# ---------------------------------------------------------------------------

def _uninstall(tool: str, version: str, install_dir: Path, extra: callable | None = None) -> None:
    prefix = _prefix(tool, version, install_dir)
    if not prefix.exists():
        log.info('%s %s is not installed', tool, version)
        return
    _unlink_from_system(prefix / 'bin')
    if extra:
        extra()
    shutil.rmtree(prefix)
    log.info('%s %s removed', tool, version)


def uninstall_htslib(version: str = HTSLIB_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('htslib', version, install_dir or INSTALL_DIR, extra=_deregister_ldconfig)


def uninstall_samtools(version: str = SAMTOOLS_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('samtools', version, install_dir or INSTALL_DIR)


def uninstall_bcftools(version: str = BCFTOOLS_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('bcftools', version, install_dir or INSTALL_DIR)


def uninstall_shapeit5(version: str = SHAPEIT5_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('shapeit5', version, install_dir or INSTALL_DIR)


def uninstall_glimpse2(version: str = GLIMPSE2_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('glimpse2', version, install_dir or INSTALL_DIR)


def uninstall_beagle(version: str = BEAGLE_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('beagle', version, install_dir or INSTALL_DIR)


def uninstall_king(version: str = KING_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('king', version, install_dir or INSTALL_DIR)


def uninstall_plink2(version: str = PLINK2_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('plink2', version, install_dir or INSTALL_DIR)


def uninstall_plink19(version: str = PLINK19_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('plink19', version, install_dir or INSTALL_DIR)


def uninstall_hapibd(version: str = HAPIBD_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('hapibd', version, install_dir or INSTALL_DIR)


def uninstall_ibdends(version: str = IBDENDS_VERSION, install_dir: Path | None = None) -> None:
    _uninstall('ibdends', version, install_dir or INSTALL_DIR)


def uninstall_fraposa(**kwargs) -> None:
    _pip_uninstall('fraposa-pgsc')


def uninstall_adamixture(**kwargs) -> None:
    _pip_uninstall('adamixture')


def uninstall_archetypal_analysis(**kwargs) -> None:
    _pip_uninstall('archetypal-analysis')


UNINSTALL_FUNCS: dict[str, callable] = {
    'htslib':    uninstall_htslib,
    'samtools':  uninstall_samtools,
    'bcftools':  uninstall_bcftools,
    'shapeit5':  uninstall_shapeit5,
    'glimpse2':  uninstall_glimpse2,
    'beagle':    uninstall_beagle,
    'king':      uninstall_king,
    'plink2':    uninstall_plink2,
    'plink19':   uninstall_plink19,
    'hapibd':    uninstall_hapibd,
    'ibdends':   uninstall_ibdends,
    'fraposa':   uninstall_fraposa,
    'adamixture': uninstall_adamixture,
    'archetypal': uninstall_archetypal_analysis,
}


def uninstall_all(install_dir: Path | None = None) -> None:
    """Uninstall all tools."""
    install_dir = install_dir or INSTALL_DIR
    for name, fn in UNINSTALL_FUNCS.items():
        log.info('=== uninstall %s ===', name)
        fn(install_dir=install_dir)


# Binary tools: (dir_name, version, relative_check_path)
_INSTALL_CHECK = {
    'htslib':    ('htslib',    HTSLIB_VERSION,    'bin/bgzip'),
    'samtools':  ('samtools',  SAMTOOLS_VERSION,  'bin/samtools'),
    'bcftools':  ('bcftools',  BCFTOOLS_VERSION,  'bin/bcftools'),
    'shapeit5':  ('shapeit5',  SHAPEIT5_VERSION,  'bin/phase_common'),
    'glimpse2':  ('glimpse2',  GLIMPSE2_VERSION,  'bin/GLIMPSE2_phase'),
    'beagle':    ('beagle',    BEAGLE_VERSION,    'bin/beagle'),
    'king':      ('king',      KING_VERSION,      'bin/king'),
    'plink2':    ('plink2',    PLINK2_VERSION,    'bin/plink2'),
    'plink19':   ('plink19',   PLINK19_VERSION,   'bin/plink'),
    'hapibd':    ('hapibd',    HAPIBD_VERSION,    'bin/hap-ibd'),
    'ibdends':   ('ibdends',   IBDENDS_VERSION,   'bin/ibd-ends'),
}

# Pip tools: pip package name used for install-check
_PIP_CHECK = {
    'fraposa':    'fraposa-pgsc',
    'adamixture': 'adamixture',
    'archetypal': 'archetypal-analysis-popgen',
}


def list_tools(install_dir: Path | None = None) -> None:
    """Print all installable tools and whether each is currently installed."""
    install_dir = install_dir or INSTALL_DIR
    rows = {}
    for name, (tool, version, check) in _INSTALL_CHECK.items():
        installed = (_prefix(tool, version, install_dir) / check).exists()
        rows[name] = (version, 'installed' if installed else 'not installed')
    for name, pkg in _PIP_CHECK.items():
        installed = _pip_installed(pkg)
        rows[name] = ('(pip)', 'installed' if installed else 'not installed')
    print(f'{"tool":<14} {"version":<20} {"status"}')
    print('-' * 46)
    for name in sorted(rows):
        version, status = rows[name]
        print(f'{name:<14} {version:<20} {status}')


