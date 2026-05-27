# cpg_notebook.genomics — design plan

Module path: `cpg_notebook/genomics.py`

This module installs bioinformatics tools into a persistent directory on a CPG
managed VM and makes them available on the system PATH for all
subsequent interactive shell sessions, including `./colab ssh` workflows.

Population genetics tools (EIGENSOFT, FLARE, ADMIXTURE) live in the companion
module `cpg_notebook/pop_genetics.py` — see `docs/pop_genetics.md`.

---

## Rationale

Tools are installed to a persistent stateful partition (default `/content/tools`)
so they survive container restarts. Binaries are symlinked into `/usr/local/bin`
at install time so they are available on PATH immediately in any shell mode
(login, non-login, interactive docker exec) without requiring the user to source
anything. Shared libraries (htslib) are registered via `ldconfig` rather than
`LD_LIBRARY_PATH` for the same reason.

---

## Python API (primary interface)

Install from a Python cell:

```python
from cpg_notebook import genomics as ng
ng.install_all()
```

Or selectively:

```python
ng.install_htslib()
ng.install_samtools()
ng.install_bcftools()
ng.setup_path()
```

Subsequent calls are fast: each tool skips installation if its binary is already
present. Pass `force=True` to recompile.

---

## %%bash cell usage

The tools write symlinks into `/usr/local/bin` and register libraries via
`ldconfig`, so they are accessible in subsequent `%%bash` cells immediately
after the Python install cell runs.

### Verify

```bash
%%bash
bcftools --version
samtools --version | head -1
bgzip   --version | head -1
shapeit5 phase_common --version 2>&1 | head -1
GLIMPSE2_phase --version 2>&1 | head -1
java -version 2>&1 | head -1
beagle 2>&1 | head -2
```

### Use tools

```bash
%%bash
bcftools view -h gs://my-bucket/data.vcf.gz
```

```bash
%%bash
shapeit5 phase_common \
  --input gs://my-bucket/input.bcf \
  --map   /content/maps/chr1.b38.gmap.gz \
  --output /content/output/chr1.phased.bcf \
  --region chr1
```

### BCFTOOLS_PLUGINS in %%bash cells

`BCFTOOLS_PLUGINS` is written to `/etc/environment` during `setup_path()`, but
`/etc/environment` is read only at PAM session start — it will not be visible in
`%%bash` cells started in the same Jupyter kernel session. Set it explicitly in
any cell that uses bcftools plugins:

```bash
%%bash
export BCFTOOLS_PLUGINS=/content/tools/bcftools-1.23.1/lib/bcftools
bcftools +gtc2vcf --help
```

Or add it to `~/.bashrc` once so all future `%%bash` cells inherit it:

```bash
%%bash
echo 'export BCFTOOLS_PLUGINS=/content/tools/bcftools-1.23.1/lib/bcftools' \
  >> ~/.bashrc
```

---

## Configuration constants

All version constants are module-level and can be overridden by passing
`version=` to individual install functions.

```python
HTSLIB_VERSION    = "1.23.1"
SAMTOOLS_VERSION  = "1.23.1"
BCFTOOLS_VERSION  = "1.23.1"
SHAPEIT5_VERSION  = "5.1.1"
BEAGLE_VERSION    = "27Feb25.75f"
GLIMPSE2_VERSION  = "2.0.0"
KING_VERSION      = "2.3.2"
PLINK2_VERSION    = "20260311"
PLINK19_VERSION   = "20250819"
HAPIBD_VERSION    = "latest"
IBDENDS_VERSION   = "latest"

WORKDIR     = Path(os.environ.get("WORKDIR", "/content"))
INSTALL_DIR = WORKDIR / "tools"
BUILD_DIR   = Path("/tmp/build")
```

---

## Public API

Every `install_*` function shares this signature:

```python
def install_<tool>(
    version: str = <TOOL>_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:   # returns prefix / "bin" directory
```

`setup_path()` registers the installed libs with ldconfig, writes
`BCFTOOLS_PLUGINS` to `/etc/environment`, and creates `/usr/local/bin`
symlinks — it is called automatically by `install_all()`.

---

## Tool inventory

### Source-compiled tools

| Tool | Install check | Build deps (apt) | Notes |
|---|---|---|---|
| `install_htslib` | `prefix/bin/bgzip` | `autoconf make gcc libcurl4-openssl-dev libbz2-dev liblzma-dev zlib1g-dev libssl-dev` | `--enable-libcurl --enable-gcs --enable-plugins` |
| `install_samtools` | `prefix/bin/samtools` | (same) | links against htslib prefix |
| `install_bcftools` | `prefix/bin/bcftools` + `prefix/lib/bcftools/*.so` | (same) | see plugin note below |

### Static binary tools

| Tool | Binaries | Source |
|---|---|---|
| `install_shapeit5` | `phase_common`, `phase_rare`, `ligate`, `switch`, `simulate`, `xcftools` | GitHub releases `odelaneau/shapeit5` — `*_static` suffix stripped |
| `install_glimpse2` | `GLIMPSE2_phase`, `GLIMPSE2_chunk`, `GLIMPSE2_ligate`, `GLIMPSE2_concordance`, `GLIMPSE2_split_reference` | GitHub releases `odelaneau/GLIMPSE` — `_static` suffix stripped |
| `install_king` | `king` | Pre-compiled binary from kingrelatedness.com |
| `install_plink2` | `plink2` | Pre-compiled AVX2 binary from plink2-assets S3 |
| `install_plink19` | `plink` | Pre-compiled binary from plink1-assets S3 |

### Java JAR tools

| Tool | JAR location | Wrapper |
|---|---|---|
| `install_beagle` | `install_dir/jars/beagle.<version>.jar` | `install_dir/bin/beagle` → `exec java -jar ... "$@"` |
| `install_hapibd` | `install_dir/jars/hap-ibd.jar` | `install_dir/bin/hap-ibd` → `exec java -jar ... "$@"` |
| `install_ibdends` | `install_dir/jars/ibd-ends.jar` | `install_dir/bin/ibd-ends` → `exec java -jar ... "$@"` |

Java runtime assumed present (`openjdk-17`). No Java installation is performed.

### pip tools

| Tool | Package | Notes |
|---|---|---|
| `install_fraposa` | `fraposa-pgsc` | installed `--no-deps` to bypass pandas<2.0 pin |
| `install_adamixture` | `adamixture` | requires torch>=2.6 |
| `install_archetypal_analysis` | `archetypal-analysis-popgen` | installed `--no-deps` from git HEAD |

---

## Key design decisions

### BCftools plugins — bug fix

The original code kept only a hardcoded list of four gtc2vcf plugin names,
silently discarding all ~25 default bcftools plugins (split-vep, trio-dnm2,
fixploidy, etc.). The fix globs all compiled `.so` files:

```python
# old — discards default plugins
for name in ["idat2gtc", "gtc2vcf", "affy2vcf", "BAFregress"]:
    so = src_dir / "plugins" / f"{name}.so"
    ...

# new — keeps everything the build produced
for so in (src_dir / "plugins").glob("*.so"):
    shutil.copy2(so, plugins_dir / so.name)
```

### PATH / shell accessibility

`os.environ` mutations die with the Python process and are invisible to the
calling shell. Instead:

- Binaries: symlinked into `/usr/local/bin` (always on PATH, no sourcing needed)
- Shared libs (htslib): `/etc/ld.so.conf.d/cpg-tools.conf` + `ldconfig`
- `BCFTOOLS_PLUGINS`: written to `/etc/environment` (read by PAM for all sessions)

This approach works correctly for `docker exec -it container bash` (interactive
non-login shell), which does not source `/etc/profile.d/`.

### GCS credentials — `setup_gcs_token` dropped

htslib compiled with `--enable-gcs` resolves GCS credentials automatically:
`GCS_OAUTH_TOKEN` env → `GOOGLE_APPLICATION_CREDENTIALS` file → Application
Default Credentials (GCE metadata server). On a GCE VM, option 3 works without
any explicit token setup. The `setup_gcs_token` helper from the original
notebook script is not needed.

---

## Private helpers

```python
def _run(cmd, cwd=None, silent=False) -> str
def _apt(packages: str) -> None
def _prefix(tool: str, version: str) -> Path
def _link_to_system(bin_dir: Path) -> None        # symlink → /usr/local/bin
def _register_ldconfig(lib_dir: Path) -> None     # write conf + run ldconfig
def _write_jar_wrapper(jar: Path, wrapper: Path) -> None   # exec java -jar wrapper
```

---

## Tests

`tests/test_genomics.py` using `unittest.TestCase` with `unittest.mock.patch`
on `subprocess.run` and `urllib.request.urlopen`. Coverage targets:

- Already-installed guard short-circuits correctly (no subprocess calls)
- `force=True` bypasses the guard
- Plugin glob behaviour (assert no hardcoded names in copy logic)
- `_link_to_system` creates symlinks for all executables in a bin dir
- `_write_jar_wrapper` produces a correctly formatted executable shell script
- `main()` argparse: `--all`, named tools, `--force`, `--install-dir`
