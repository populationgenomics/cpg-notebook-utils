# cpg_notebook.pop_genetics — design plan

Module path: `cpg_notebook/pop_genetics.py`

This module installs population genetics tools into a persistent directory on a
CPG managed VM and makes them available on the system PATH for all subsequent
interactive shell sessions, including `./colab ssh` workflows.

General genomics tools (htslib, samtools, bcftools, shapeit5, glimpse2, beagle,
king, plink2, plink19, hapibd, ibdends, fraposa) live in the companion module
`cpg_notebook/genomics.py` — see `docs/genomics.md`.

---

## Rationale

Same infrastructure as `genomics`: tools install to `/content/tools`
(persistent local SSD), binaries are symlinked into `/usr/local/bin` at install
time. These three tools are grouped separately because they share a common use
case (population structure, ancestry, dimensionality reduction) and have heavier
build requirements than the core genomics tools.

---

## Python API (primary interface)

Install from a Python cell:

```python
from cpg_notebook import pop_genetics as npg
npg.install_all()
```

Or selectively:

```python
npg.install_eigensoft()
npg.install_flare()
npg.install_admixture()
```

---

## %%bash cell usage

After the Python install cell runs, tools are available in subsequent `%%bash`
cells immediately via `/usr/local/bin`.

### Verify

```bash
%%bash
smartpca 2>&1 | head -2
flare    2>&1 | head -2
admixture 2>&1 | head -2
```

### Use tools

```bash
%%bash
smartpca -p /content/my.par
```

```bash
%%bash
flare gt=input.vcf.gz out=output
```

---

## Configuration constants

```python
EIGENSOFT_VERSION = "8.0.0"
FLARE_VERSION     = "latest"   # no versioned release — downloads current jar from faculty page
ADMIXTURE_VERSION = "1.3.1"
```

All constants can be overridden by passing `version=` to individual install functions.
`INSTALL_DIR` and `BUILD_DIR` are inherited from `cpg_notebook.genomics`.

---

## Public API

```python
def install_<tool>(
    version: str = <TOOL>_VERSION,
    install_dir: Path | None = None,
    build_dir: Path = BUILD_DIR,
    force: bool = False,
) -> Path:   # returns prefix / "bin" directory
```

---

## Tool inventory

### Source-compiled tools

| Tool | Install check | Build deps (apt) | Notes |
|---|---|---|---|
| `install_eigensoft` | `prefix/bin/smartpca` | `make gcc g++ libgsl-dev liblapack-dev liblapacke-dev libopenblas-dev` | Makefile patch required — see debug note below |

### Java JAR tools

| Tool | JAR location | Wrapper |
|---|---|---|
| `install_flare` | `install_dir/jars/flare.jar` | `install_dir/bin/flare` → `exec java -jar ... "$@"` |

Java runtime assumed present (`openjdk-17`). No Java installation is performed.

### Static binary tools

| Tool | Binary | Source |
|---|---|---|
| `install_admixture` | `admixture` | Pre-compiled binary from dalexander.github.io |

---

## Key design decisions

### EIGENSOFT — normalised install layout

The upstream Makefile scatters binaries across `src/CONVERTF/` and
`src/EIGENSTRAT/`. The install function copies all ELF executables from those
directories into a normalised `prefix/bin/` to match the layout of every other
tool in this module.

### EIGENSOFT — liblapacke-dev debug

The build fails with `undefined reference to LAPACKE_dsyevd` unless two things
are in place together:

1. `liblapacke-dev` is installed (Debian/Ubuntu split this into a separate
   package from `liblapack-dev`).
2. The Makefile links against it — upstream ships `# override LDLIBS += -llapacke`
   commented out. The install function uncomments this line before building.

### FLARE — unversioned URL

The faculty.washington.edu JAR URL (`browning/flare.jar`) is unversioned —
there is no GitHub release tag with a stable download URL. The install function
downloads the current JAR at install time.

---

## Private helpers

All private helpers (`_run`, `_apt`, `_prefix`, `_download`, `_untar`,
`_find_binary`, `_link_to_system`, `_write_jar_wrapper`, `_uninstall`, etc.)
are imported from `cpg_notebook.genomics`.

---

## Tests

`tests/test_pop_genetics.py` — same patterns as `tests/test_genomics.py`:
- Already-installed guard short-circuits correctly
- `force=True` bypasses the guard
- EIGENSOFT Makefile patch is applied
- `main()` argparse: `--all`, named tools, `--force`, `--install-dir`
