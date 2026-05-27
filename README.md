# cpg-notebook-utils

Utilities for CPG COS notebook VMs (the interactive analysis environment
backed by Google Cloud's Container-Optimized OS image with Colab on top).

Three independent modules:

| Module | Purpose |
|---|---|
| `cpg_notebook.genomics` | Install bioinformatics CLI tools (htslib, samtools, bcftools, shapeit5, glimpse2, beagle, king, plink2, plink19, hapibd, ibdends, fraposa) |
| `cpg_notebook.pop_genetics` | Install population genetics tools (EIGENSOFT, FLARE, ADMIXTURE) |
| `cpg_notebook.datascience` | Hail ↔ Polars conversion helpers |

## Install

```bash
pip install cpg-notebook-utils
```

## Usage

```python
# Bioinformatics tools — installs to /content/tools, symlinks into /usr/local/bin
from cpg_notebook import genomics as ng
ng.install_all()
ng.install_bcftools()

# Population genetics tools
from cpg_notebook import pop_genetics as npg
npg.install_eigensoft()

# Hail → Polars round-trip
from cpg_notebook import datascience as nds
pt = nds.to_polars(my_hail_table)
ht = nds.from_polars(pt)

# Interactive PCA plot with X/Y/SD dropdowns
fig = nds.plot_pca(
    df,
    layers=[
        nds.PCALayer(name='Ref', group_col='genetic_ancestry',
                     mask=df['set'] == 'reference', marker_symbol='cross'),
        nds.PCALayer(name='OurDNA', group_col='screening_ancestry_group',
                     mask=df['set'] == 'ourdna', draw_ellipses=True),
    ],
)
fig.show()
```

See `docs/genomics.md` and `docs/pop_genetics.md` for full design notes.

## Runtime assumptions

- **`genomics` / `pop_genetics`**: target the CPG COS notebook VM. They write
  to `/content/tools` (local SSD, ~2.9T, does not persist across VM shutdown)
  and symlink into `/usr/local/bin`. They invoke `apt-get` for build deps.
- **`datascience`**: expects `hail`, `polars`, `pandas`, `numpy` to be
  importable. All are pre-installed in the CPG notebook image, so they are
  deliberately **not** declared as install dependencies — pip-installing hail
  outside the image is fragile (version-coupled to the cluster). The one
  plotting dep that *is* declared is `plotly`, used by `plot_pca`.

## Development

```bash
pip install -e '.[dev]'
pre-commit install
pytest
```
