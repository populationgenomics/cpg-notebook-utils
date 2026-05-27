"""Utilities for CPG COS notebook VMs.

Submodules:
    - genomics: install bioinformatics tools (htslib, samtools, bcftools, …)
    - pop_genetics: install population genetics tools (EIGENSOFT, FLARE, ADMIXTURE)
    - datascience: Hail ↔ Polars conversion helpers

The genomics modules assume the COS VM environment described in the README.
The datascience module assumes hail and polars are already installed in the
runtime (true for the CPG notebook image).
"""

__version__ = '0.1.0'
