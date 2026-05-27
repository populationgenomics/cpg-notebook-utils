"""cpg_notebook.datascience — data science utilities for CPG notebook environments.

Provides helpers for converting between Hail and Polars data structures, plus
interactive plotly visualisations, targeting CPG COS notebook VMs where Hail
(via Spark), Polars, and Plotly are all available.

From a notebook Python cell::

    from cpg_notebook import datascience as nds

    # Hail → Polars
    pt = nds.to_polars(my_hail_table)
    pt.df          # pl.DataFrame
    pt.globals     # dict of Hail globals

    # Polars → Hail (round-trip)
    ht = nds.from_polars(pt)

    # Interactive PCA scatter with X/Y/SD dropdowns
    fig = nds.plot_pca(
        df,
        layers=[
            nds.PCALayer(name='Ref', group_col='ancestry'),
            nds.PCALayer(name='Study', group_col='cohort', draw_ellipses=True),
        ],
    )
    fig.show()

Conversion paths
----------------
Both directions support two paths, chosen automatically based on the schema:

1. **Parquet round-trip** (default for nested schemas)
   Spark writes parquet in parallel; Polars reads it multi-threaded and
   columnar.  Handles tuples, structs, sets, locus, and interval cleanly via
   Hail's ``expand_types`` step that runs inside ``to_spark``.

2. **Arrow + pandas** (default for fully-flat schemas)
   Faster than parquet when the schema is flat and the table is small.
   Risky for nested schemas: PySpark's ``toPandas()`` can silently fall back
   to a row-based conversion when Arrow can't handle the schema.

Hail-native type round-tripping
--------------------------------
Hail types that have no direct Arrow/parquet equivalent — ``tlocus``,
``tinterval``, ``tset``, ``ttuple``, and ``tcall`` — are expanded to plain
structs and arrays by Hail's ``expand_types`` step during the forward
conversion.  :class:`PolarsTable` preserves the original Hail row schema so
that :func:`from_polars` can reconstruct these types on the way back.

Globals are extracted separately via ``hl.eval(ht.globals)`` because Spark
DataFrames have no equivalent concept and would otherwise be dropped.

Notes on columns added or renamed in Polars
--------------------------------------------
If you add new columns in Polars before calling :func:`from_polars`, Hail
will infer their types from the Arrow/parquet schema — no action is needed.

If you **rename** a column that originally held a Hail-native type (e.g.
``locus``), the stored ``row_schema`` will no longer match by name and that
column will be treated as new — its type will not be reconstructed.  In that
case pass the updated schema via ``row_schema`` or reconstruct the type
manually after conversion.

Globals caveat
--------------
Globals that originally contained Hail-native types (``tlocus``, etc.) are
converted to plain Python dicts/lists by :func:`to_polars` and are
re-attached as plain Hail structs/arrays by :func:`from_polars`.  The
original Hail type is not restored for globals.  This is rarely an issue in
practice as globals are usually plain scalars or dicts.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import hail as hl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from hail.expr import types as htypes

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------


@dataclass
class PolarsTable:
    """A Polars DataFrame paired with metadata needed to round-trip back to Hail.

    Instances are normally produced by :func:`to_polars`; all metadata fields
    are populated automatically.  You can also construct one manually — in
    that case leave *row_schema* and *keys* at their defaults and
    :func:`from_polars` will skip type reconstruction and key restoration
    (with a warning).

    Attributes:
        df: The row data as a Polars DataFrame.
        globals: Plain Python dict of the Hail Table's globals, recursively
            converted from the underlying ``hl.Struct``.  Empty dict when the
            source table carried no globals.
        row_schema: The full Hail row schema (``ht.row.dtype``) captured
            before conversion.  Used by :func:`from_polars` to reconstruct
            Hail-native types (``tlocus``, ``tinterval``, ``tset``,
            ``ttuple``) that are expanded to plain structs/arrays during the
            forward conversion.  ``None`` when ``flatten=True`` was used (the
            flattened column layout is incompatible with schema-guided
            reconstruction).
        keys: Names of the Hail Table's key fields (``list(ht.key.dtype.fields)``).
            Used by :func:`from_polars` to restore the row key via
            ``ht.key_by(*keys)``.  Empty list for unkeyed tables.
    """

    df: pl.DataFrame
    globals: dict = field(default_factory=dict)
    row_schema: htypes.tstruct | None = None
    keys: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# hl.Struct → plain Python (used for globals)
# ---------------------------------------------------------------------------


def _hail_to_python(value) -> object:
    """Recursively convert an ``hl.Struct`` (and nested containers) to plain Python.

    Newer Hail versions no longer expose ``hl.Struct.to_dict()``, and even
    when they did it was non-recursive.  This function walks the value tree
    and converts every Struct, list, tuple, set, and dict it encounters.
    Sets and frozensets are returned as lists so they remain JSON- and
    Polars-friendly downstream.

    Args:
        value: Any value that may include ``hl.Struct`` instances at any
            nesting depth.

    Returns:
        A plain Python object (dict, list, tuple, or scalar) with all
        ``hl.Struct`` instances replaced by dicts.
    """
    if isinstance(value, hl.Struct):
        return {k: _hail_to_python(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _hail_to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_hail_to_python(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_hail_to_python(v) for v in value)
    if isinstance(value, (set, frozenset)):
        # Return as list: easier to consume in Polars / JSON than a set.
        return [_hail_to_python(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------

_NESTED_HAIL_TYPES = (
    htypes.tstruct,
    htypes.tarray,
    htypes.tset,
    htypes.tdict,
    htypes.ttuple,
    htypes.tlocus,
    htypes.tinterval,
)

_RECONSTRUCTIBLE_HAIL_TYPES = (
    htypes.tlocus,
    htypes.tinterval,
    htypes.tcall,
    htypes.tset,
    htypes.ttuple,
    htypes.tstruct,
    htypes.tarray,
    htypes.tdict,
)


def _has_nested_types(struct_dtype: htypes.tstruct) -> bool:
    """Return True if any top-level field in *struct_dtype* is a nested Hail type.

    Used by :func:`to_polars` to select the optimal conversion path when
    ``prefer='auto'``.

    Args:
        struct_dtype: The ``dtype`` of a Hail row expression (``ht.row.dtype``).

    Returns:
        True when at least one top-level field has a struct, array, set,
        dict, tuple, locus, or interval type.
    """
    return any(isinstance(t, _NESTED_HAIL_TYPES) for t in struct_dtype.types)


def _polars_schema_is_nested(df: pl.DataFrame) -> bool:
    """Return True if *df* contains any List, Array, or Struct columns.

    Used by :func:`from_polars` to select the optimal ingestion path when
    ``prefer='auto'``.

    Args:
        df: Polars DataFrame to inspect.

    Returns:
        True when at least one column has a ``pl.List``, ``pl.Array``, or
        ``pl.Struct`` dtype.
    """
    return any(isinstance(dt, (pl.List, pl.Array, pl.Struct)) for dt in df.dtypes)


# ---------------------------------------------------------------------------
# Hail type reconstruction after expand_types
# ---------------------------------------------------------------------------


def _needs_reconstruction(hail_type) -> bool:
    """Return True if *hail_type* (or any nested type) was transformed by expand_types.

    ``expand_types`` (invoked inside ``to_spark``) rewrites:

    - ``tlocus``   → ``tstruct(contig: tstr, position: tint32)``
    - ``tinterval`` → ``tstruct(start, end, includes_start, includes_end)``
    - ``tcall``    → ``tint32``
    - ``tset``     → ``tarray``
    - ``ttuple``   → ``tstruct(_0, _1, ...)``

    Plain scalar types and ``tstruct``/``tarray``/``tdict`` with no
    special-type descendants are left unchanged and do not need reconstruction.

    Args:
        hail_type: Any Hail type object.

    Returns:
        True when reconstruction is required.
    """
    if isinstance(hail_type, (htypes.tlocus, htypes.tinterval, htypes.tcall, htypes.tset, htypes.ttuple)):
        return True
    if isinstance(hail_type, htypes.tstruct):
        return any(_needs_reconstruction(t) for t in hail_type.types)
    if isinstance(hail_type, htypes.tarray):
        return _needs_reconstruction(hail_type.element_type)
    if isinstance(hail_type, htypes.tdict):
        return _needs_reconstruction(hail_type.key_type) or _needs_reconstruction(hail_type.value_type)
    return False


def _reconstruct_expr(expr, hail_type):
    """Return a Hail expression that restores *expr* to *hail_type* after expand_types.

    Handles all five transformed types recursively:

    - ``tlocus``   — ``hl.locus(contig, position, reference_genome=<rg>)``
    - ``tinterval`` — ``hl.interval(start, end, includes_start, includes_end)``
    - ``tcall``    — not reconstructed (warns and returns as-is; see note below)
    - ``tset``     — ``hl.set(array_expr)`` with element reconstruction
    - ``ttuple``   — ``hl.tuple([expr['_0'], expr['_1'], ...])``
    - ``tstruct``  — recursively reconstructs each field
    - ``tarray``   — maps reconstruction over elements
    - ``tdict``    — maps reconstruction over key/value pairs

    Note on ``tcall``: Hail's ``expand_types`` encodes calls as a bit-packed
    ``int32`` using an internal format.  There is no public API to reconstruct
    an arbitrary call from its int32 representation; callers that need call
    fields should reconstruct them manually (e.g. with ``hl.call(a0, a1)`` or
    ``hl.parse_call(hl.str(expr))``).

    Args:
        expr: A Hail expression produced after ``hl.Table.from_spark()``.
        hail_type: The original Hail type of this expression before
            ``expand_types`` transformed it.

    Returns:
        A Hail expression with the original type restored (or *expr* unchanged
        for unsupported types such as ``tcall``).
    """
    if isinstance(hail_type, htypes.tlocus):
        return hl.locus(
            expr.contig,
            expr.position,
            reference_genome=hail_type.reference_genome.name,
        )

    if isinstance(hail_type, htypes.tinterval):
        point_type = hail_type.point_type
        start = _reconstruct_expr(expr.start, point_type)
        end = _reconstruct_expr(expr.end, point_type)
        return hl.interval(start, end, includes_start=expr.includes_start, includes_end=expr.includes_end)

    if isinstance(hail_type, htypes.tcall):
        log.warning(
            'A tcall column cannot be automatically reconstructed from its int32 '
            'representation after a round-trip through Polars.  The column is left '
            'as int32.  Reconstruct manually, e.g. hl.call(a0, a1) or '
            'hl.parse_call(hl.str(expr)).'
        )
        return expr

    if isinstance(hail_type, htypes.tset):
        elem_type = hail_type.element_type
        if _needs_reconstruction(elem_type):
            arr = hl.map(lambda x: _reconstruct_expr(x, elem_type), expr)
        else:
            arr = expr
        return hl.set(arr)

    if isinstance(hail_type, htypes.ttuple):
        elements = [
            _reconstruct_expr(expr[f'_{i}'], t)
            for i, t in enumerate(hail_type.types)
        ]
        return hl.tuple(elements)

    if isinstance(hail_type, htypes.tstruct):
        if not _needs_reconstruction(hail_type):
            return expr
        return hl.struct(**{
            f: _reconstruct_expr(expr[f], t)
            for f, t in zip(hail_type.fields, hail_type.types)
        })

    if isinstance(hail_type, htypes.tarray):
        elem_type = hail_type.element_type
        if not _needs_reconstruction(elem_type):
            return expr
        return hl.map(lambda x: _reconstruct_expr(x, elem_type), expr)

    if isinstance(hail_type, htypes.tdict):
        key_type = hail_type.key_type
        val_type = hail_type.value_type
        if not (_needs_reconstruction(key_type) or _needs_reconstruction(val_type)):
            return expr
        # expand_types encodes tdict<K,V> as tarray<tstruct(key: K', value: V')>
        return hl.dict(hl.map(
            lambda kv: hl.tuple([
                _reconstruct_expr(kv.key, key_type),
                _reconstruct_expr(kv.value, val_type),
            ]),
            expr,
        ))

    return expr


def _apply_schema_reconstruction(ht: hl.Table, row_schema: htypes.tstruct) -> hl.Table:
    """Reconstruct Hail-native types in *ht* using the original *row_schema*.

    Iterates over every field in *row_schema* that requires reconstruction.
    Fields present in *row_schema* but absent from *ht* (e.g. dropped in
    Polars) are skipped with a warning.  Fields present in *ht* but absent
    from *row_schema* (e.g. added in Polars) are left as-is — Hail's type
    inference from Arrow/parquet is correct for Polars-native types.

    Args:
        ht: Hail Table immediately after :func:`hl.Table.from_spark` — all
            fields are unkeyed row-value fields at this point.
        row_schema: The original ``ht.row.dtype`` captured by :func:`to_polars`.

    Returns:
        *ht* with Hail-native type columns restored to their original types.
    """
    current_fields = set(ht.row)
    annotations = {}

    for fname, ftype in zip(row_schema.fields, row_schema.types):
        if fname not in current_fields:
            log.warning(
                'Field %r is in row_schema but not in the current table '
                '(was it dropped or renamed in Polars?); skipping reconstruction.',
                fname,
            )
            continue
        if _needs_reconstruction(ftype):
            annotations[fname] = _reconstruct_expr(ht[fname], ftype)

    if annotations:
        ht = ht.annotate(**annotations)
    return ht


# ---------------------------------------------------------------------------
# Hail → Polars conversion paths
# ---------------------------------------------------------------------------


def _to_polars_via_parquet(ht: hl.Table, tmp_dir: str | None = None) -> pl.DataFrame:
    """Convert a Hail Table to a Polars DataFrame via a local parquet round-trip.

    Most robust conversion path.  ``to_spark(flatten=False)`` invokes Hail's
    ``expand_types`` step internally, rewriting tuples, sets, locus, and
    interval fields into structs and arrays that parquet can store losslessly.
    Polars then reads the resulting parquet files directly with full
    nested-type support.

    The output parquet is coalesced to at most eight shards to avoid the
    overhead of reading many tiny files on a single VM, while still allowing
    some write parallelism.

    Args:
        ht: Source Hail Table.
        tmp_dir: Parent directory for the temporary parquet output.  Defaults
            to the system temporary directory.  The directory and all its
            contents are deleted after Polars finishes reading.

    Returns:
        Polars DataFrame containing all rows of *ht*.
    """
    sdf = ht.to_spark(flatten=False)

    tmp = tempfile.mkdtemp(prefix='ht2pl_', dir=tmp_dir)
    out_path = os.path.join(tmp, 'data.parquet')
    try:
        n_part = sdf.rdd.getNumPartitions()
        sdf.coalesce(max(1, min(8, n_part))).write.mode('overwrite').parquet(out_path)
        return pl.read_parquet(os.path.join(out_path, '*.parquet'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _to_polars_via_arrow(ht: hl.Table) -> pl.DataFrame:
    """Convert a Hail Table to a Polars DataFrame via Spark Arrow and pandas.

    Fastest path for small, fully-flat tables.  Enables PySpark's Arrow
    optimisation and converts the resulting pandas DataFrame to Polars.

    Arrow fallback is left enabled so that unsupported types do not hard-error
    at runtime; disable ``spark.sql.execution.arrow.pyspark.fallback.enabled``
    locally if you want to detect Arrow misses during debugging.

    Warning:
        Not safe for nested schemas.  When Arrow can't handle a column type,
        PySpark silently falls back to a slow row-based conversion, negating
        the performance benefit and potentially producing incorrect results for
        complex types.

    Args:
        ht: Source Hail Table.  Must have a fully-flat row schema (no struct,
            array, set, dict, tuple, locus, or interval fields).

    Returns:
        Polars DataFrame containing all rows of *ht*.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    spark.conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
    spark.conf.set('spark.sql.execution.arrow.pyspark.fallback.enabled', 'true')

    pdf = ht.to_pandas(flatten=False)
    return pl.from_pandas(pdf)


# ---------------------------------------------------------------------------
# Polars → Hail conversion paths
# ---------------------------------------------------------------------------


def _to_hail_via_parquet(pt: PolarsTable, tmp_dir: str | None = None) -> hl.Table:
    """Convert the DataFrame in *pt* to an unkeyed Hail Table via parquet.

    Polars writes a single parquet file; Spark reads it back and
    :func:`hl.Table.from_spark` converts the resulting DataFrame.  Handles
    nested Polars types (List, Struct) correctly.

    Args:
        pt: Source :class:`PolarsTable`.
        tmp_dir: Parent directory for the temporary parquet file.  Defaults to
            the system temporary directory.  Cleaned up automatically.

    Returns:
        Unkeyed Hail Table (key and globals are not yet restored).
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    tmp = tempfile.mkdtemp(prefix='pl2ht_', dir=tmp_dir)
    out_path = os.path.join(tmp, 'data.parquet')
    try:
        pt.df.write_parquet(out_path)
        sdf = spark.read.parquet(out_path)
        return hl.Table.from_spark(sdf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _to_hail_via_arrow(pt: PolarsTable) -> hl.Table:
    """Convert the DataFrame in *pt* to an unkeyed Hail Table via Arrow (no disk I/O).

    Polars converts to a PyArrow Table; Spark ingests it zero-copy via
    ``createDataFrame``; :func:`hl.Table.from_spark` converts the result.
    Fastest path for small, fully-flat DataFrames.

    Warning:
        Not safe for nested Polars schemas (List, Struct columns).  Use the
        parquet path for those.

    Args:
        pt: Source :class:`PolarsTable`.

    Returns:
        Unkeyed Hail Table (key and globals are not yet restored).
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    arrow_table = pt.df.to_arrow()
    sdf = spark.createDataFrame(arrow_table)
    return hl.Table.from_spark(sdf)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_polars(
    ht: hl.Table,
    flatten: bool = False,
    prefer: str = 'auto',
    tmp_dir: str | None = None,
) -> PolarsTable:
    """Convert a Hail Table to a :class:`PolarsTable` (DataFrame + round-trip metadata).

    Chooses the fastest conversion path that is safe for the table's row
    schema.  For tables produced by long upstream pipelines, checkpoint first
    to avoid re-running the pipeline on every call::

        ht = ht.checkpoint('/tmp/foo.ht', overwrite=True)
        pt = to_polars(ht)

    Examples::

        pt = to_polars(ancestry_sample_qc)   # nested → parquet, auto
        pt.df                                # pl.DataFrame
        pt.globals                           # {'hard_filter_cutoffs': {...}}

        # Force a specific path:
        pt = to_polars(ht, prefer='parquet')
        pt = to_polars(ht, prefer='arrow', flatten=True)

        # Access nested fields in Polars:
        pt.df['sample_qc'].struct.field('n_het')

        # Round-trip back to Hail:
        ht2 = from_polars(pt)

    Args:
        ht: Hail Table.  Should comfortably fit in driver memory once
            materialised.
        flatten: When True, run ``ht.flatten()`` before conversion, producing
            a flat DataFrame with ``'.'`` in column names.  Default is False
            because Polars handles Struct and List types natively, and the
            nested form is more ergonomic for downstream code.

            Note: when True, ``row_schema`` is set to ``None`` in the
            returned :class:`PolarsTable` because the flattened column layout
            is incompatible with schema-guided type reconstruction.  The
            round-trip via :func:`from_polars` will still work but Hail-native
            types will not be restored.
        prefer: Conversion strategy.  One of:

            - ``'auto'`` (default) — uses parquet when any nested type is
              present in the row schema, else arrow.
            - ``'parquet'`` — force the parquet round-trip.  Most robust;
              handles all schema types.
            - ``'arrow'`` — force the Spark Arrow → pandas → Polars path.
              Fastest for small fully-flat tables; risky for nested schemas.

        tmp_dir: Directory for the parquet intermediate (parquet path only).
            Defaults to the system temp dir.  Cleaned up automatically.

    Returns:
        :class:`PolarsTable` with:

        - ``.df`` — the row data as a ``pl.DataFrame``.
        - ``.globals`` — dict of the source table's global fields.
        - ``.row_schema`` — original ``ht.row.dtype``; used by
          :func:`from_polars` for type reconstruction.  ``None`` when
          ``flatten=True``.
        - ``.keys`` — original key field names; used by :func:`from_polars`
          to restore ``ht.key_by(*keys)``.

    Raises:
        ValueError: If *prefer* is not one of ``'auto'``, ``'parquet'``, or
            ``'arrow'``.
    """
    globals_struct = hl.eval(ht.globals)
    globals_dict = _hail_to_python(globals_struct) if globals_struct is not None else {}

    # Capture schema and keys before flatten changes the layout.
    row_schema: htypes.tstruct | None = ht.row.dtype
    keys: list[str] = list(ht.key.dtype.fields)

    if flatten:
        ht = ht.flatten()
        # Flattened column names (with '.') don't match row_schema field names.
        row_schema = None

    if prefer == 'auto':
        prefer = 'parquet' if _has_nested_types(ht.row.dtype) else 'arrow'
    elif prefer not in ('parquet', 'arrow'):
        raise ValueError(f"prefer must be 'auto', 'parquet', or 'arrow'; got {prefer!r}")

    log.info('to_polars: using %s path (flatten=%s)', prefer, flatten)

    if prefer == 'parquet':
        df = _to_polars_via_parquet(ht, tmp_dir=tmp_dir)
    else:
        df = _to_polars_via_arrow(ht)

    return PolarsTable(df=df, globals=globals_dict, row_schema=row_schema, keys=keys)


def from_polars(
    pt: PolarsTable,
    prefer: str = 'auto',
    key: list[str] | None = None,
    tmp_dir: str | None = None,
) -> hl.Table:
    """Convert a :class:`PolarsTable` back to a Hail Table.

    Mirrors :func:`to_polars` in reverse.  When *pt* was produced by
    :func:`to_polars`, the round-trip is lossless for all Hail types except
    ``tcall`` (see below)::

        pt  = to_polars(ht)
        # ... analyse or mutate pt.df in Polars ...
        ht2 = from_polars(pt)

    Type reconstruction
    -------------------
    Hail-native types are restored from ``pt.row_schema`` after ingestion:

    - ``tlocus<GRCh38>`` — reconstructed via ``hl.locus(contig, position, 'GRCh38')``
    - ``tinterval<tlocus>`` — reconstructed recursively (start, end, flags)
    - ``tset<X>`` — reconstructed from the ``tarray<X>`` produced by expand_types
    - ``ttuple<A, B>`` — reconstructed from the ``tstruct(_0: A, _1: B)`` form
    - ``tcall`` — **not reconstructed**; left as ``int32`` with a warning.
      Reconstruct manually, e.g. ``hl.parse_call(hl.str(ht.GT))``.

    If ``pt.row_schema`` is ``None`` (e.g. ``flatten=True`` was used, or the
    :class:`PolarsTable` was constructed manually), reconstruction is skipped
    and a warning is emitted.

    New or renamed columns
    ----------------------
    Columns present in ``pt.df`` but absent from ``pt.row_schema`` (new
    columns added in Polars) are left with the type that Hail infers from the
    Arrow/parquet schema — this is correct for Polars-native types.

    If you **renamed** a column that originally held a Hail-native type, the
    stored schema will not match by name and that column will not be
    reconstructed.  Pass ``key`` explicitly if the key field was renamed.

    Examples::

        # Basic round-trip
        ht2 = from_polars(pt)

        # Override the key (e.g. after renaming the locus column)
        ht2 = from_polars(pt, key=['variant_locus', 'alleles'])

        # Force parquet path regardless of schema
        ht2 = from_polars(pt, prefer='parquet')

    Args:
        pt: Source :class:`PolarsTable`, normally produced by :func:`to_polars`.
        prefer: Ingestion strategy.  One of:

            - ``'auto'`` (default) — uses parquet when any List, Array, or
              Struct column is present in ``pt.df``, else arrow.
            - ``'parquet'`` — force the Polars → parquet → Spark → Hail path.
              Most robust; handles all Polars column types.
            - ``'arrow'`` — force the Polars → Arrow → Spark → Hail path.
              Fastest for small fully-flat DataFrames; unsafe for nested types.

        key: Key field names to use for ``ht.key_by(*key)``.  Overrides
            ``pt.keys`` when provided.  Pass an empty list ``[]`` to produce
            an unkeyed table even if ``pt.keys`` is non-empty.
        tmp_dir: Directory for the parquet intermediate (parquet path only).
            Defaults to the system temp dir.  Cleaned up automatically.

    Returns:
        Hail Table with the row data from ``pt.df``, Hail-native types
        reconstructed where possible, the original row key restored, and
        globals re-attached.

    Raises:
        ValueError: If *prefer* is not one of ``'auto'``, ``'parquet'``, or
            ``'arrow'``.
    """
    if prefer not in ('auto', 'parquet', 'arrow'):
        raise ValueError(f"prefer must be 'auto', 'parquet', or 'arrow'; got {prefer!r}")

    resolved_prefer = prefer
    if prefer == 'auto':
        resolved_prefer = 'parquet' if _polars_schema_is_nested(pt.df) else 'arrow'

    log.info('from_polars: using %s path', resolved_prefer)

    if resolved_prefer == 'parquet':
        ht = _to_hail_via_parquet(pt, tmp_dir=tmp_dir)
    else:
        ht = _to_hail_via_arrow(pt)

    # Reconstruct Hail-native types from the stored schema.
    if pt.row_schema is not None:
        ht = _apply_schema_reconstruction(ht, pt.row_schema)
    else:
        log.warning(
            'from_polars: pt.row_schema is None — Hail-native types (tlocus, '
            'tinterval, tset, ttuple, tcall) will not be reconstructed.  '
            'Produce PolarsTable via to_polars() to enable round-trip type '
            'fidelity, or set pt.row_schema manually.'
        )

    # Restore the row key.
    resolved_key = key if key is not None else pt.keys
    if resolved_key:
        ht = ht.key_by(*resolved_key)

    # Re-attach globals.
    if pt.globals:
        ht = ht.annotate_globals(**pt.globals)

    return ht


# ---------------------------------------------------------------------------
# Interactive PCA plot
# ---------------------------------------------------------------------------

# Default 10-colour palette for layers without an explicit color_map.
_DEFAULT_PALETTE = (
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
)


@dataclass
class PCALayer:
    """One cohort to draw on a PCA plot.

    Args:
        name: Legend prefix, e.g. 'Ref' or 'OurDNA'. Each group within the
            layer becomes a separate trace named '{name}: {group}'.
        group_col: Column whose distinct values define the per-group traces.
        mask: Optional boolean Series selecting which rows of the input
            DataFrame belong to this layer. None means "all rows".
        color_map: Mapping group label -> hex colour. If None, groups are
            assigned colours from the default palette in sort order.
        marker_symbol: Plotly marker symbol ('circle', 'cross', 'diamond', …).
        marker_alpha: Marker opacity in [0, 1].
        marker_size: Marker size in pixels.
        marker_line_width: Outline width in pixels. 0 disables the outline.
        draw_ellipses: If True, also draw axis-aligned ±N·SD ellipses per
            group (median ± N·std on each PC axis). Only the SD selected by
            the dropdown is visible at any time.
        sample_col: Column treated as the primary point identifier — shown
            bolded as the first line of the hover tooltip.
        hover_cols: Extra columns to include in the hover tooltip, as a
            mapping column -> display label. Use {} for none.
    """
    name: str
    group_col: str
    mask: pd.Series | None = None
    color_map: dict[str, str] | None = None
    marker_symbol: str = 'circle'
    marker_alpha: float = 0.5
    marker_size: int = 6
    marker_line_width: int = 0
    draw_ellipses: bool = False
    sample_col: str = 'sample'
    hover_cols: dict[str, str] = field(default_factory=dict)


def plot_pca(
    df: pd.DataFrame | pl.DataFrame,
    layers: Sequence[PCALayer],
    *,
    pc_prefix: str = 'PC',
    n_pcs: int | None = None,
    sd_thresholds: Sequence[int] = (1, 2, 3, 4, 5, 6),
    default_sd: int | None = None,
    title: str = 'PCA projection',
    height: int = 800,
    output_html: str | None = None,
) -> go.Figure:
    """Build an interactive plotly PCA scatter with X / Y / SD dropdowns.

    Renders one trace per (layer, group) and, for layers with draw_ellipses
    set, one axis-aligned ±N·SD ellipse per (layer, group, SD) — only the SD
    selected via the dropdown is visible at a time. The dropdowns relabel the
    axes by remapping the trace data, so the same figure can be flipped
    between any two PCs without rebuilding.

    Args:
        df: Pandas or polars DataFrame. Must contain `{pc_prefix}{i}` columns
            for i in 1..n_pcs. Polars input is converted to pandas internally.
        layers: One PCALayer per cohort. Layers are drawn in order; later
            layers paint on top.
        pc_prefix: Prefix for PC column names. 'PC' -> 'PC1', 'PC2', …
        n_pcs: How many PCs to expose in the dropdowns. If None, auto-detect
            from the columns. The X dropdown ranges over PC1..PCn and the Y
            dropdown over PC2..PCn.
        sd_thresholds: SD values to precompute ellipses for. Only consulted
            for layers with draw_ellipses=True.
        default_sd: SD value visible on initial render. None ('Off') hides
            all ellipses until the user picks one from the dropdown.
        title: Figure title.
        height: Figure height in pixels.
        output_html: If given, also write the figure to a standalone HTML
            file at this path.

    Returns:
        A `plotly.graph_objects.Figure`. Call `.show()` in a notebook to
        render it; the dropdowns are baked into the figure's own JSON, so the
        exported HTML is fully interactive without a Python kernel.
    """
    pdf = _to_pandas(df)

    if n_pcs is None:
        n_pcs = _detect_n_pcs(pdf, pc_prefix)
    if n_pcs < 2:
        raise ValueError(
            f'plot_pca: need at least 2 {pc_prefix}* columns; '
            f'found {n_pcs} starting from {pc_prefix}1',
        )

    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 100)

    # trace_recipes parallels fig.data: one entry per trace, in order.
    # Used later to rebuild x/y arrays for each PC dropdown option.
    trace_recipes: list[dict[str, Any]] = []

    for layer in layers:
        rows = pdf if layer.mask is None else pdf[layer.mask]
        groups = list(rows[layer.group_col].dropna().unique())
        cmap = _resolve_color_map(layer.color_map, groups)

        for group in groups:
            sub = rows[rows[layer.group_col] == group]
            if sub.empty:
                continue

            color = cmap.get(group, _DEFAULT_PALETTE[0])
            customdata, hovertemplate = _build_hover(sub, layer)

            marker: dict[str, Any] = dict(
                symbol=layer.marker_symbol,
                size=layer.marker_size,
                color=color,
                opacity=layer.marker_alpha,
            )
            if layer.marker_line_width > 0:
                marker['line'] = dict(width=layer.marker_line_width, color='black')

            trace_recipes.append({'kind': 'scatter', 'sub': sub})
            fig.add_trace(go.Scattergl(
                x=sub[f'{pc_prefix}1'],
                y=sub[f'{pc_prefix}2'],
                mode='markers',
                name=f'{layer.name}: {group}',
                marker=marker,
                customdata=customdata,
                hovertemplate=hovertemplate,
            ))

            if not layer.draw_ellipses:
                continue

            mu_x = sub[f'{pc_prefix}1'].median()
            sig_x = sub[f'{pc_prefix}1'].std()
            mu_y = sub[f'{pc_prefix}2'].median()
            sig_y = sub[f'{pc_prefix}2'].std()
            for sd in sd_thresholds:
                is_active = sd == default_sd
                trace_recipes.append({'kind': 'ellipse', 'sub': sub, 'sd': sd})
                fig.add_trace(go.Scatter(
                    x=mu_x + sd * sig_x * np.cos(theta),
                    y=mu_y + sd * sig_y * np.sin(theta),
                    mode='lines',
                    name=f'{layer.name}: {group} bounds',
                    legendgroup=f'{layer.name}_ellipse_{group}',
                    visible=is_active,
                    showlegend=is_active,
                    hoverinfo='skip',
                    line=dict(color=color, width=2, dash='dot' if sd <= 3 else 'dash'),
                ))

    x_buttons = _pc_axis_buttons(trace_recipes, pc_prefix, theta, range(1, n_pcs + 1), axis='x')
    y_buttons = _pc_axis_buttons(trace_recipes, pc_prefix, theta, range(2, n_pcs + 1), axis='y')
    sd_buttons = _sd_buttons(trace_recipes, sd_thresholds)

    axis_style = dict(
        showgrid=True, gridcolor='black', griddash='dash',
        zeroline=True, zerolinecolor='black', zerolinewidth=1.5,
        linecolor='black',
    )
    fig.update_layout(
        title=title,
        legend_title='Cohorts & bounds (click to toggle)',
        height=height,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(title_text=f'{pc_prefix}1', **axis_style),
        yaxis=dict(title_text=f'{pc_prefix}2', **axis_style),
        updatemenus=[
            dict(buttons=x_buttons, direction='down', pad={'r': 10, 't': 10},
                 showactive=True, x=0.08, xanchor='left', y=1.1, yanchor='top'),
            dict(buttons=y_buttons, direction='down', pad={'r': 10, 't': 10},
                 showactive=True, x=0.25, xanchor='left', y=1.1, yanchor='top'),
            dict(buttons=sd_buttons, direction='down', pad={'r': 10, 't': 10},
                 showactive=True, x=0.45, xanchor='left', y=1.1, yanchor='top'),
        ],
        annotations=[
            dict(text='X-Axis:', x=0.04, y=1.08, xref='paper', yref='paper', showarrow=False),
            dict(text='Y-Axis:', x=0.21, y=1.08, xref='paper', yref='paper', showarrow=False),
            dict(text='Ellipses:', x=0.40, y=1.08, xref='paper', yref='paper', showarrow=False),
        ],
    )

    if output_html:
        fig.write_html(output_html)

    return fig


def _to_pandas(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(
        f'plot_pca: expected pandas or polars DataFrame, got {type(df).__name__}',
    )


def _detect_n_pcs(pdf: pd.DataFrame, pc_prefix: str) -> int:
    n = 0
    while f'{pc_prefix}{n + 1}' in pdf.columns:
        n += 1
    return n


def _resolve_color_map(
    color_map: dict[str, str] | None,
    groups: Sequence[str],
) -> dict[str, str]:
    if color_map is not None:
        return color_map
    return {
        g: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
        for i, g in enumerate(sorted(groups))
    }


def _build_hover(sub: pd.DataFrame, layer: PCALayer):
    """Return (customdata, hovertemplate) for a single (layer, group) trace."""
    cols: list[str] = []
    labels: list[str] = []
    if layer.sample_col in sub.columns:
        cols.append(layer.sample_col)
        labels.append('')  # rendered bold without a label
    for c, lbl in layer.hover_cols.items():
        if c in sub.columns:
            cols.append(c)
            labels.append(lbl)

    if not cols:
        return None, 'X: %{x:.3f} | Y: %{y:.3f}<extra></extra>'

    customdata = sub[cols].values
    lines: list[str] = []
    for i, lbl in enumerate(labels):
        token = f'%{{customdata[{i}]}}'
        if i == 0 and lbl == '':
            lines.append(f'<b>{token}</b>')
        elif lbl == '':
            lines.append(token)
        else:
            lines.append(f'{lbl}: {token}')
    lines.append('X: %{x:.3f} | Y: %{y:.3f}')
    return customdata, '<br>'.join(lines) + '<extra></extra>'


def _pc_axis_buttons(
    trace_recipes: Sequence[dict[str, Any]],
    pc_prefix: str,
    theta: np.ndarray,
    pc_range: range,
    *,
    axis: str,
) -> list[dict[str, Any]]:
    """Build dropdown buttons that swap the X or Y axis to a different PC."""
    buttons = []
    trig = np.cos(theta) if axis == 'x' else np.sin(theta)
    for i in pc_range:
        pc_col = f'{pc_prefix}{i}'
        data = []
        for recipe in trace_recipes:
            sub = recipe['sub']
            if recipe['kind'] == 'scatter':
                data.append(sub[pc_col].values)
            else:  # ellipse
                mu = sub[pc_col].median()
                sig = sub[pc_col].std()
                data.append(mu + recipe['sd'] * sig * trig)
        buttons.append(dict(
            label=pc_col,
            method='update',
            args=[{axis: data}, {f'{axis}axis.title.text': pc_col}],
        ))
    return buttons


def _sd_buttons(
    trace_recipes: Sequence[dict[str, Any]],
    sd_thresholds: Sequence[int],
) -> list[dict[str, Any]]:
    """Build dropdown buttons that show one SD ellipse (or Off) at a time."""
    buttons = []
    for target_sd in [None, *sd_thresholds]:
        vis, show_leg = [], []
        for recipe in trace_recipes:
            if recipe['kind'] == 'scatter':
                vis.append(True)
                show_leg.append(True)
            else:
                is_active = recipe['sd'] == target_sd
                vis.append(is_active)
                show_leg.append(is_active)
        label = f'{target_sd} SD' if target_sd is not None else 'Off'
        buttons.append(dict(
            label=label,
            method='restyle',
            args=[{'visible': vis, 'showlegend': show_leg}],
        ))
    return buttons
