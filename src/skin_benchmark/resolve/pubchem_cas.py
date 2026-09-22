"""CAS-to-SMILES resolution using PubChemPy with a reproducible cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import logging
import re

import pandas as pd
import pubchempy as pcp

from skin_benchmark.utils.io import save_csv
from skin_benchmark.utils.text import append_note, clean_text, normalize_cas

CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


@dataclass(slots=True)
class CasResolutionRecord:
    """A cached CAS resolution entry."""

    cas_number_raw: str | None
    cas_number_normalized: str | None
    pubchem_status: str
    pubchem_cid: str | None
    pubchem_smiles_raw: str | None
    resolver_note: str | None


MISSING_CAS_STATUS = "missing"
MISSING_CAS_NOTE = "CAS was missing in the Cheruvu source row, so PubChem resolution was not attempted."


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, dtype=object).fillna(pd.NA)
    return pd.DataFrame(
        columns=[
            "cas_number_raw",
            "cas_number_normalized",
            "pubchem_status",
            "pubchem_cid",
            "pubchem_smiles_raw",
            "resolver_note",
        ]
    )


def _save_cache(cache_df: pd.DataFrame, cache_path: Path) -> None:
    deduped = cache_df.drop_duplicates(subset=["cas_number_normalized"], keep="last").sort_values(
        by=["cas_number_normalized"], na_position="last"
    )
    save_csv(deduped, cache_path)


def _normalize_synonym(value: str) -> str:
    return normalize_cas(value) or ""


def _candidate_has_exact_cas(cid: int, normalized_cas: str) -> bool:
    synonyms = pcp.get_synonyms(cid, namespace="cid")
    for entry in synonyms:
        for synonym in entry.get("Synonym", []):
            if _normalize_synonym(synonym) == normalized_cas:
                return True
    return False


def _get_smiles_from_compound(compound: pcp.Compound) -> str | None:
    for attribute in ("smiles", "connectivity_smiles", "isomeric_smiles", "canonical_smiles"):
        value = getattr(compound, attribute, None)
        if value:
            return str(value)
    return None


def _query_pubchem(normalized_cas: str) -> CasResolutionRecord:
    if not normalized_cas or not CAS_PATTERN.match(normalized_cas):
        return CasResolutionRecord(
            cas_number_raw=normalized_cas,
            cas_number_normalized=normalized_cas,
            pubchem_status="invalid_cas",
            pubchem_cid=None,
            pubchem_smiles_raw=None,
            resolver_note="CAS failed normalization or format validation.",
        )

    try:
        cids = pcp.get_cids(normalized_cas, namespace="name")
        if not cids:
            compounds = pcp.get_compounds(normalized_cas, namespace="name")
        else:
            compounds = pcp.get_compounds(cids, namespace="cid")
        if not compounds:
            return CasResolutionRecord(
                cas_number_raw=normalized_cas,
                cas_number_normalized=normalized_cas,
                pubchem_status="unresolved",
                pubchem_cid=None,
                pubchem_smiles_raw=None,
                resolver_note="PubChem returned no matching compound for the CAS query.",
            )

        exact_matches: list[pcp.Compound] = []
        for compound in compounds:
            cid = getattr(compound, "cid", None)
            if cid is None:
                continue
            if _candidate_has_exact_cas(int(cid), normalized_cas):
                exact_matches.append(compound)

        if len(exact_matches) == 1:
            compound = exact_matches[0]
            return CasResolutionRecord(
                cas_number_raw=normalized_cas,
                cas_number_normalized=normalized_cas,
                pubchem_status="resolved",
                pubchem_cid=str(compound.cid),
                pubchem_smiles_raw=_get_smiles_from_compound(compound),
                resolver_note="Resolved through PubChem exact CAS synonym match.",
            )
        if len(exact_matches) > 1:
            cids_text = ",".join(str(match.cid) for match in exact_matches if getattr(match, "cid", None))
            return CasResolutionRecord(
                cas_number_raw=normalized_cas,
                cas_number_normalized=normalized_cas,
                pubchem_status="ambiguous",
                pubchem_cid=cids_text or None,
                pubchem_smiles_raw=None,
                resolver_note="Multiple PubChem compounds matched the exact CAS synonym.",
            )
        return CasResolutionRecord(
            cas_number_raw=normalized_cas,
            cas_number_normalized=normalized_cas,
            pubchem_status="unresolved",
            pubchem_cid=None,
            pubchem_smiles_raw=None,
            resolver_note="PubChem returned candidates, but none carried an exact CAS synonym match.",
        )
    except Exception as exc:  # pragma: no cover - network errors are environment-specific.
        return CasResolutionRecord(
            cas_number_raw=normalized_cas,
            cas_number_normalized=normalized_cas,
            pubchem_status="query_error",
            pubchem_cid=None,
            pubchem_smiles_raw=None,
            resolver_note=f"{type(exc).__name__}: {exc}",
        )


def resolve_cheruvu_smiles(
    df: pd.DataFrame,
    cache_path: Path,
    logger: logging.Logger,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve missing Cheruvu structures via CAS lookup and persist a cache."""

    working = df.copy()
    cache_df = _load_cache(cache_path)
    if "cas_number_normalized" not in cache_df.columns:
        cache_df = _load_cache(Path("__missing__"))
    cache_map = {
        clean_text(row["cas_number_normalized"]): row
        for _, row in cache_df.iterrows()
        if clean_text(row.get("cas_number_normalized"))
    }

    working["cas_number_normalized"] = working["cas_number_raw"].map(normalize_cas)
    unique_cas = sorted({value for value in working["cas_number_normalized"].dropna().astype(str).tolist() if value})

    new_records: list[dict[str, str | None]] = []
    for cas in unique_cas:
        if cas in cache_map and not refresh_cache:
            continue
        record = _query_pubchem(cas)
        logger.info("Cheruvu CAS lookup %s -> %s", cas, record.pubchem_status)
        new_records.append(asdict(record))

    if new_records:
        cache_df = pd.concat([cache_df, pd.DataFrame(new_records)], ignore_index=True)
        _save_cache(cache_df, cache_path)
        cache_map = {
            clean_text(row["cas_number_normalized"]): row
            for _, row in cache_df.iterrows()
            if clean_text(row.get("cas_number_normalized"))
        }

    statuses: list[str | None] = []
    cids: list[str | None] = []
    notes: list[str | None] = []
    resolver_notes: list[str | None] = []
    smiles_values: list[str | None] = []
    for _, row in working.iterrows():
        normalized_cas = clean_text(row.get("cas_number_normalized"))
        if not normalized_cas:
            status = MISSING_CAS_STATUS
            cid = None
            resolved_smiles = None
            resolver_note = MISSING_CAS_NOTE
        else:
            cached_row = cache_map.get(normalized_cas)
            status = clean_text(cached_row.get("pubchem_status")) if cached_row is not None else None
            cid = clean_text(cached_row.get("pubchem_cid")) if cached_row is not None else None
            resolved_smiles = clean_text(cached_row.get("pubchem_smiles_raw")) if cached_row is not None else None
            resolver_note = clean_text(cached_row.get("resolver_note")) if cached_row is not None else None

        statuses.append(status)
        cids.append(cid)
        notes.append(resolver_note)
        resolver_notes.append(resolver_note)
        smiles_values.append(clean_text(row.get("smiles_raw")) or resolved_smiles)

    working["pubchem_status"] = statuses
    working["pubchem_cid"] = cids
    working["pubchem_resolver_note"] = resolver_notes
    working["smiles_raw"] = smiles_values
    working["notes"] = [
        append_note(existing_note, resolver_note)
        for existing_note, resolver_note in zip(working["notes"], notes, strict=False)
    ]
    return working, cache_df
