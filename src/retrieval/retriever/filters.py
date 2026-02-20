from __future__ import annotations
from typing import Dict, Any, Iterable, List, Optional, Sequence

# Simple metadata-based filtering utilities


def match_meta(
    meta: Dict[str, Any],
    ticker=None,
    year=None,
    form=None,
    entity_org: Optional[Sequence[str]] = None,
    entity_gpe: Optional[Sequence[str]] = None,
):
    """Filter by metadata fields, optionally including entity constraints."""
    if ticker and (str(meta.get("ticker") or "").upper() != str(ticker).upper()):
        return False
    if form and (str(meta.get("form") or "").upper() != str(form).upper()):
        return False
    # Year: try fy or year
    if year is not None:
        fy = meta.get("fy") or meta.get("year")
        try:
            if int(fy) != int(year):
                return False
        except Exception:
            return False
    # Entity-based filtering
    if entity_org:
        chunk_orgs = set(meta.get("entities_org", []))
        if not any(org.upper() in chunk_orgs for org in entity_org):
            return False
    if entity_gpe:
        chunk_gpes = set(meta.get("entities_gpe", []))
        if not any(gpe.upper() in chunk_gpes for gpe in entity_gpe):
            return False
    return True


def filter_hits(hits: Iterable[Dict[str, Any]], ticker=None, year=None, form=None,
                entity_org=None, entity_gpe=None) -> List[Dict[str, Any]]:
    return [h for h in hits if match_meta(h.get("meta", {}), ticker, year, form,
                                          entity_org=entity_org, entity_gpe=entity_gpe)]
