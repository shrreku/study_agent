"""Neo4j write helpers for pedagogy-oriented knowledge-graph edges."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional

from .base import canonicalize_concept, managed_driver
from .relationships import merge_alias, merge_prerequisite_edge


def merge_section_node(
    resource_id: str,
    section_path: Iterable[str] | None,
    title: str | None,
    number: str | None,
    level: int | None,
) -> Optional[str]:
    segments: List[str] = []
    if section_path:
        for seg in section_path:
            if seg is None:
                continue
            cleaned = str(seg).strip()
            if cleaned:
                segments.append(cleaned)
    if not segments and number:
        segments = [number.strip()]
    if not segments and title:
        segments = [title.strip()]
    if not segments:
        return None

    path_key = "|".join(seg.lower() for seg in segments)
    parent_segments = segments[:-1]
    parent_key = "|".join(seg.lower() for seg in parent_segments) if parent_segments else None

    with managed_driver() as driver:
        if driver is None:
            return path_key

        def _tx(tx):
            tx.run(
                """
                MERGE (sec:Section {resource_id: $resid, path_key: $path_key})
                ON CREATE SET sec.created_at = datetime()
                SET sec.path = $path,
                    sec.title = coalesce(sec.title, $title),
                    sec.number = coalesce(sec.number, $number),
                    sec.level = coalesce(sec.level, $level),
                    sec.updated_at = datetime()
                WITH sec
                FOREACH (_ IN CASE WHEN $parent_key IS NULL THEN [] ELSE [1] END |
                    MERGE (parent:Section {resource_id: $resid, path_key: $parent_key})
                    ON CREATE SET parent.created_at = datetime()
                    SET parent.path = $parent_path,
                        parent.title = coalesce(parent.title, $parent_title),
                        parent.level = coalesce(parent.level, CASE WHEN $level IS NULL THEN NULL ELSE $level - 1 END),
                        parent.updated_at = datetime()
                    MERGE (sec)-[:SUBSECTION_OF]->(parent)
                )
                """,
                resid=resource_id,
                path_key=path_key,
                path=segments,
                title=title,
                number=number,
                level=level,
                parent_key=parent_key,
                parent_path=parent_segments,
                parent_title=parent_segments[-1] if parent_segments else None,
            )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_merge_section_failed", extra={"resource_id": resource_id, "path_key": path_key})

    return path_key


def link_chunk_to_section(
    chunk_id: str,
    resource_id: str,
    section_path: Iterable[str] | None,
    title: str | None,
    number: str | None,
    level: int | None,
) -> None:
    path_key = merge_section_node(resource_id, section_path, title, number, level)
    if not path_key:
        return

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            tx.run(
                """
                MERGE (ch:Chunk {id: $chunk_id})
                MERGE (sec:Section {resource_id: $resid, path_key: $path_key})
                MERGE (ch)-[:IN_SECTION]->(sec)
                """,
                chunk_id=chunk_id,
                resid=resource_id,
                path_key=path_key,
            )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_link_chunk_section_failed", extra={"chunk_id": chunk_id, "path_key": path_key})


def merge_chunk_figures(
    chunk_id: str,
    resource_id: str,
    figure_labels: Iterable[str] | None,
    concept_canonicals: Iterable[str] | None = None,
) -> None:
    if not figure_labels:
        return

    labels = [str(label).strip() for label in figure_labels if str(label or "").strip()]
    if not labels:
        return

    concepts = [c for c in (concept_canonicals or []) if c]

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            for label in labels:
                tx.run(
                    """
                    MERGE (fig:Figure {resource_id: $resid, label: $label})
                    ON CREATE SET fig.created_at = datetime()
                    SET fig.updated_at = datetime()
                    MERGE (ch:Chunk {id: $chunk_id})
                    MERGE (ch)-[:HAS_FIGURE]->(fig)
                    WITH fig
                    FOREACH (concept IN $concepts |
                        MERGE (c:Concept {canonical_name: concept})
                        SET c.last_seen = datetime()
                        MERGE (fig)-[:ILLUSTRATES]->(c)
                    )
                    """,
                    resid=resource_id,
                    label=label,
                    chunk_id=chunk_id,
                    concepts=concepts[:3],
                )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_merge_figures_failed", extra={"chunk_id": chunk_id, "labels": labels})


def merge_chunk_formulas(
    chunk_id: str,
    resource_id: str,
    formulas: Iterable[str] | None,
    concept_canonicals: Iterable[str] | None = None,
) -> None:
    if not formulas:
        return

    normalized = [str(f).strip() for f in formulas if str(f or "").strip()]
    if not normalized:
        return

    concepts = [c for c in (concept_canonicals or []) if c]

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            for formula in normalized:
                formula_id = hashlib.sha256(f"{resource_id}:{formula}".encode("utf-8")).hexdigest()
                tx.run(
                    """
                    MERGE (form:Formula {formula_id: $formula_id})
                    ON CREATE SET form.created_at = datetime(), form.resource_id = $resid
                    SET form.latex = coalesce(form.latex, $formula),
                        form.updated_at = datetime()
                    MERGE (ch:Chunk {id: $chunk_id})
                    MERGE (ch)-[:HAS_FORMULA]->(form)
                    WITH form
                    FOREACH (concept IN $concepts |
                        MERGE (c:Concept {canonical_name: concept})
                        SET c.last_seen = datetime()
                        MERGE (form)-[:ABOUT]->(c)
                    )
                    """,
                    formula_id=formula_id,
                    formula=formula,
                    resid=resource_id,
                    chunk_id=chunk_id,
                    concepts=concepts[:3],
                )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception("neo4j_merge_formulas_failed", extra={"chunk_id": chunk_id, "formulas": normalized})


def merge_chunk_formulas_enhanced(
    chunk_id: str,
    resource_id: str,
    formulas: List[Dict[str, Any]],
    concept_canonicals: Iterable[str] | None = None,
) -> None:
    """Enhanced formula merging that creates Variable nodes from INGEST-04 metadata.
    
    Args:
        chunk_id: Chunk ID containing the formulas
        resource_id: Resource ID
        formulas: List of formula dicts with structure:
            {
                'latex': str,
                'type': str,
                'variables': [
                    {'symbol': str, 'meaning': str, 'units': str, ...}
                ]
            }
        concept_canonicals: Optional list of concept canonical names to link
    """
    if not formulas:
        return
    
    concepts = [c for c in (concept_canonicals or []) if c]
    
    with managed_driver() as driver:
        if driver is None:
            return
        
        def _tx(tx):
            for formula_data in formulas:
                # Extract formula fields
                latex = formula_data.get('latex', '')
                if not latex:
                    continue
                
                formula_type = formula_data.get('type', 'unknown')
                formula_id = hashlib.sha256(f"{resource_id}:{latex}".encode("utf-8")).hexdigest()
                variables = formula_data.get('variables', [])
                
                # Create Formula node with enhanced metadata
                tx.run(
                    """
                    MERGE (form:Formula {formula_id: $formula_id})
                    ON CREATE SET 
                        form.created_at = datetime(), 
                        form.resource_id = $resid
                    SET 
                        form.latex = coalesce(form.latex, $latex),
                        form.formula_type = $formula_type,
                        form.updated_at = datetime()
                    
                    // Link to chunk
                    MERGE (ch:Chunk {id: $chunk_id})
                    MERGE (ch)-[:HAS_FORMULA]->(form)
                    
                    // Link to concepts if provided
                    WITH form
                    FOREACH (concept IN $concepts |
                        MERGE (c:Concept {canonical_name: concept})
                        SET c.last_seen = datetime()
                        MERGE (form)-[:ABOUT]->(c)
                    )
                    """,
                    formula_id=formula_id,
                    latex=latex,
                    formula_type=formula_type,
                    resid=resource_id,
                    chunk_id=chunk_id,
                    concepts=concepts[:3],
                )
                
                # Create Variable nodes
                for var_data in variables:
                    symbol = var_data.get('symbol', '')
                    if not symbol:
                        continue
                    
                    meaning = var_data.get('meaning', '')
                    units = var_data.get('units', '')
                    role = var_data.get('role', '')
                    
                    # Create unique variable ID (symbol + formula)
                    var_id = hashlib.sha256(f"{formula_id}:{symbol}".encode("utf-8")).hexdigest()[:16]
                    
                    tx.run(
                        """
                        MERGE (v:Variable {variable_id: $var_id, formula_id: $formula_id})
                        ON CREATE SET v.created_at = datetime()
                        SET 
                            v.symbol = $symbol,
                            v.meaning = $meaning,
                            v.units = $units,
                            v.role = $role,
                            v.updated_at = datetime()
                        
                        // Link to formula
                        WITH v
                        MATCH (form:Formula {formula_id: $formula_id})
                        MERGE (form)-[:HAS_VARIABLE]->(v)
                        
                        // Link to concept if meaning exists
                        WITH v
                        FOREACH (_ IN CASE WHEN $meaning <> '' THEN [1] ELSE [] END |
                            MERGE (c:Concept {canonical_name: lower($meaning)})
                            ON CREATE SET 
                                c.display_name = $meaning,
                                c.created_at = datetime(),
                                c.name_lower = lower($meaning)
                            SET c.last_seen = datetime()
                            MERGE (v)-[:REPRESENTS_CONCEPT]->(c)
                        )
                        """,
                        var_id=var_id,
                        formula_id=formula_id,
                        symbol=symbol,
                        meaning=meaning,
                        units=units,
                        role=role,
                    )
        
        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception(
                "neo4j_merge_formulas_enhanced_failed",
                extra={"chunk_id": chunk_id, "formula_count": len(formulas)}
            )


def merge_chunk_pedagogy_relations(
    chunk_id: str,
    resource_id: str,
    pedagogy: Dict[str, Any],
    *,
    chunk_type: str | None = None,
    method: str = "llm_pedagogy",
) -> Dict[str, List[str]]:
    if not pedagogy:
        return {"concept_canonicals": []}

    canonical_set: set[str] = set()
    relation_specs: List[Dict[str, Any]] = []
    alias_pairs: List[tuple[str, str]] = []
    figure_entries: List[tuple[str, List[str]]] = []
    derived_formulas: List[str] = []
    evidence_entries: List[Dict[str, Any]] = pedagogy.get("evidence") or []

    for entry in pedagogy.get("defines", []) or []:
        canonical = entry.get("canonical")
        display = entry.get("term")
        if not canonical or not display:
            continue
        canonical_set.add(canonical)
        relation_specs.append({"type": "DEFINES", "canonical": canonical, "display": display, "props": {}})
        for alias in entry.get("aliases", []) or []:
            alias_pairs.append((alias, display))

    def _consume_relation(items: Iterable[Dict[str, Any]], relation: str) -> None:
        for entry in items or []:
            canonical = entry.get("canonical")
            display = entry.get("term")
            if not canonical or not display:
                continue
            canonical_set.add(canonical)
            relation_specs.append({"type": relation, "canonical": canonical, "display": display, "props": {}})

    _consume_relation(pedagogy.get("explains"), "EXPLAINS")
    _consume_relation(pedagogy.get("exemplifies"), "EXEMPLIFIES")

    # Handle FORMULAS - equations associated with concepts
    for entry in pedagogy.get("formulas", []) or []:
        equation = entry.get("equation")
        canonical = entry.get("canonical")
        display = entry.get("about")
        if not equation:
            continue
        if canonical and display:
            canonical_set.add(canonical)
            relation_specs.append({"type": "DERIVES", "canonical": canonical, "display": display, "props": {"equation": equation}})
        derived_formulas.append(equation)

    for entry in pedagogy.get("figure_links", []) or []:
        label = str(entry.get("label") or "").strip()
        if not label:
            continue
        concepts = [c.get("canonical") for c in entry.get("concepts", []) or [] if c.get("canonical")]
        if concepts:
            canonical_set.update(concepts)
        figure_entries.append((label, concepts))

    with managed_driver() as driver:
        if driver is not None and relation_specs:
            try:
                def _tx(tx):
                    for relation in relation_specs:
                        rel_type = relation["type"]
                        canonical = relation["canonical"]
                        display = relation["display"]
                        rel_props = relation.get("props") or {}
                        props = dict(rel_props)
                        if chunk_type:
                            props.setdefault("chunk_type", chunk_type)
                        rel_evidence = [
                            ev
                            for ev in evidence_entries
                            if ev.get("relation") == rel_type and ev.get("target_canonical") == canonical
                        ]
                        if rel_evidence:
                            best = max(rel_evidence, key=lambda ev: ev.get("confidence", 0))
                            sentences = best.get("sentences") or []
                            if sentences:
                                props["evidence_sentences"] = sentences
                            confidence = best.get("confidence")
                            if confidence is not None:
                                props["evidence_confidence"] = confidence
                        tx.run(
                            f"""
                            MERGE (chunk:Chunk {{id: $chunk_id}})
                            MERGE (concept:Concept {{canonical_name: $canonical}})
                            ON CREATE SET concept.display_name = $display, concept.name_lower = $canonical, concept.created_at = datetime()
                            SET concept.display_name = coalesce(concept.display_name, $display),
                                concept.last_seen = datetime(),
                                concept.name_lower = $canonical
                            MERGE (chunk)-[rel:{rel_type}]->(concept)
                            SET rel.resource_id = $resource_id,
                                rel.method = $method,
                                rel.updated_at = datetime()
                            SET rel += $rel_props
                            """,
                            chunk_id=chunk_id,
                            resource_id=resource_id,
                            canonical=canonical,
                            display=display,
                            method=method,
                            rel_props=props,
                        )

                with driver.session() as session:
                    session.execute_write(_tx)
            except Exception:
                logging.exception("neo4j_merge_pedagogy_relations_failed", extra={"chunk_id": chunk_id, "resource_id": resource_id})

    for alias, target in alias_pairs:
        try:
            if not alias or not target:
                continue
            if alias.strip().lower() == target.strip().lower():
                continue
            merge_alias(alias, target, method="llm_defines", evidence_chunk_id=chunk_id)
        except Exception:
            logging.exception("neo4j_merge_alias_from_pedagogy_failed", extra={"alias": alias, "target": target})

    for label, concepts in figure_entries:
        try:
            merge_chunk_figures(chunk_id, resource_id, [label], concept_canonicals=concepts)
        except Exception:
            logging.exception("neo4j_merge_figure_links_failed", extra={"chunk_id": chunk_id, "label": label})

    if derived_formulas:
        try:
            merge_chunk_formulas(chunk_id, resource_id, derived_formulas, concept_canonicals=list(canonical_set))
        except Exception:
            logging.exception("neo4j_merge_derived_formulas_failed", extra={"chunk_id": chunk_id})

    prereq_evidence_index = {}
    for ev in evidence_entries:
        if ev.get("relation") == "PREREQ" and ev.get("from_canonical") and ev.get("to_canonical"):
            prereq_evidence_index[(ev["from_canonical"], ev["to_canonical"])] = ev

    for prereq in pedagogy.get("prerequisites", []) or []:
        if not isinstance(prereq, dict):
            continue
        prereq_term = prereq.get("from")
        target_term = prereq.get("to")
        confidence = prereq.get("confidence")
        if not prereq_term or not target_term:
            continue

        try:
            confidence_value = float(confidence) if confidence is not None else 0.5
        except (TypeError, ValueError):
            continue

        canonical_from = prereq.get("from_canonical")
        canonical_to = prereq.get("to_canonical")
        if canonical_from:
            canonical_set.add(canonical_from)
        if canonical_to:
            canonical_set.add(canonical_to)

        evidence_entry = (
            prereq_evidence_index.get((canonical_from, canonical_to))
            if canonical_from and canonical_to
            else None
        )
        sentences = evidence_entry.get("sentences") if evidence_entry else None
        evidence_conf = evidence_entry.get("confidence") if evidence_entry else None

        try:
            merge_prerequisite_edge(
                prereq_term,
                target_term,
                confidence=confidence_value,
                evidence_chunk_id=chunk_id,
                method=method,
                evidence_sentences=sentences,
                evidence_confidence=evidence_conf,
            )
        except Exception:
            logging.exception(
                "neo4j_merge_pedagogy_prereq_failed",
                extra={"from": prereq_term, "to": target_term},
            )

    return {"concept_canonicals": sorted(canonical_set)}


def merge_next_chunk(prev_chunk_id: str | None, next_chunk_id: str | None, resource_id: str) -> None:
    if not prev_chunk_id or not next_chunk_id or prev_chunk_id == next_chunk_id:
        return

    with managed_driver() as driver:
        if driver is None:
            return

        def _tx(tx):
            tx.run(
                """
                MERGE (prev:Chunk {id: $prev_id})
                MERGE (next:Chunk {id: $next_id})
                MERGE (prev)-[rel:NEXT_CHUNK]->(next)
                SET rel.resource_id = $resid,
                    rel.updated_at = datetime()
                """,
                prev_id=prev_chunk_id,
                next_id=next_chunk_id,
                resid=resource_id,
            )

        try:
            with driver.session() as session:
                session.execute_write(_tx)
        except Exception:
            logging.exception(
                "neo4j_merge_next_chunk_failed",
                extra={"prev": prev_chunk_id, "next": next_chunk_id},
            )