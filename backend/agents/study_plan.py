from typing import List, Dict, Any
import os
import logging
import uuid
from datetime import date, timedelta, datetime
from .retrieval import hybrid_search, diversify_by_page
from llm import call_llm_json
from prompts import get as prompt_get, render as prompt_render


def study_plan_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    target = payload.get("target_concepts") or payload.get("concepts") or []
    if not target:
        return {"plan_id": str(uuid.uuid4()), "todos": []}

    prereq_order: List[str] = []
    try:
        try:
            from neo4j import GraphDatabase
        except Exception:
            raise
        uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
        driver = GraphDatabase.driver(uri, auth=(user, password))

        def _fetch_prereqs(tx, name: str, max_depth: int = 6):
            try:
                q = (
                    "MATCH (c:Concept {name:$name})\n"
                    "CALL apoc.path.subgraphNodes(c, {relationshipFilter:'<PREREQUISITE_OF', maxLevel:$max_depth}) YIELD node\n"
                    "RETURN DISTINCT node.name AS name"
                )
                res = tx.run(q, name=name, max_depth=max_depth)
            except Exception:
                q2 = (
                    "MATCH (p:Concept)-[:PREREQUISITE_OF*0..$max_depth]->(c:Concept {name:$name})\n"
                    "RETURN DISTINCT p.name AS name"
                )
                res = tx.run(q2, name=name, max_depth=max_depth)
            return [r["name"] for r in res]

        with driver.session() as session:
            seen = []
            for t in target:
                try:
                    names = session.execute_read(_fetch_prereqs, t, 6)
                except Exception:
                    names = []
                if t not in names:
                    names.append(t)
                for n in names:
                    if n not in seen:
                        seen.append(n)
            prereq_order = seen
    except Exception:
        logging.exception("study_plan_neo4j_failed_fallback")
        prereq_order = list(dict.fromkeys(target))

    exam_date_str = payload.get("exam_date")
    try:
        exam_date = datetime.fromisoformat(exam_date_str).date() if exam_date_str else None
    except Exception:
        exam_date = None

    daily_minutes = int(payload.get("daily_minutes") or 60)
    minutes_per_concept = int(payload.get("minutes_per_concept") or 30)
    concepts_per_day = max(1, daily_minutes // minutes_per_concept)

    todos: List[Dict[str, Any]] = []
    resource_id = payload.get("resource_id")
    start = date.today()
    if exam_date and (exam_date - start).days > 0:
        day = 0
        i = 0
        while i < len(prereq_order):
            group = prereq_order[i : i + concepts_per_day]
            dt = (start + timedelta(days=day)).isoformat()
            for c in group:
                refs = []
                try:
                    rs = hybrid_search(c, k=6, resource_id=resource_id)
                    rs = diversify_by_page(rs, per_page=1)
                    for r in rs[:3]:
                        refs.append({"chunk_id": r.get("id"), "snippet": r.get("snippet")})
                except Exception:
                    pass
                # generate a short summary using top snippet
                summary = ""
                try:
                    top_snip = (refs[0]["snippet"] if refs else "")
                    tmpl = prompt_get("study.summary")
                    prompt = prompt_render(tmpl, {"concept": c, "snippet": top_snip})
                    out = call_llm_json(prompt, {"summary": ""})
                    summary = str(out.get("summary") or "")[:400]
                except Exception:
                    summary = ""
                todos.append({"date": dt, "minutes": minutes_per_concept, "concept": c, "summary": summary, "chunk_refs": refs})
            i += concepts_per_day
            day += 1
    else:
        for day_idx in range(0, len(prereq_order), concepts_per_day):
            group = prereq_order[day_idx : day_idx + concepts_per_day]
            dt = (start + timedelta(days=day_idx // concepts_per_day)).isoformat()
            for c in group:
                refs = []
                try:
                    rs = hybrid_search(c, k=6, resource_id=resource_id)
                    rs = diversify_by_page(rs, per_page=1)
                    for r in rs[:3]:
                        refs.append({"chunk_id": r.get("id"), "snippet": r.get("snippet")})
                except Exception:
                    pass
                summary = ""
                try:
                    top_snip = (refs[0]["snippet"] if refs else "")
                    tmpl = prompt_get("study.summary")
                    prompt = prompt_render(tmpl, {"concept": c, "snippet": top_snip})
                    out = call_llm_json(prompt, {"summary": ""})
                    summary = str(out.get("summary") or "")[:400]
                except Exception:
                    summary = ""
                todos.append({"date": dt, "minutes": minutes_per_concept, "concept": c, "summary": summary, "chunk_refs": refs})

    return {"plan_id": str(uuid.uuid4()), "todos": todos}


