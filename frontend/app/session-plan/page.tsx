"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useAuth } from "../hooks/useAuth";
import { API_BASE } from "../lib/api";
import { loadNotesMetadata, mapJobStatusToLabel } from "../lib/notes";
import { Card, PrimaryButton, SecondaryButton } from "../components/ui";

interface NoteMeta {
  id: string;
  name?: string;
  status?: string;
  last_job_status?: string;
}

interface ResourceConceptSummary {
  concept: string;
  canonical: string;
  mastery?: number | null;
  level?: string | null;
  occurrences: number;
  resource_ids: string[];
  pages: number[];
  pedagogy_roles: Record<string, number>;
  path_index?: number | null;
  hidden?: boolean;
}

interface SessionStep {
  step_id: number;
  concept_id: string;
  concept_name: string;
  reason: string;
  status: string;
  estimated_duration: number;
}

interface SessionPlan {
  session_id: string;
  steps: SessionStep[];
  resource_ids: string[];
  created_at: number;
}

type SortMode = "path" | "chronology" | "mastery";

export default function SessionPlanPage() {
  const { token } = useAuth({ requireAuth: true });
  const [notes, setNotes] = useState<NoteMeta[]>([]);
  const [selectedNoteIds, setSelectedNoteIds] = useState<string[]>([]);
  const [concepts, setConcepts] = useState<ResourceConceptSummary[]>([]);
  const [conceptsLoading, setConceptsLoading] = useState(false);
  const [conceptsError, setConceptsError] = useState<string | null>(null);
  const [selectedCanonicalIds, setSelectedCanonicalIds] = useState<string[]>([]);
  const [sortMode, setSortMode] = useState<SortMode>("path");
  const [filterWeakOnly, setFilterWeakOnly] = useState(false);
  const [plan, setPlan] = useState<SessionPlan | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);
  const [sessionStarting, setSessionStarting] = useState(false);

  function authHeader() {
    return token ? `Bearer ${token}` : "Bearer test-token";
  }

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = loadNotesMetadata() as NoteMeta[];
    setNotes(stored || []);
    const ready = (stored || []).filter(
      (n) => !n.status || n.status === "Ready" || mapJobStatusToLabel(n.status) === "Ready"
    );
    if (ready.length > 0) {
      setSelectedNoteIds([ready[0].id]);
    }
  }, []);

  const readyNotes = useMemo(
    () =>
      (notes || []).filter(
        (n) => !n.status || n.status === "Ready" || mapJobStatusToLabel(n.status) === "Ready"
      ),
    [notes]
  );

  useEffect(() => {
    const ids = (selectedNoteIds || []).filter(Boolean);
    if (!ids.length) {
      setConcepts([]);
      setConceptsError(null);
      setSelectedCanonicalIds([]);
      return;
    }

    let cancelled = false;
    async function fetchConcepts() {
      setConceptsLoading(true);
      setConceptsError(null);
      try {
        const res = await fetch(`${API_BASE}/api/resources/concepts`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: authHeader(),
          },
          body: JSON.stringify({ resource_ids: ids }),
        });
        if (!res.ok) {
          throw new Error(`Concepts HTTP ${res.status}`);
        }
        const data = await res.json();
        if (!cancelled) {
          setConcepts(Array.isArray(data) ? data : []);
          setSelectedCanonicalIds([]);
        }
      } catch (e: any) {
        if (!cancelled) {
          setConceptsError(String(e?.message || e));
          setConcepts([]);
        }
      } finally {
        if (!cancelled) {
          setConceptsLoading(false);
        }
      }
    }

    void fetchConcepts();
    return () => {
      cancelled = true;
    };
  }, [token, JSON.stringify(selectedNoteIds)]); // eslint-disable-line react-hooks/exhaustive-deps

  const visibleConcepts = useMemo(() => {
    let items = Array.isArray(concepts) ? concepts.slice() : [];

    if (filterWeakOnly) {
      items = items.filter((c) => !c.level || c.level === "beginner" || c.level === "developing");
    }

    const byName = (a: ResourceConceptSummary, b: ResourceConceptSummary) => {
      const an = (a.concept || "").toLowerCase();
      const bn = (b.concept || "").toLowerCase();
      if (an < bn) return -1;
      if (an > bn) return 1;
      return 0;
    };

    if (sortMode === "path") {
      items.sort((a, b) => {
        const ap = typeof a.path_index === "number" ? a.path_index : Number.MAX_SAFE_INTEGER;
        const bp = typeof b.path_index === "number" ? b.path_index : Number.MAX_SAFE_INTEGER;
        if (ap !== bp) return ap - bp;
        const am = typeof a.mastery === "number" ? a.mastery : 1;
        const bm = typeof b.mastery === "number" ? b.mastery : 1;
        if (am !== bm) return am - bm;
        return byName(a, b);
      });
      return items;
    }

    if (sortMode === "chronology") {
      items.sort((a, b) => {
        const ap = Array.isArray(a.pages) && a.pages.length ? a.pages[0] : Number.MAX_SAFE_INTEGER;
        const bp = Array.isArray(b.pages) && b.pages.length ? b.pages[0] : Number.MAX_SAFE_INTEGER;
        if (ap !== bp) return ap - bp;
        return byName(a, b);
      });
      return items;
    }

    items.sort((a, b) => {
      const am = typeof a.mastery === "number" ? a.mastery : 1;
      const bm = typeof b.mastery === "number" ? b.mastery : 1;
      if (am !== bm) return am - bm;
      return byName(a, b);
    });
    return items;
  }, [concepts, sortMode, filterWeakOnly]);

  function toggleNoteSelection(id: string) {
    setSelectedNoteIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((x) => x !== id);
      }
      return [...prev, id];
    });
  }

  function toggleConceptSelection(canonical: string) {
    const key = (canonical || "").trim();
    if (!key) return;
    setSelectedCanonicalIds((prev) => {
      if (prev.includes(key)) {
        return prev.filter((x) => x !== key);
      }
      return [...prev, key];
    });
  }

  async function requestPlan(useSelectedConcepts: boolean) {
    if (!selectedNoteIds.length) {
      setPlanError("Select at least one note");
      return;
    }
    if (useSelectedConcepts && !selectedCanonicalIds.length) {
      setPlanError("Select at least one concept");
      return;
    }

    setPlanLoading(true);
    setPlanError("");
    setPlan(null);

    try {
      const res = await fetch(`${API_BASE}/api/session/plan`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: authHeader(),
        },
        body: JSON.stringify({
          resource_ids: selectedNoteIds,
          concept_ids: useSelectedConcepts ? selectedCanonicalIds : undefined,
        }),
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err || `Plan HTTP ${res.status}`);
      }
      const data = await res.json();
      setPlan(data);
    } catch (e: any) {
      setPlanError(String(e?.message || e));
    } finally {
      setPlanLoading(false);
    }
  };

  const startSessionFromPlan = async () => {
    if (!plan) {
      setPlanError("No plan available");
      return;
    }

    setSessionStarting(true);
    setPlanError("");

    try {
      const res = await fetch(`${API_BASE}/api/mdp/start_from_plan`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: authHeader(),
        },
        body: JSON.stringify({
          session_plan: plan,
          // TODO: Get actual user ID from token or context. 
          // Using a known valid UUID for testing to satisfy DB constraints.
          student_id: "133747cc-5c82-43e2-9d37-644eaa02eaee", 
        }),
      });
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err || `Start session HTTP ${res.status}`);
      }
      const data = await res.json();
      const sessionId = data.session_id;
      
      // Redirect to the chat interface with the new session
      if (typeof window !== "undefined") {
        window.location.href = `/mdp-chat?session_id=${sessionId}`;
      }
    } catch (e: any) {
      setPlanError(String(e?.message || e));
    } finally {
      setSessionStarting(false);
    }
  };

  return (
    <main>
      <h1 style={{ fontSize: 26, marginBottom: 8 }}>Session Planner (MDP)</h1>
      <p style={{ marginTop: 0, marginBottom: 16, color: "#4b5563", maxWidth: 720 }}>
        Pick ingested notes and concepts, then generate a session-level concept plan using the MDP planner
        (prerequisites + chronology). This page is for debugging the session planning layer.
      </p>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 16 }}>
        <Card title="1. Choose notes" style={{ flex: "1 1 260px", minWidth: 260 }}>
          {readyNotes.length === 0 ? (
            <p style={{ fontSize: 13, color: "#6b7280" }}>
              No ready notes found. Upload and ingest notes on the <Link href="/notes">Notes</Link> page.
            </p>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {readyNotes.map((n) => {
                const checked = selectedNoteIds.includes(n.id);
                return (
                  <label
                    key={n.id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      fontSize: 13,
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleNoteSelection(n.id)}
                      style={{ marginRight: 4 }}
                    />
                    <span style={{ flex: 1 }}>
                      <span style={{ display: "block", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {n.name || n.id}
                      </span>
                      <span style={{ fontSize: 11, color: "#6b7280" }}>{n.status || "Ready"}</span>
                    </span>
                  </label>
                );
              })}
            </div>
          )}
        </Card>

        <Card
          title="2. Choose concepts"
          subtitle="Concepts extracted from the selected notes"
          style={{ flex: "2 1 320px", minWidth: 320 }}
        >
          {conceptsError && (
            <div style={{ marginBottom: 8, fontSize: 12, color: "#b91c1c" }}>{conceptsError}</div>
          )}
          {conceptsLoading && !conceptsError && (
            <div style={{ marginBottom: 8, fontSize: 12, color: "#6b7280" }}>Loading concepts…</div>
          )}
          {!conceptsLoading && !conceptsError && selectedNoteIds.length > 0 && concepts.length === 0 && (
            <div style={{ marginBottom: 8, fontSize: 12, color: "#6b7280" }}>
              No concepts found yet for the selected notes.
            </div>
          )}

          {concepts.length > 0 && (
            <>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 8,
                  marginBottom: 8,
                  flexWrap: "wrap",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <label style={{ fontSize: 11, color: "#4b5563" }}>
                    Sort:
                    <select
                      value={sortMode}
                      onChange={(e) => setSortMode(e.target.value as SortMode)}
                      style={{
                        marginLeft: 4,
                        fontSize: 11,
                        borderRadius: 9999,
                        padding: "2px 6px",
                        border: "1px solid #d1d5db",
                      }}
                    >
                      <option value="path">Learning path (prereqs)</option>
                      <option value="chronology">Chronology (pages)</option>
                      <option value="mastery">Mastery (weak first)</option>
                    </select>
                  </label>
                  <label style={{ fontSize: 11, color: "#4b5563" }}>
                    <input
                      type="checkbox"
                      checked={filterWeakOnly}
                      onChange={(e) => setFilterWeakOnly(e.target.checked)}
                      style={{ marginRight: 4 }}
                    />
                    Only weak / unknown
                  </label>
                </div>
                <div style={{ fontSize: 11, color: "#6b7280" }}>{visibleConcepts.length} concepts</div>
              </div>

              <div
                style={{
                  maxHeight: 260,
                  overflow: "auto",
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                }}
              >
                {visibleConcepts.map((c) => {
                  const checked = selectedCanonicalIds.includes(c.canonical);
                  const mastery =
                    typeof c.mastery === "number" ? ` · mastery ${(c.mastery as number).toFixed(2)}` : "";
                  const pathIndex =
                    typeof c.path_index === "number" ? ` · path #${c.path_index}` : "";
                  const firstPage = Array.isArray(c.pages) && c.pages.length ? c.pages[0] : null;
                  return (
                    <label
                      key={c.canonical}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: 8,
                        borderRadius: 8,
                        border: "1px solid #e5e7eb",
                        padding: "6px 8px",
                        fontSize: 12,
                        background: "#ffffff",
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div
                          style={{
                            fontSize: 13,
                            fontWeight: 500,
                            color: "#111827",
                            whiteSpace: "nowrap",
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                          }}
                        >
                          {c.concept}
                        </div>
                        <div style={{ marginTop: 2, fontSize: 11, color: "#6b7280" }}>
                          {c.level || "Unknown"}
                          {mastery}
                          {pathIndex}
                          {firstPage != null && ` · pg ${firstPage}`}
                        </div>
                      </div>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleConceptSelection(c.canonical)}
                      />
                    </label>
                  );
                })}
              </div>
            </>
          )}
        </Card>

        <Card
          title="3. Generate plan"
          subtitle="Run the session-level MDP over the chosen notes/concepts"
          style={{ flex: "1 1 260px", minWidth: 260 }}
        >
          {planError && (
            <div style={{ marginBottom: 8, fontSize: 12, color: "#b91c1c" }}>{planError}</div>
          )}

          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 12 }}>
            <PrimaryButton onClick={() => requestPlan(false)} disabled={planLoading || !selectedNoteIds.length}>
              {planLoading ? "Planning..." : "Plan full session from notes"}
            </PrimaryButton>
            <SecondaryButton
              onClick={() => requestPlan(true)}
              disabled={planLoading || !selectedCanonicalIds.length}
            >
              {planLoading ? "Planning..." : "Plan only selected concepts"}
            </SecondaryButton>
          </div>

          {plan && plan.steps && plan.steps.length > 0 && (
            <div style={{ maxHeight: 260, overflow: "auto", fontSize: 12 }}>
              <div style={{ marginBottom: 6, color: "#6b7280" }}>
                Session <code>{plan.session_id.slice(0, 8)}...</code> · {plan.steps.length} steps
              </div>
              <ol style={{ paddingLeft: 18, margin: 0 }}>
                {plan.steps.map((s) => (
                  <li key={s.step_id} style={{ marginBottom: 4 }}>
                    <div style={{ fontWeight: 500 }}>{s.concept_name || s.concept_id}</div>
                    <div style={{ fontSize: 11, color: "#6b7280" }}>
                      {s.reason} · est. {s.estimated_duration} min
                    </div>
                  </li>
                ))}
              </ol>
              
              <div style={{ marginTop: 12, paddingTop: 8, borderTop: "1px solid #e5e7eb" }}>
                <PrimaryButton
                  onClick={startSessionFromPlan}
                  disabled={sessionStarting}
                  style={{ width: "100%" }}
                >
                  {sessionStarting ? "Starting..." : "Start Session"}
                </PrimaryButton>
              </div>
            </div>
          )}

          {!plan && !planLoading && (
            <p style={{ fontSize: 12, color: "#6b7280" }}>
              Choose notes and optionally concepts, then generate a plan to inspect the session MDP ordering.
            </p>
          )}
        </Card>
      </div>
    </main>
  );
}
