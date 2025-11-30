"use client";

import React, { useState, useRef, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";

// Types based on our backend schemas
interface Step {
  step_id: number;
  concept: string;
  pedagogy: string;
  content: string;
}

interface Analysis {
  intent: string;
  correctness: string | null;
  feedback: string;
  sentiment: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  debug?: {
    analysis?: Analysis;
    step?: Step;
    action?: string;
    thinking?: string;
    policy?: string;
    mastery?: number;
  };
}

function ChatContent() {
  const searchParams = useSearchParams();
  const initialSessionId = searchParams.get("session_id");

  const [concept, setConcept] = useState("convection");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<Step | null>(null);
  const [useV2, setUseV2] = useState(true); // Toggle between v1 and v2 API
  const [policyType, setPolicyType] = useState<string>("hybrid"); // "rule", "llm", "unified", "hybrid"
  const [transitions, setTransitions] = useState<any[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // API base path based on version toggle
  const apiBase = useV2 ? "http://localhost:8000/api/mdp/v2" : "http://localhost:8000/api/mdp";

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize session from URL if present
  useEffect(() => {
    if (initialSessionId && !sessionId) {
      setSessionId(initialSessionId);
      fetchSessionState(initialSessionId);
    }
  }, [initialSessionId]);

  const fetchSessionState = async (sid: string) => {
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/session/${sid}`);
      if (!res.ok) throw new Error("Failed to fetch session state");

      const data = await res.json();
      const result = data.result;

      if (result.status === "rendered") {
        const assistantMsg: Message = {
          role: "assistant",
          content: result.rendered_content,
          debug: {
            step: result.step,
            action: result.action,
            thinking: result.thinking,
            policy: result.debug?.policy,
            mastery: result.debug?.mastery,
          },
        };
        setMessages([assistantMsg]);
        setCurrentStep(result.step);
      } else if (result.status === "done") {
        setMessages([{ role: "assistant", content: "[Session Complete]" }]);
      }
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const startSession = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          concept, 
          student_id: "test_user",
          policy_type: policyType 
        }),
      });
      const data = await res.json();
      setSessionId(data.session_id);

      const result = data.result;
      if (result.status === "rendered") {
        const assistantMsg: Message = {
          role: "assistant",
          content: result.rendered_content,
          debug: {
            step: result.step,
            action: result.action,
            policy: result.debug?.policy,
          },
        };
        setMessages([assistantMsg]);
        setCurrentStep(result.step);
      }
    } catch (e) {
      console.error(e);
      alert("Failed to start session");
    }
    setLoading(false);
  };

  const sendMessage = async () => {
    if (!input.trim() || !sessionId) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: userMsg.content }),
      });
      const data = await res.json();
      const result = data.result;

      if (result.status === "rendered") {
        const assistantMsg: Message = {
          role: "assistant",
          content: result.rendered_content,
          debug: {
            step: result.step,
            action: result.action,
            thinking: result.thinking,
            policy: result.debug?.policy,
            mastery: result.debug?.mastery,
          },
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setCurrentStep(result.step);
        
        // Fetch transitions if using v2
        if (useV2) {
          fetchTransitions(sessionId);
        }
      } else if (result.status === "done") {
        setMessages((prev) => [...prev, { role: "assistant", content: "[Session Complete]" }]);
        setSessionId(null);
      }
    } catch (e) {
      console.error(e);
      alert("Failed to send message");
    }
    setLoading(false);
  };
  
  const fetchTransitions = async (sid: string) => {
    try {
      const res = await fetch(`${apiBase}/transitions/${sid}`);
      if (res.ok) {
        const data = await res.json();
        setTransitions(data.transitions || []);
      }
    } catch (e) {
      console.error("Failed to fetch transitions:", e);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 p-4 gap-4">
      {/* Sidebar / Debug Panel */}
      <div className="w-1/3 flex flex-col gap-4">
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-xl font-bold mb-4">MDP Session Setup</h2>
          
          {/* API Version Toggle */}
          <div className="mb-4 p-2 bg-gray-50 rounded">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={useV2}
                onChange={(e) => setUseV2(e.target.checked)}
                disabled={!!sessionId}
              />
              <span className={useV2 ? "font-semibold text-blue-600" : ""}>
                Use MDP v2 (with transition logging)
              </span>
            </label>
            {useV2 && (
              <div className="mt-2">
                <label className="text-sm font-medium">Policy Type:</label>
                <select
                  className="ml-2 border rounded p-1 text-sm"
                  value={policyType}
                  onChange={(e) => setPolicyType(e.target.value)}
                  disabled={!!sessionId}
                >
                  <option value="rule">Rule-based (fast, deterministic)</option>
                  <option value="llm">LLM Policy (nuanced)</option>
                  <option value="unified">Unified LLM (single call)</option>
                  <option value="hybrid">Hybrid (recommended)</option>
                </select>
              </div>
            )}
          </div>
          
          {!sessionId ? (
            <div className="flex gap-2">
              <input
                className="border p-2 rounded flex-1"
                value={concept}
                onChange={(e) => setConcept(e.target.value)}
                placeholder="Enter concept (e.g. convection)"
              />
              <button
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                onClick={startSession}
                disabled={loading}
              >
                Start
              </button>
            </div>
          ) : (
            <div>
              <p className="text-sm text-green-600 font-semibold mb-2">
                Active Session: {sessionId.slice(0, 8)}...
              </p>
              <p className="text-xs text-gray-500 mb-2">
                API: {useV2 ? "v2" : "v1"} | Policy: {policyType}
              </p>
              <button
                className="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600"
                onClick={() => {
                  setSessionId(null);
                  setMessages([]);
                  setCurrentStep(null);
                  setTransitions([]);
                  if (typeof window !== "undefined") {
                    window.history.pushState({}, "", "/mdp-chat");
                  }
                }}
              >
                Reset
              </button>
            </div>
          )}
        </div>

        {currentStep && (
          <div className="bg-white p-4 rounded shadow flex-1 overflow-auto">
            <h3 className="font-bold mb-2">Current Step</h3>
            <pre className="text-xs bg-gray-50 p-2 rounded border overflow-x-auto">
              {JSON.stringify(currentStep, null, 2)}
            </pre>

            {/* Last message debug info */}
            {messages.length > 0 && messages[messages.length - 1].debug && (
              <>
                <h3 className="font-bold mt-4 mb-2">Last Action</h3>
                <div className="text-xs space-y-1">
                  <p><span className="font-semibold">Action:</span> {messages[messages.length - 1].debug?.action || "N/A"}</p>
                  <p><span className="font-semibold">Policy:</span> {messages[messages.length - 1].debug?.policy || "N/A"}</p>
                  <p><span className="font-semibold">Mastery:</span> {messages[messages.length - 1].debug?.mastery?.toFixed(2) || "N/A"}</p>
                </div>
                {messages[messages.length - 1].debug?.thinking && (
                  <div className="mt-2">
                    <p className="font-semibold text-xs">Thinking:</p>
                    <pre className="text-xs bg-yellow-50 p-2 rounded border mt-1 whitespace-pre-wrap">
                      {messages[messages.length - 1].debug?.thinking}
                    </pre>
                  </div>
                )}
              </>
            )}

            {/* Transitions log (v2 only) */}
            {useV2 && transitions.length > 0 && (
              <>
                <h3 className="font-bold mt-4 mb-2">Transitions ({transitions.length})</h3>
                <div className="max-h-40 overflow-auto">
                  {transitions.map((t, idx) => (
                    <div key={idx} className="text-xs p-1 border-b border-gray-100">
                      <span className="font-semibold">Turn {t.turn_number}:</span>{" "}
                      <span className="text-blue-600">{t.action?.action}</span> →{" "}
                      <span className={t.reward?.total > 0 ? "text-green-600" : "text-red-600"}>
                        r={t.reward?.total?.toFixed(2)}
                      </span>
                      <span className="text-gray-400 ml-2">Δm={t.mastery_delta?.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Chat Area */}
      <div className="flex-1 bg-white rounded shadow flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] p-3 rounded-lg ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white rounded-br-none"
                    : "bg-gray-100 text-gray-800 rounded-bl-none"
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 border-t bg-gray-50">
          <div className="flex gap-2">
            <input
              className="flex-1 border p-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
              placeholder={sessionId ? "Type your answer..." : "Start a session first"}
              disabled={!sessionId || loading}
            />
            <button
              className={`px-6 py-2 rounded text-white font-medium ${
                !sessionId || loading ? "bg-gray-400" : "bg-blue-600 hover:bg-blue-700"
              }`}
              onClick={sendMessage}
              disabled={!sessionId || loading}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function MDPChatPage() {
  return (
    <Suspense fallback={<div>Loading session...</div>}>
      <ChatContent />
    </Suspense>
  );
}
