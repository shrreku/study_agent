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

  const messagesEndRef = useRef<HTMLDivElement>(null);

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
      const res = await fetch(`http://localhost:8000/api/mdp/session/${sid}`);
      if (!res.ok) throw new Error("Failed to fetch session state");

      const data = await res.json();
      const result = data.result;

      if (result.status === "rendered") {
        const assistantMsg: Message = {
          role: "assistant",
          content: result.rendered_content,
          debug: {
            step: result.step,
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
      const res = await fetch("http://localhost:8000/api/mdp/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ concept, student_id: "test_user" }),
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
      const res = await fetch("http://localhost:8000/api/mdp/chat", {
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
          },
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setCurrentStep(result.step);
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

  return (
    <div className="flex h-screen bg-gray-100 p-4 gap-4">
      {/* Sidebar / Debug Panel */}
      <div className="w-1/3 flex flex-col gap-4">
        <div className="bg-white p-4 rounded shadow">
          <h2 className="text-xl font-bold mb-4">Session Setup</h2>
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
                disabled= {loading}
              >
                Start
              </button>
            </div>
          ) : (
            <div>
              <p className="text-sm text-green-600 font-semibold mb-2">
                Active Session: {sessionId.slice(0, 8)}...
              </p>
              <button
                className="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600"
                onClick={() => {
                  setSessionId(null);
                  setMessages([]);
                  setCurrentStep(null);
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
            <h3 className="font-bold mb-2">Current Step Debug</h3>
            <pre className="text-xs bg-gray-50 p-2 rounded border overflow-x-auto">
              {JSON.stringify(currentStep, null, 2)}
            </pre>

            <h3 className="font-bold mt-4 mb-2">Last Analysis</h3>
            <p className="text-xs text-gray-500">Check backend logs for full analysis details.</p>
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
