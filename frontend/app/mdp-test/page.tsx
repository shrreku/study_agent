"use client";

import React, { useState, useEffect, useRef } from 'react';

export default function MDPTestPage() {
    const [session, setSession] = useState(null);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [plan, setPlan] = useState(null);
    const [debug, setDebug] = useState(null);
    const [concept, setConcept] = useState("Photosynthesis"); // Default concept
    const [sessionId, setSessionId] = useState("");

    // Generate session ID only on client to avoid hydration mismatch
    useEffect(() => {
        setSessionId(`sess_${Math.floor(Math.random() * 10000)}`);
    }, []);

    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async (text = null, button = null, startConcept = null) => {
        setLoading(true);
        try {
            const payload: {
                user_id: string;
                session_id: string;
                concept?: string;
                button?: string;
                message?: string;
            } = {
                user_id: "test_user",
                session_id: sessionId,
            };

            if (startConcept) {
                payload.concept = startConcept;
            } else if (button) {
                payload.button = button;
            } else {
                payload.message = text || input;
            }

            const res = await fetch("/api/agent/tutor-mdp", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-token"
                },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                throw new Error(`API Error: ${res.status}`);
            }

            const data = await res.json();

            // Update state
            if (data.response) {
                setMessages(prev => [...prev, { role: "assistant", content: data.response }]);
            }

            if (data.plan) {
                setPlan(data.plan);
            }

            if (data.debug) {
                setDebug(data.debug);
            }

            if (!button && !startConcept) {
                setMessages(prev => [...prev, { role: "user", content: text || input }]);
                setInput("");
            }

        } catch (err) {
            console.error(err);
            setMessages(prev => [...prev, { role: "system", content: `Error: ${err.message}` }]);
        } finally {
            setLoading(false);
        }
    };

    const handleStart = () => {
        setMessages([]);
        sendMessage(null, null, concept);
    };

    return (
        <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
            {/* Sidebar: Plan & Debug */}
            <div className="w-1/3 border-r border-gray-700 flex flex-col">
                <div className="p-4 border-b border-gray-700 bg-gray-800">
                    <h2 className="text-xl font-bold mb-2">MDP Tutor Debugger</h2>
                    <div className="flex gap-2 mb-2">
                        <input
                            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 flex-1 text-sm"
                            value={concept}
                            onChange={(e) => setConcept(e.target.value)}
                            placeholder="Concept..."
                        />
                        <button
                            onClick={handleStart}
                            className="bg-blue-600 hover:bg-blue-500 px-3 py-1 rounded text-sm font-medium"
                        >
                            Start
                        </button>
                    </div>
                    <div className="text-xs text-gray-400">Session: {sessionId || 'Initializing...'}</div>
                </div>

                <div className="flex-1 overflow-y-auto p-4">
                    {plan ? (
                        <div>
                            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Lesson Plan</h3>
                            <div className="space-y-3">
                                {plan.steps.map((step, idx) => (
                                    <div
                                        key={idx}
                                        className={`p-3 rounded border ${step.status === 'current' ? 'border-blue-500 bg-blue-900/20' :
                                            step.status === 'completed' ? 'border-green-600/50 bg-green-900/10 opacity-70' :
                                                'border-gray-700 bg-gray-800 opacity-50'
                                            }`}
                                    >
                                        <div className="flex justify-between items-center mb-1">
                                            <span className="text-xs font-bold uppercase text-gray-400">{step.pedagogy}</span>
                                            {step.status === 'current' && <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>}
                                        </div>
                                        <div className="text-sm">{step.content}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="text-gray-500 text-sm italic">No plan loaded yet. Start a session.</div>
                    )}

                    {debug && (
                        <div className="mt-8">
                            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">Last Turn Debug</h3>
                            <pre className="text-xs bg-black p-2 rounded overflow-x-auto text-green-400 font-mono">
                                {JSON.stringify(debug, null, 2)}
                            </pre>
                        </div>
                    )}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col">
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-2xl p-4 rounded-lg ${msg.role === 'user' ? 'bg-blue-600 text-white' :
                                msg.role === 'system' ? 'bg-red-900/50 text-red-200 border border-red-800' :
                                    'bg-gray-800 text-gray-200 border border-gray-700'
                                }`}>
                                {msg.role === 'assistant' ? (
                                    <div className="prose prose-invert max-w-none">
                                        {/* Simple rendering for now, could use ReactMarkdown */}
                                        <p className="whitespace-pre-wrap">{msg.content}</p>
                                    </div>
                                ) : (
                                    <p>{msg.content}</p>
                                )}
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                {/* Controls & Input */}
                <div className="p-4 border-t border-gray-700 bg-gray-800">
                    {/* Action Buttons */}
                    <div className="flex justify-center gap-4 mb-4">
                        <button
                            onClick={() => sendMessage(null, "replan")}
                            className="px-4 py-2 bg-amber-700 hover:bg-amber-600 rounded text-sm font-medium transition-colors"
                            disabled={loading}
                        >
                            Replan
                        </button>
                        <button
                            onClick={() => sendMessage(null, "continue")}
                            className="px-4 py-2 bg-green-700 hover:bg-green-600 rounded text-sm font-medium transition-colors"
                            disabled={loading}
                        >
                            Next Step
                        </button>
                        <button
                            onClick={() => sendMessage(null, "finish")}
                            className="px-4 py-2 bg-red-900/50 hover:bg-red-800 rounded text-sm font-medium border border-red-800 transition-colors"
                            disabled={loading}
                        >
                            Finish
                        </button>
                    </div>

                    <div className="flex gap-2">
                        <input
                            className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
                            placeholder="Type your response..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                            disabled={loading}
                        />
                        <button
                            onClick={() => sendMessage()}
                            disabled={loading || !input.trim()}
                            className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        >
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
