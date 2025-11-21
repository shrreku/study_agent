'use client';

import Link from 'next/link';
import { EnvironmentTutorPage } from '../../../components/tutor/EnvironmentTutorPage';

export default function EnvironmentPedagogyPage() {
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="max-w-6xl mx-auto px-4 py-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Pedagogical Environment</p>
            <h1 className="text-2xl font-bold text-slate-900">Environment Tutor Testbed</h1>
            <p className="text-sm text-slate-600">
              Drive the new 3-layer MDP orchestrator via the /api/tutor/pedagogy endpoint.
            </p>
          </div>
          <Link
            href="/tutor/pedagogy"
            className="inline-flex items-center gap-2 text-sm font-semibold text-blue-600 hover:text-blue-800"
          >
             Back to Tutor Home
          </Link>
        </div>
      </header>

      <main className="py-6">
        <EnvironmentTutorPage />
      </main>
    </div>
  );
}
