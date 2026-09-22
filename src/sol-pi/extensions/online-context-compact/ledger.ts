/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { appendFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";

/**
 * Append-only JSONL record of every `turn_end` invocation's boundary/decision
 * classification, including the ones that never reach `decideCompaction()`.
 *
 * Modeled on `observation-pack/ledger.ts`. A missing or empty ledger is not
 * evidence that compaction was correctly deferred — it may mean this
 * mechanism's `pendingBoundary` gate was never satisfied (e.g. the session
 * only used a different plan-tracking tool). Fail-open: a write failure is
 * logged and swallowed, never thrown, and never blocks the caller's own
 * return. This is fail-open, not latency-free — the caller still awaits the
 * write attempt before proceeding.
 */
export type CompactionLedger = (entry: Record<string, unknown>) => Promise<void>;

export function createCompactionLedger(path: string): CompactionLedger {
	return async (entry) => {
		try {
			await mkdir(dirname(path), { recursive: true });
			await appendFile(path, `${JSON.stringify({ timestamp: new Date().toISOString(), ...entry })}\n`, "utf8");
		} catch (error) {
			const reason = error instanceof Error ? error.message : String(error);
			console.error(`[online-context-compact] fail-open ledger write: ${reason}`);
		}
	};
}
