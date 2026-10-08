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
 * logged and swallowed, never thrown.
 *
 * Not latency-free, and deliberately NOT synchronous with its caller's own
 * state handoff: the returned function's body runs synchronously only up to
 * its own first `await` (`await mkdir`), so calling it starts the write and
 * returns a pending promise immediately without blocking anything the caller
 * does next. `online-context-compact/extension.ts`'s `turn_end` handler
 * relies on exactly this — it fires the write, then performs its
 * `selected`/`context.abort()` handoff in the same synchronous tick with no
 * `await` in between (a real race existed here: awaiting the write BEFORE
 * that handoff let `agent_settled` consume `selected` before it was ever
 * set), and only awaits the write's promise right before each of its own
 * return paths — so every invocation still produces exactly one ledger row,
 * just no longer on the handoff's critical path.
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
