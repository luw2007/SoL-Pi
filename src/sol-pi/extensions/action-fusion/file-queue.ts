/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

import { realpath } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const queueTails = new Map<string, Promise<void>>();

/**
 * The host strips a leading `@` only when it prefixes a path or a URL (`@/x`,
 * `@~/x`, `@C:\x`, `@local://x`); `@config.json` is a file whose name starts
 * with `@`. Strip it the same way, or the queue and the hash guard would watch
 * a different file than the mutation writes.
 */
function stripToolPathPrefix(filePath: string): string {
	const prefixed = filePath.startsWith("@") && /^[/~]|[A-Za-z]:[\\/]|[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(filePath.slice(1));
	return prefixed ? filePath.slice(1) : filePath;
}

/**
 * True for a target the host resolves itself instead of the filesystem:
 * a URL scheme of its own (`local://`, internal URLs) or another user's home
 * (`~alice/x`, which needs the host's account lookup). `file://` is a
 * filesystem path resolved here, and `C://` is a Windows path, not a scheme.
 */
export function isHostSchemeTarget(filePath: string): boolean {
	const stripped = stripToolPathPrefix(filePath);
	if (/^[A-Za-z]:[\\/]/.test(stripped)) return false;
	if (/^~[^/\\]/.test(stripped)) return true;
	return /^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(stripped) && !stripped.startsWith("file://");
}

export function resolveToolPath(cwd: string, filePath: string): string {
	const stripped = stripToolPathPrefix(filePath);
	// Pi accepts file URLs; the queue and hash guard must use the same target.
	const expanded = stripped.startsWith("file://") ? fileURLToPath(stripped) : stripped;
	if (expanded === "~") return homedir();
	if (expanded.startsWith("~/")) return resolve(homedir(), expanded.slice(2));
	return resolve(cwd, expanded);
}

export function isMissingPathError(error: unknown): boolean {
	return (
		typeof error === "object" &&
		error !== null &&
		"code" in error &&
		(error.code === "ENOENT" || error.code === "ENOTDIR")
	);
}

async function canonicalQueueKey(filePath: string): Promise<string> {
	// A host-scheme target has no filesystem identity; its literal name is the key.
	if (isHostSchemeTarget(filePath)) return filePath;
	const resolvedPath = resolve(filePath);
	let current = resolvedPath;
	const missingSegments: string[] = [];

	while (true) {
		try {
			return resolve(await realpath(current), ...missingSegments);
		} catch (error) {
			if (!isMissingPathError(error)) throw error;
			const parent = dirname(current);
			if (parent === current) return resolvedPath;
			missingSegments.unshift(basename(current));
			current = parent;
		}
	}
}

async function withQueueKey<T>(key: string, work: () => Promise<T>): Promise<T> {
	const previous = queueTails.get(key) ?? Promise.resolve();
	let release!: () => void;
	const owned = new Promise<void>((resolveOwned) => {
		release = resolveOwned;
	});
	const tail = previous.then(() => owned);
	queueTails.set(key, tail);

	await previous;
	try {
		return await work();
	} finally {
		release();
		if (queueTails.get(key) === tail) queueTails.delete(key);
	}
}

/**
 * Serialize fused operations for one canonical file path. This queue belongs
 * to SoL-Pi and intentionally does not nest Pi's built-in mutation queue.
 */
export async function withFusedFileQueue<T>(filePath: string, work: () => Promise<T>): Promise<T> {
	return withQueueKey(await canonicalQueueKey(filePath), work);
}

/**
 * Serialize fused operations over every path a mutation touches. Keys are
 * acquired one at a time in sorted order, so two calls sharing any targets
 * always take them in the same order and cannot deadlock.
 */
export async function withFusedFileQueues<T>(filePaths: readonly string[], work: () => Promise<T>): Promise<T> {
	const keys = [...new Set(await Promise.all(filePaths.map(canonicalQueueKey)))].sort();
	const acquire = (index: number): Promise<T> => {
		const key = keys[index];
		return key === undefined ? work() : withQueueKey(key, () => acquire(index + 1));
	};
	return acquire(0);
}
