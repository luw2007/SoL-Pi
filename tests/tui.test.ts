/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { showSolPiSavings } from "../src/sol-pi/tui.ts";

function uiContext(mode: ExtensionContext["mode"]) {
	const notify = vi.fn();
	const setStatus = vi.fn();
	return {
		context: { mode, ui: { notify, setStatus } } as unknown as ExtensionContext,
		notify,
		setStatus,
	};
}

describe("SoL-Pi TUI savings presentation", () => {
	afterEach(() => {
		vi.useRealTimers();
	});

	it("shows and clears a TUI-only notification and footer status", () => {
		vi.useFakeTimers();
		const { context, notify, setStatus } = uiContext("tui");

		showSolPiSavings(context, "Online Context Compact", "84,026 context tokens removed");

		expect(notify).toHaveBeenCalledTimes(1);
		expect(notify.mock.calls[0]?.[1]).toBe("info");
		expect(setStatus).toHaveBeenCalledTimes(1);
		vi.advanceTimersByTime(4_000);
		expect(setStatus).toHaveBeenLastCalledWith("sol-pi-savings", undefined);
	});

	it.each(["rpc", "json", "print"] as const)("does not emit presentation in %s mode", (mode) => {
		const { context, notify, setStatus } = uiContext(mode);

		showSolPiSavings(context, "Action Fusion", "1 model round-trip avoided");

		expect(notify).not.toHaveBeenCalled();
		expect(setStatus).not.toHaveBeenCalled();
	});
});
