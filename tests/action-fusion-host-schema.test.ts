/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */
import { describe, expect, it } from "vitest";
import { Type } from "typebox";
import { fusedToolParameters } from "../src/sol-pi/extensions/action-fusion/index.ts";

/**
 * A Pi-compatible host that describes tool parameters with its own schema value:
 * callable rather than a plain object, carrying no `properties`, and exposing the
 * contract only through `toJsonSchema()`. This is the shape OMP 18.13 publishes,
 * and its properties are plain JSON, not TypeBox values.
 */
function hostSchema(json: { properties: Record<string, unknown>; required?: string[] }): unknown {
	const schema = (value: unknown) => value;
	schema.toJsonSchema = () => json;
	return schema;
}

describe("fusedToolParameters", () => {
	it("keeps the TypeBox properties Pi publishes", () => {
		const parameters = Type.Object({
			path: Type.String(),
			content: Type.String(),
		});

		const fused = fusedToolParameters(parameters, "run next");
		expect(Object.keys(fused?.properties ?? {})).toEqual(["path", "content", "then_run"]);
		expect(fused?.required).toEqual(["path", "content"]);
	});

	it("rebuilds the host's plain JSON properties and their required list", () => {
		const parameters = hostSchema({
			properties: {
				path: { type: "string", description: "file path" },
				content: { type: "string", description: "file content" },
			},
			required: ["path", "content"],
		});

		const fused = fusedToolParameters(parameters, "run next");

		expect(fused?.required).toEqual(["path", "content"]);
		expect(fused?.properties.then_run).toMatchObject({ type: "object", required: ["command"] });
	});

	it("keeps a host-optional parameter optional, without wrapping it in TypeBox", () => {
		// OMP's write tool requires `path` and leaves `content` optional, as plain
		// JSON. `Type.Optional` throws on that value under the TypeBox OMP ships,
		// so the fused schema must express optionality through `required` alone.
		const parameters = hostSchema({
			properties: { path: { type: "string" }, content: { type: "string" } },
			required: ["path"],
		});

		const fused = fusedToolParameters(parameters, "run next");

		expect(fused?.required).toEqual(["path"]);
		expect(fused?.properties.content).toEqual({ type: "string" });
	});

	it("preserves optional fields in direct JSON Schema parameters", () => {
		const fused = fusedToolParameters({
			properties: { path: { type: "string" }, encoding: { type: "string" } },
			required: ["path"],
		}, "run next");
		expect(fused?.required).toEqual(["path"]);
	});

	it("reports nothing for throwing and malformed host schemas, leaving the tool replaced", () => {
		const throwsProperties = Object.defineProperty({}, "properties", {
			get() { throw new Error("unsupported host schema"); },
		});
		const cases = [
			throwsProperties,
			{ toJsonSchema() { throw new Error("unsupported host schema"); } },
			{ toJsonSchema() { return null; } },
			{ toJsonSchema() { return { properties: { path: true } }; } },
			{ properties: { path: false } },
			{ type: "object" },
			undefined,
		];
		for (const parameters of cases) {
			expect(fusedToolParameters(parameters, "run next")).toBeUndefined();
		}
	});
});
