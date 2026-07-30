import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import * as path from "node:path";
import * as fs from "node:fs";

/**
 * Execute a `mags-codedev` CLI command and return stdout+stderr as a string.
 *
 * Uses `Promise.withResolvers()` for linear control flow.
 */
function runMags(args: string[], cwd?: string): Promise<string> {
	const { promise, resolve, reject } = Promise.withResolvers<string>();

	const child = spawn("mags-codedev", args, {
		cwd: cwd ?? process.cwd(),
		env: { ...process.env },
	});

	const stdout: Buffer[] = [];
	const stderr: Buffer[] = [];

	child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
	child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));

	child.on("close", (code) => {
		const output = Buffer.concat([...stdout, ...stderr]).toString("utf-8");
		if (code !== 0) {
			reject(new Error(`mags-codedev exited with code ${code}\n${output}`));
		} else {
			resolve(output);
		}
	});

	child.on("error", reject);

	return promise;
}

/**
 * Spawn `mags-codedev build --json` and stream JSONL events to `onEvent`.
 *
 * Each line of stdout is parsed as a JSON object and passed to `onEvent`.
 * Returns the combined output string when the process exits.
 *
 * Uses `Promise.withResolvers()` and handles cancellation via `signal`.
 */
function runMagsJson(
	args: string[],
	cwd: string,
	signal: AbortSignal | undefined,
	onEvent: (event: Record<string, unknown>) => void,
	onUpdate?: (text: string) => void,
): Promise<string> {
	const { promise, resolve, reject } = Promise.withResolvers<string>();

	const child = spawn("mags-codedev", args, { cwd, env: { ...process.env } });
	const chunks: string[] = [];
	let buffer = "";

	const cleanup = () => {
		child.stdout.removeAllListeners();
		child.stderr.removeAllListeners();
		child.removeAllListeners();
	};

	child.stdout.on("data", (chunk: Buffer) => {
		const text = chunk.toString("utf-8");
		chunks.push(text);
		buffer += text;

		const lines = buffer.split("\n");
		buffer = lines.pop() ?? "";

		for (const line of lines) {
			if (!line.trim()) continue;
			try {
				const event = JSON.parse(line) as Record<string, unknown>;
				onEvent(event);

				// Stream progress to the OMP TUI
				if (onUpdate && event.event === "module_step") {
					const loc = event.location as string;
					const step = event.step as string;
					const iter = event.iteration as number;
					onUpdate(`${loc}: ${step} (iter ${iter})`);
				}
			} catch {
				// Non-JSON line (e.g. a warning) — skip
			}
		}
	});

	child.stderr.on("data", (chunk: Buffer) => {
		chunks.push(chunk.toString("utf-8"));
	});

	child.on("close", (code) => {
		const output = chunks.join("");
		if (code !== 0) {
			reject(new Error(`mags-codedev exited with code ${code}\n${output}`));
		} else {
			resolve(output);
		}
	});

	child.on("error", (err) => {
		cleanup();
		reject(err);
	});

	if (signal) {
		signal.addEventListener("abort", () => {
			child.kill("SIGTERM");
		}, { once: true });
	}

	return promise;
}

/**
 * Format a build_end event as a human-readable summary string.
 */
function formatBuildSummary(event: Record<string, unknown>): string {
	const succeeded = (event.succeeded as number) ?? 0;
	const failed = (event.failed as number) ?? 0;
	const blocked = (event.blocked as number) ?? 0;
	const tokensIn = (event.tokens_in as number) ?? 0;
	const tokensOut = (event.tokens_out as number) ?? 0;

	const lines: string[] = [];
	if (failed > 0) {
		lines.push(`Build completed with ${failed} failure(s).`);
		if (blocked > 0) {
			lines.push(`${blocked} module(s) blocked by failed dependencies.`);
		}
	} else {
		lines.push("Build cycle complete! All modules built successfully.");
	}
	lines.push(`Tokens: ${tokensIn + tokensOut} (in: ${tokensIn}, out: ${tokensOut})`);
	return lines.join("\n");
}

export default function magsExtension(pi: ExtensionAPI) {
	const { z } = pi.zod;

	pi.setLabel("MAGs-CodeDev");

	// ------------------------------------------------------------------
	// init — initialize workspace, create AGENT.md and manifest
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_init",
		label: "MAGs Init",
		description:
			"Initialize a MAGs-CodeDev workspace: create AGENT.md, manifest.json, " +
			".gitignore, SQLite database, and config. Use interactive=false for non-interactive mode.",
		parameters: z.object({
			manifest_path: z
				.string()
				.optional()
				.describe("Path to the manifest JSON file (default: <base_dir>/manifest.json)"),
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
			interactive: z
				.boolean()
				.optional()
				.describe(
					"Use AI to interactively design the project structure (default: true)",
				),
		}),
		async execute(_id, params) {
			const args: string[] = ["init"];
			if (params.manifest_path) args.push("--manifest", params.manifest_path);
			if (params.config_path) args.push("--config", params.config_path);
			if (params.interactive === false) args.push("--non-interactive");
			const output = await runMags(args);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "init" },
			};
		},
	});

	// ------------------------------------------------------------------
	// build — run the multi-agent build loop
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_build",
		label: "MAGs Build",
		description:
			"Run the multi-agent build loop: generate code, tests, run lint/type-check, " +
			"iterate until all modules pass or max iterations. Use --module to rerun a single task " +
			"after editing its spec or fixing a failure. Use json=true for streaming JSONL output.",
		parameters: z.object({
			manifest_path: z
				.string()
				.optional()
				.describe("Path to manifest.json (default: <base_dir>/manifest.json)"),
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
			module: z
				.string()
				.optional()
				.describe(
					"Build only this module (by location). Forces a rebuild even if already built.",
				),
			force_fresh: z
				.boolean()
				.optional()
				.describe("Discard existing worktrees/branches, rebuild from scratch."),
			skip_validation: z
				.boolean()
				.optional()
				.describe("Skip the pre-build LLM connection check."),
			verbose: z
				.number()
				.int()
				.min(0)
				.max(2)
				.optional()
				.describe("Verbosity: 0=info, 1=debug (full LLM chat + file contents), 2=trace."),
			json: z
				.boolean()
				.optional()
				.describe(
					"Emit JSONL status events to stdout for streaming progress. " +
						"Recommended when called from OMP for live status updates.",
				),
		}),
		async execute(_id, params, signal, onUpdate) {
			const cwd = process.cwd();
			const args: string[] = ["build"];

			// Default to JSON mode for OMP — gives structured, streamable output.
			const useJson = params.json !== false;
			if (useJson) args.push("--json");
			if (params.manifest_path) args.push("--manifest", params.manifest_path);
			if (params.config_path) args.push("--config", params.config_path);
			if (params.force_fresh) args.push("--force-fresh");
			if (params.skip_validation) args.push("--skip-validation");
			if (params.verbose) args.push("-" + "v".repeat(params.verbose));
			if (params.module) args.push("--module", params.module);

			if (useJson) {
				let finalEvent: Record<string, unknown> | null = null;

				const output = await runMagsJson(
					args, cwd, signal,
					(event) => {
						if (event.event === "build_end") {
							finalEvent = event;
						}
					},
					onUpdate
						? (text: string) => {
								onUpdate({ content: [{ type: "text", text }] });
							}
						: undefined,
				);

				const summary = finalEvent
					? formatBuildSummary(finalEvent)
					: "Build completed (no final event received).";

				return {
					content: [{ type: "text", text: summary }],
					details: finalEvent ?? { command: "build", json: true },
				};
			}

			const output = await runMags(args, cwd);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "build", json: false },
			};
		},
	});

	// ------------------------------------------------------------------
	// test — run all project tests in the isolated environment
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_test",
		label: "MAGs Test",
		description:
			"Run pytest, flake8, mypy, and bandit for all modules in the configured " +
			"container environment (Docker, Apptainer, or local).",
		parameters: z.object({
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
			verbose: z
				.number()
				.int()
				.min(0)
				.max(2)
				.optional()
				.describe("Verbosity: 0=info, 1=debug, 2=trace."),
		}),
		async execute(_id, params) {
			const args: string[] = ["test"];
			if (params.config_path) args.push("--config", params.config_path);
			if (params.verbose) args.push("-" + "v".repeat(params.verbose));
			const output = await runMags(args);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "test" },
			};
		},
	});

	// ------------------------------------------------------------------
	// debug — pass an error trace to the LLM for automatic fixing
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_debug",
		label: "MAGs Debug",
		description:
			"Pass an error trace or bug description to the LLM for automatic fixing " +
			"of a module. Accepts a log file path (auto-detects module from hash). " +
			"After the fix, use mags_build --module to rerun the task.",
		parameters: z.object({
			error_msg: z
				.string()
				.describe(
					"The error trace to fix, or a path to an error trace logfile.",
				),
			module_location: z
				.string()
				.optional()
				.describe(
					"The module location in manifest.json to apply the fix to " +
						"(auto-detected from log file hash if omitted).",
				),
			manifest_path: z
				.string()
				.optional()
				.describe("Path to manifest.json (default: <base_dir>/manifest.json)"),
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
			verbose: z
				.number()
				.int()
				.min(0)
				.max(2)
				.optional()
				.describe("Verbosity: 0=info, 1=debug, 2=trace."),
		}),
		async execute(_id, params) {
			const args: string[] = ["debug", params.error_msg];
			if (params.module_location) args.push("--mod", params.module_location);
			if (params.manifest_path) args.push("--manifest", params.manifest_path);
			if (params.config_path) args.push("--config", params.config_path);
			if (params.verbose) args.push("-" + "v".repeat(params.verbose));
			const output = await runMags(args);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "debug" },
			};
		},
	});

	// ------------------------------------------------------------------
	// tokens — show token usage summary
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_tokens",
		label: "MAGs Tokens",
		description:
			"Display token usage statistics broken down by role (coder, tester, etc.) " +
			"and model, with input/output/total columns.",
		parameters: z.object({}),
		async execute(_id) {
			const output = await runMags(["tokens"]);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "tokens" },
			};
		},
	});

	// ------------------------------------------------------------------
	// list-models — show available models per provider
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_list_models",
		label: "MAGs List Models",
		description:
			"List available models from OpenAI, Anthropic, Google, and Mistral " +
			"providers based on configured API keys.",
		parameters: z.object({
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
		}),
		async execute(_id, params) {
			const args: string[] = ["list-models"];
			if (params.config_path) args.push("--config", params.config_path);
			const output = await runMags(args);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "list-models" },
			};
		},
	});

	// ------------------------------------------------------------------
	// clean — remove cache files, logs, and worktrees
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_clean",
		label: "MAGs Clean",
		description:
			"Remove all generated MAGs-CodeDev artifacts: cache.db, workflow.log, " +
			"worktree directories, and hash log files. Use force=true to skip confirmation.",
		parameters: z.object({
			force: z
				.boolean()
				.optional()
				.describe("Skip confirmation prompt (default: false)"),
		}),
		async execute(_id, params) {
			const args: string[] = ["clean"];
			if (params.force) args.push("--force");
			const output = await runMags(args);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "clean" },
			};
		},
	});

	// ------------------------------------------------------------------
	// Session start: auto-detect MAGs-CodeDev workspace
	// ------------------------------------------------------------------
	pi.on("session_start", async (_event, ctx) => {
		const manifest = path.join(ctx.cwd, ".mags-codedev", "manifest.json");
		if (fs.existsSync(manifest)) {
			ctx.ui.notify("MAGs-CodeDev workspace detected", "info");
		}
	});
}
