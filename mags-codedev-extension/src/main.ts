import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { spawn } from "node:child_process";

/**
 * Execute a `mags-codedev` CLI command and return stdout+stderr as a string.
 *
 * @param args - CLI arguments to pass after `mags-codedev`
 * @returns Promise<string> - combined stdout/stderr
 */
function runMags(args: string[]): Promise<string> {
	return new Promise((resolve, reject) => {
		const child = spawn("mags-codedev", args, {
			cwd: process.cwd(),
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
	});
}

export default function magsExtension(pi: ExtensionAPI) {
	const { z } = pi.zod;

	// ------------------------------------------------------------------
	// init — initialize workspace, create AGENT.md and manifest
	// ------------------------------------------------------------------
	pi.registerTool({
		name: "mags_init",
		label: "MAGs Init",
		description:
			"Initialize a MAGs-CodeDev workspace: create AGENT.md, manifest.json, " +
			".gitignore, and SQLite database. Set interactive=false for non-interactive mode.",
		parameters: z.object({
			manifest_path: z
				.string()
				.optional()
				.describe("Path to the manifest JSON file (default: manifest.json)"),
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
			if (params.manifest_path) {
				args.push("--manifest", params.manifest_path);
			}
			if (params.config_path) {
				args.push("--config", params.config_path);
			}
			if (params.interactive === false) {
				args.push("--no-interactive");
			}
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
			"iterate until all modules pass or max_iterations is reached.",
		parameters: z.object({
			manifest_path: z
				.string()
				.optional()
				.describe("Path to manifest.json (default: manifest.json)"),
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
			parallelism: z
				.number()
				.int()
				.min(1)
				.optional()
				.describe("Number of modules to process in parallel (default: 2)"),
			max_iterations: z
				.number()
				.int()
				.min(1)
				.optional()
				.describe("Maximum fix iterations per module (default: 3)"),
			force_fresh: z
				.boolean()
				.optional()
				.describe("Start from scratch, discarding previous worktree state"),
		}),
		async execute(_id, params) {
			const args: string[] = ["build"];
			if (params.manifest_path) {
				args.push("--manifest", params.manifest_path);
			}
			if (params.config_path) {
				args.push("--config", params.config_path);
			}
			if (params.parallelism) {
				args.push("--parallelism", String(params.parallelism));
			}
			if (params.max_iterations) {
				args.push("--max-iterations", String(params.max_iterations));
			}
			if (params.force_fresh) {
				args.push("--force-fresh");
			}
			const output = await runMags(args);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "build" },
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
			"Run pytest, flake8, mypy, and bandit for all modules in an isolated " +
			"environment (Docker, Apptainer, or local).",
		parameters: z.object({
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
		}),
		async execute(_id, params) {
			const args: string[] = ["test"];
			if (params.config_path) {
				args.push("--config", params.config_path);
			}
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
			"of a module. Can accept a log file path as the error message.",
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
						"(optional if providing a log file — auto-detected from hash)",
				),
			manifest_path: z
				.string()
				.optional()
				.describe("Path to manifest.json (default: manifest.json)"),
			config_path: z
				.string()
				.optional()
				.describe("Path to config.yaml (default: auto-detected)"),
		}),
		async execute(_id, params) {
			const args: string[] = ["debug", params.error_msg];
			if (params.module_location) {
				args.push("--module", params.module_location);
			}
			if (params.manifest_path) {
				args.push("--manifest", params.manifest_path);
			}
			if (params.config_path) {
				args.push("--config", params.config_path);
			}
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
			"and model.",
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
		parameters: z.object({}),
		async execute(_id) {
			const output = await runMags(["list-models"]);
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
			if (params.force) {
				args.push("--force");
			}
			const output = await runMags(args);
			return {
				content: [{ type: "text", text: output }],
				details: { command: "clean" },
			};
		},
	});
}
