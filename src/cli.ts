#!/usr/bin/env node
import { main } from "./index.js";
import { input, select } from "@inquirer/prompts";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const { version } = require("../../package.json");

const args = process.argv.slice(2);

const getArgValue = <T>(argName: string, defaultValue: T) => {
    const index = args.indexOf(argName);
    return index !== -1 ? args[index + 1] : defaultValue;
};

if (args.includes("--help") || args.includes("-h")) {
    console.log(`
    Usage: npx autospecai --url <url> [--model <model>] [--plan_model <model>] [--spec_limit <limit>] [--specFile <file>] [--trajectories-path <path>] [--help | -h]

    Required:
    --url <url>          The target URL to run the autospec tests against.

    Optional:
    --help, -h           Show this help message.
    --spec_limit <limit> The max number of specs to generate. Default 10.
    --trajectories-path <path> Path to store trajectories, videos, and specs. Default: ./trajectories
    --model <model>      The model to use for test execution (with tool calling support)
                          * "claude-opus-4-6" (default)
                          * "gpt-5.4"
                          * "gemini-2.5-flash" or "gemini-3.1-flash-lite"
                          * "cloudflare" or "cloudflare:<model-name>"
                          * "opencode" or "opencode:<model-name>" (default: big-pickle)
    --plan_model <model> The model to use for test planning only (no tool calling required).
                          If not provided, defaults to --model.
                          Useful for cheaper models: "gemini-2.5-flash", etc.
    --apikey <key>       The relevant API key for the execution model's API.
                          * If not specified, we'll fall back on the
                            following environment variables:
                            * OPENAI_API_KEY (for gpt-5.4)
                            * GOOGLE_GENERATIVE_AI_API_KEY (for gemini-*)
                            * ANTHROPIC_API_KEY (for claude-*)
                            * CLOUDFLARE_AUTH_TOKEN (for cloudflare)
                            * OPENCODE_MODEL (no API key required for opencode)
    --specFile <file>    Path to the file containing specs to run.
                         Use "-" to read from stdin.
        Cloudflare requirements:
            CLOUDFLARE_ACCOUNT_ID
            CLOUDFLARE_AUTH_TOKEN
            CLOUDFLARE_MODEL (optional, default: @cf/meta/llama-3.1-8b-fast-v2)
        OpenCode requirements:
            npm install -g opencode-ai
            OPENCODE_MODEL (optional, default: big-pickle)
    `);
    process.exit(0);
}

const getInteractiveInput = async () => {
    const models = [
        "claude-opus-4-6",
        "gpt-5.4",
        "gemini-2.5-flash",
        "cloudflare",
        "opencode",
    ];

    const testUrl = await input({
        message: "Enter the target URL:",
    });

    const modelName = await select({
        message: "Choose a model:",
        choices: models.map((model) => ({ name: model, value: model })),
        default: models[0],
    });

    const specLimit = await input({
        message: "Enter the spec limit (default: 10):",
        default: "10",
    });

    const apiKey = await input({
        message: "Enter the API key:",
    });

    const specFile = await input({
        message: "Enter the spec file path (or leave blank):",
    });

    return {
        testUrl,
        modelName,
        planModelName: undefined,
        specLimit: parseInt(specLimit, 10) || 10,
        apiKey,
        specFile: specFile || undefined,
        trajectoriesPath: undefined,
    };
};

const getVars = async () => {
    if (!getArgValue("--url", null)) {
        console.warn("No URL provided. Entering interactive mode...");
        return await getInteractiveInput();
    } else {
        return {
            testUrl: getArgValue<string | undefined>("--url", undefined),
            modelName: getArgValue<string | undefined>(
                "--model",
                "claude-opus-4-6",
            ),
            planModelName: getArgValue<string | undefined>(
                "--plan_model",
                undefined,
            ),
            specLimit: getArgValue<string | number>("--spec_limit", 10),
            apiKey: getArgValue<string | undefined>("--apikey", undefined),
            specFile: getArgValue<string | undefined>("--specFile", undefined),
            trajectoriesPath: getArgValue<string | undefined>(
                "--trajectories-path",
                undefined,
            ),
        };
    }
};

const run = async () => {
    if (args.includes("--version") || args.includes("-v")) {
        console.log(`autospec version ${version}`);
        process.exit(0);
    }

    const { testUrl, modelName, planModelName, specLimit, apiKey, specFile, trajectoriesPath } = await getVars();
    if (!apiKey) {
        console.warn(
            "Warning: No API key provided. Falling back to environment variables.",
        );
    }
    const { testResults } = await main({
        testUrl,
        modelName,
        planModelName,
        specLimit:
            typeof specLimit == "string" ? parseInt(specLimit) : specLimit,
        apiKey,
        specFile,
        trajectoriesPath,
    });
    process.exit(
        testResults.every((result) => result.status === "passed") ? 0 : 1,
    );
};

run().catch(console.error);
