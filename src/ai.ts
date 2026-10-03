import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export type ModelInstance = ReturnType<
    ReturnType<
        | typeof createGoogleGenerativeAI
        | typeof createOpenAI
        | typeof createAnthropic
    >
>;

function isOpenCodeModelName(modelName: string) {
    return (
        modelName === "opencode" ||
        modelName.startsWith("opencode:") ||
        modelName.startsWith("opencode/")
    );
}

function resolveOpenCodeModelName(modelName: string) {
    const defaultModel = process.env.OPENCODE_MODEL || "big-pickle";
    return modelName === "opencode"
        ? defaultModel
        : modelName.replace(/^opencode[:/]/, "");
}

function stripOpenCodeNoise(raw: string): string {
    let text = raw.replace(/\x1b\[[0-9;]*m/g, "");
    text = text.replace(/^> build · .*\n?/gm, "");
    text = text.replace(/<think>[\s\S]*?<\/think>/gi, "");
    text = text.replace(/```(?:json)?\s*([\s\S]*?)```/gi, (_, block) => block);
    return text.trim();
}

function unwrapEchoedSchema(text: string): string {
    const first = text.indexOf("{");
    const last = text.lastIndexOf("}");
    if (first !== -1 && last !== -1 && last > first) {
        return text.slice(first, last + 1);
    }
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fenced?.[1]) {
        return unwrapEchoedSchema(fenced[1]);
    }
    return text;
}

function supportsOpenCodeJsonOutput(): boolean {
    try {
        const result = spawnSync("opencode", ["run", "--help"], {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "pipe"],
        });

        if (result.error && result.error.message.includes("ENOENT")) {
            return false;
        }

        const output = `${result.stdout ?? ""}\n${result.stderr ?? ""}`;
        return /--(?:output|format)\b/i.test(output) || /json/i.test(output);
    } catch {
        return false;
    }
}

function parseOpencodeEvents(raw: string) {
    let text = "";
    let usage: { input?: number; output?: number; total?: number } = {};

    for (const line of raw.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("{")) continue;

        try {
            const event = JSON.parse(trimmed);
            if (event.type === "text" && typeof event.part?.text === "string") {
                text += event.part.text;
            } else if (event.type === "step_finish" && event.part?.tokens) {
                const tokens = event.part.tokens;
                usage = {
                    input: tokens.input,
                    output: tokens.output,
                    total: tokens.total,
                };
            }
        } catch {
            // ignore non-JSON noise
        }
    }

    return { text, usage };
}

function getOpenCodeModelConfig(modelName: string): ModelInstance {
    const resolvedModel = resolveOpenCodeModelName(modelName);
    const cliModel = resolvedModel.includes("/")
        ? resolvedModel
        : `opencode/${resolvedModel}`;
    const supportsJson = supportsOpenCodeJsonOutput();

    return {
        specificationVersion: "v2",
        provider: "opencode",
        modelId: resolvedModel,
        supportedUrls: {},
        async doGenerate({ prompt, system, messages, responseFormat, tools }: any) {
            // OpenCode CLI does not support tool calling
            if (tools && Object.keys(tools).length > 0) {
                throw new Error(
                    "OpenCode provider does not support tool calling. Use a model like Gemini, OpenAI, or Anthropic for --model that supports the agentic features.",
                );
            }

            const requestBody = messages ?? [
                ...(system ? [{ role: "system", content: system }] : []),
                { role: "user", content: prompt ?? "" },
            ];

            const promptText = requestBody
                .map((message: any) => {
                    const content =
                        typeof message.content === "string"
                            ? message.content
                            : JSON.stringify(message.content ?? "");
                    return `${message.role ?? "user"}:\n${content}`;
                })
                .join("\n\n");

            const finalPrompt = responseFormat?.type === "json"
                ? `${promptText}\n\nReturn only valid JSON. Do not wrap it in markdown fences.\nSchema:\n${JSON.stringify(responseFormat.schema ?? {}, null, 2)}`
                : promptText;

            let attempt = 0;
            while (attempt <= 1) {
                const cwd = mkdtempSync(join(tmpdir(), "autospec-opencode-"));
                try {
                    const args = ["run", "-m", cliModel];
                    if (supportsJson) {
                        args.push("--format", "json");
                    }
                    args.push(finalPrompt || " ");

                    // Minimal environment: only PATH and HOME to avoid leaking secrets
                    const minimalEnv: Record<string, string> = {
                        PATH: process.env.PATH || "/usr/local/bin:/usr/bin:/bin",
                        HOME: process.env.HOME || "/tmp",
                        NO_COLOR: "1",
                    };

                    const child = spawn("opencode", args, {
                        cwd,
                        stdio: ["ignore", "pipe", "pipe"],
                        env: minimalEnv,
                    });

                    let stdout = "";
                    let stderr = "";
                    const timer = setTimeout(() => {
                        child.kill("SIGKILL");
                    }, 120_000);

                    const exitCode = await new Promise<number | null>((resolve, reject) => {
                        child.stdout?.on("data", (chunk) => {
                            stdout += chunk.toString();
                        });
                        child.stderr?.on("data", (chunk) => {
                            stderr += chunk.toString();
                        });
                        child.on("error", (error) => reject(error));
                        child.on("close", (code) => resolve(code ?? null));
                    });

                    clearTimeout(timer);
                    rmSync(cwd, { recursive: true, force: true });

                    const cliOutput = stdout || stderr || "";
                    const { text: rawText, usage } = parseOpencodeEvents(cliOutput);

                    let text = rawText
                        .replace(/\x1b\[[0-9;]*m/g, "")
                        .replace(/<think>[\s\S]*?<\/think>/g, "")
                        .trim();

                    if (!text) {
                        throw new Error(
                            `opencode: aucune réponse texte. Début de sortie: ${cliOutput.slice(0, 400)}`,
                        );
                    }

                    if (responseFormat?.type === "json") {
                        const start = text.indexOf("{");
                        const end = text.lastIndexOf("}");
                        if (start !== -1 && end > start) {
                            text = text.slice(start, end + 1);
                        }
                        text = unwrapEchoedSchema(text);
                    }

                    console.info(
                        `[opencode] model=${resolvedModel} exitCode=${exitCode ?? "n/a"}`,
                    );

                    if (exitCode !== 0) {
                        throw new Error(
                            `opencode CLI failed for ${resolvedModel} (exit ${exitCode ?? "unknown"}): ${stderr || cliOutput || "No output"}`,
                        );
                    }

                    return {
                        content: [{ type: "text", text }],
                        finishReason: "stop",
                        usage: {
                            inputTokens: usage.input,
                            outputTokens: usage.output,
                            totalTokens: usage.total,
                        },
                        warnings: [],
                        request: { body: promptText },
                        response: { body: { raw: cliOutput } },
                    };
                } catch (error) {
                    rmSync(cwd, { recursive: true, force: true });
                    if (attempt === 1) {
                        const message =
                            error instanceof Error ? error.message : String(error);
                        if (message.includes("ENOENT")) {
                            throw new Error("opencode CLI introuvable, installe opencode-ai");
                        }
                        throw new Error(`OpenCode provider failed after retry: ${message}`);
                    }
                    attempt += 1;
                }
            }

            throw new Error(`OpenCode provider failed unexpectedly for ${resolvedModel}`);
        },
        async doStream() {
            throw new Error("Streaming is not supported for opencode provider.");
        },
    } as any as ModelInstance;
}

function getCloudflareToken(apiKey?: string) {
    return (
        apiKey ||
        process.env.CLOUDFLARE_AUTH_TOKEN ||
        process.env.CLOUDFLARE_API_TOKEN ||
        process.env.CLOUDFLARE_API_KEY
    );
}

function wrapModelWithToolCheck(model: ModelInstance, providerName: string): ModelInstance {
    return {
        ...model,
        async doGenerate(options: any) {
            if (options.tools && Object.keys(options.tools).length > 0) {
                throw new Error(
                    `${providerName} provider does not support tool calling. Use a model like Gemini, OpenAI, or Anthropic for --model that supports the agentic features.`,
                );
            }
            return model.doGenerate(options);
        },
        async doStream(options: any) {
            if (options.tools && Object.keys(options.tools).length > 0) {
                throw new Error(
                    `${providerName} provider does not support tool calling. Use a model like Gemini, OpenAI, or Anthropic for --model that supports the agentic features.`,
                );
            }
            return model.doStream(options);
        },
    } as any as ModelInstance;
}

function getCloudflareModelConfig(modelName: string, apiKey?: string): ModelInstance {
    const token = getCloudflareToken(apiKey);
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;

    if (!token) {
        throw new Error("CLOUDFLARE_AUTH_TOKEN manquant.");
    }

    if (!accountId) {
        throw new Error("CLOUDFLARE_ACCOUNT_ID manquant.");
    }

    const defaultModel =
        process.env.CLOUDFLARE_MODEL || "@cf/meta/llama-3.1-8b-fast-v2";

    const resolvedModel =
        modelName === "cloudflare" ? defaultModel : modelName.replace(/^cloudflare[:/]/, "");

    const baseModel = createOpenAI({
        apiKey: token,
        baseURL: `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`,
    }).chat(resolvedModel);

    return wrapModelWithToolCheck(baseModel, "Cloudflare");
}

export function getModel({
    modelName,
    apiKey,
}: {
    modelName: string;
    apiKey?: string;
}): ModelInstance {
    // Handle Cloudflare models
    if (
        modelName === "cloudflare" ||
        modelName.startsWith("cloudflare:") ||
        modelName.startsWith("cloudflare/") ||
        modelName.startsWith("@cf/")
    ) {
        return getCloudflareModelConfig(modelName, apiKey);
    }

    // Handle OpenCode models
    if (isOpenCodeModelName(modelName)) {
        return getOpenCodeModelConfig(modelName);
    }

    // Handle Gemini models (supports dynamic model names like "gemini-3.1-flash-lite")
    if (modelName === "gemini-2.5-flash" || modelName.startsWith("gemini-")) {
        return createGoogleGenerativeAI({
            apiKey: apiKey || process.env.GOOGLE_GENERATIVE_AI_API_KEY,
        })(modelName);
    }

    // Handle predefined models
    const configs: Record<string, () => ModelInstance> = {
        "gpt-5.4": () =>
            createOpenAI({
                apiKey: apiKey || process.env.OPENAI_API_KEY,
            })("gpt-5.4"),
        "claude-opus-4-6": () =>
            createAnthropic({
                apiKey: apiKey || process.env.ANTHROPIC_API_KEY,
            })("claude-opus-4-6"),
    };

    const factory = configs[modelName];
    if (!factory) {
        throw new Error(
            `Unknown model: ${modelName}. Supported: gpt-5.4, claude-opus-4-6, gemini-2.5-flash, gemini-3.1-flash-lite (or any gemini-*), cloudflare, cloudflare:<model>, @cf/<model>, opencode, opencode:<model>`,
        );
    }
    return factory();
}
