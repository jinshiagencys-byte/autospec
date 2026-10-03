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

function getOpenCodeModelConfig(modelName: string): ModelInstance {
    const resolvedModel = resolveOpenCodeModelName(modelName);
    const supportsJson = supportsOpenCodeJsonOutput();

    const runModel = async (prompt: string, system?: string, responseFormat?: any) => {
        let attempt = 0;
        while (attempt <= 1) {
            const cwd = mkdtempSync(join(tmpdir(), "autospec-opencode-"));
            try {
                const args = ["run", "-m", resolvedModel];
                if (supportsJson) {
                    args.push("--output", "json");
                }

                const child = spawn("opencode", args, {
                    cwd,
                    stdio: ["pipe", "pipe", "pipe"],
                    env: {
                        ...process.env,
                        NO_COLOR: "1",
                    },
                });

                let stdout = "";
                let stderr = "";

                child.stdout?.on("data", (chunk) => {
                    stdout += chunk.toString();
                });

                child.stderr?.on("data", (chunk) => {
                    stderr += chunk.toString();
                });

                const timeoutMs = 120_000;
                const timer = setTimeout(() => {
                    child.kill("SIGKILL");
                }, timeoutMs);

                const now = Date.now();
                let rawPrompt = [system ? `SYSTEM:\n${system}` : "", prompt ? `USER:\n${prompt}` : ""]
                    .filter(Boolean)
                    .join("\n\n");

                if (responseFormat?.type === "json") {
                    const schemaText = responseFormat.schema
                        ? JSON.stringify(responseFormat.schema, null, 2)
                        : "{}";
                    rawPrompt += `\n\nReturn only valid JSON. Do not wrap it in markdown fences.\nSchema:\n${schemaText}`;
                }

                child.stdin?.write(rawPrompt);
                child.stdin?.end();

                const exitCode = await new Promise<number | null>((resolve, reject) => {
                    child.on("error", (error) => {
                        reject(error);
                    });
                    child.on("close", (code) => {
                        resolve(code ?? null);
                    });
                });

                clearTimeout(timer);
                rmSync(cwd, { recursive: true, force: true });

                const outputText = stripOpenCodeNoise(stdout || stderr || "");
                const normalizedOutput =
                    responseFormat?.type === "json"
                        ? unwrapEchoedSchema(outputText)
                        : outputText || "";

                console.info(
                    `[opencode] model=${resolvedModel} durationMs=${Date.now() - now} exitCode=${exitCode ?? "n/a"}`,
                );

                if (exitCode !== 0) {
                    throw new Error(
                        `opencode CLI failed for ${resolvedModel} (exit ${exitCode ?? "unknown"}): ${stderr || outputText || "No output"}`,
                    );
                }

                return {
                    content: normalizedOutput || "",
                    finishReason: "stop",
                    usage: {
                        inputTokens: undefined,
                        outputTokens: undefined,
                    },
                    warnings: [],
                    request: {
                        model: resolvedModel,
                        provider: "opencode",
                    },
                    response: {
                        text: outputText || "",
                        exitCode,
                    },
                };
            } catch (error) {
                rmSync(cwd, { recursive: true, force: true });
                if (attempt === 1) {
                    const message =
                        error instanceof Error
                            ? error.message
                            : String(error);
                    if (message.includes("ENOENT")) {
                        throw new Error("opencode CLI introuvable, installe opencode-ai");
                    }
                    throw new Error(`OpenCode provider failed after retry: ${message}`);
                }
                attempt += 1;
            }
        }

        throw new Error(`OpenCode provider failed unexpectedly for ${resolvedModel}`);
    };

    return {
        specificationVersion: "v2",
        provider: "opencode",
        modelId: resolvedModel,
        supportedUrls: {},
        doGenerate: async ({ prompt, system, messages, responseFormat }: any) => {
            const requestBody = messages ?? [
                ...(system ? [{ role: "system", content: system }] : []),
                { role: "user", content: prompt ?? "" },
            ];

            let attempt = 0;
            while (attempt <= 1) {
                const cwd = mkdtempSync(join(tmpdir(), "autospec-opencode-"));
                try {
                    const args = ["run", "-m", resolvedModel];
                    if (supportsOpenCodeJsonOutput()) {
                        args.push("--output", "json");
                    }

                    const child = spawn("opencode", args, {
                        cwd,
                        stdio: ["pipe", "pipe", "pipe"],
                        env: {
                            ...process.env,
                            NO_COLOR: "1",
                        },
                    });

                    let stdout = "";
                    let stderr = "";
                    child.stdout?.on("data", (chunk) => {
                        stdout += chunk.toString();
                    });
                    child.stderr?.on("data", (chunk) => {
                        stderr += chunk.toString();
                    });

                    const timer = setTimeout(() => {
                        child.kill("SIGKILL");
                    }, 120_000);

                    const rawPrompt = requestBody
                        .map((m: any) => {
                            const content =
                                typeof m.content === "string"
                                    ? m.content
                                    : JSON.stringify(m.content ?? "");
                            return `${m.role ?? "user"}:\n${content}`;
                        })
                        .join("\n\n");

                    const finalPrompt = responseFormat?.type === "json"
                        ? `${rawPrompt}\n\nReturn only valid JSON. Do not wrap it in markdown fences.\nSchema:\n${JSON.stringify(responseFormat.schema ?? {}, null, 2)}`
                        : rawPrompt;

                    child.stdin?.write(finalPrompt);
                    child.stdin?.end();

                    const exitCode = await new Promise<number | null>((resolve, reject) => {
                        child.on("error", (error) => reject(error));
                        child.on("close", (code) => resolve(code ?? null));
                    });

                    clearTimeout(timer);
                    rmSync(cwd, { recursive: true, force: true });

                    const outputText = stripOpenCodeNoise(stdout || stderr || "");
                    const normalizedText =
                        responseFormat?.type === "json"
                            ? unwrapEchoedSchema(outputText)
                            : outputText;

                    console.info(
                        `[opencode] model=${resolvedModel} exitCode=${exitCode ?? "n/a"}`,
                    );

                    if (exitCode !== 0) {
                        throw new Error(
                            `opencode CLI failed for ${resolvedModel} (exit ${exitCode ?? "unknown"}): ${stderr || outputText || "No output"}`,
                        );
                    }

                    return {
                        content: [{ type: "text", text: normalizedText || "" }],
                        finishReason: "stop",
                        usage: {
                            inputTokens: undefined,
                            outputTokens: undefined,
                            totalTokens: undefined,
                        },
                        warnings: [],
                        request: { body: requestBody },
                        response: { body: { raw: normalizedText || "" } },
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
        doStream: async () => {
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

    return createOpenAI({
        apiKey: token,
        baseURL: `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/v1`,
    }).chat(resolvedModel);
}

export function getModel({
    modelName,
    apiKey,
}: {
    modelName: string;
    apiKey?: string;
}): ModelInstance {
    const configs: Record<string, () => ModelInstance> = {
        "gpt-5.4": () =>
            createOpenAI({
                apiKey: apiKey || process.env.OPENAI_API_KEY,
            })("gpt-5.4"),
        "claude-opus-4-6": () =>
            createAnthropic({
                apiKey: apiKey || process.env.ANTHROPIC_API_KEY,
            })("claude-opus-4-6"),
        "gemini-2.5-flash": () =>
            createGoogleGenerativeAI({
                apiKey: apiKey || process.env.GOOGLE_GENERATIVE_AI_API_KEY,
            })("gemini-2.5-flash"),
        cloudflare: () => getCloudflareModelConfig(modelName, apiKey),
        opencode: () => getOpenCodeModelConfig(modelName),
    };

    if (
        modelName === "cloudflare" ||
        modelName.startsWith("cloudflare:") ||
        modelName.startsWith("cloudflare/") ||
        modelName.startsWith("@cf/")
    ) {
        return getCloudflareModelConfig(modelName, apiKey);
    }

    if (isOpenCodeModelName(modelName)) {
        return getOpenCodeModelConfig(modelName);
    }

    const factory = configs[modelName];
    if (!factory) {
        throw new Error(
            `Unknown model: ${modelName}. Supported: ${Object.keys(configs).join(", ")}, cloudflare, cloudflare:<model>, @cf/<model>, opencode, opencode:<model>`,
        );
    }
    return factory();
}
