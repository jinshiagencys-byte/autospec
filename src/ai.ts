import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";

export type ModelInstance = ReturnType<
    ReturnType<
        | typeof createGoogleGenerativeAI
        | typeof createOpenAI
        | typeof createAnthropic
    >
>;

function getCloudflareToken(apiKey?: string) {
    return (
        apiKey ||
        process.env.CLOUDFLARE_AUTH_TOKEN ||
        process.env.CLOUDFLARE_API_TOKEN ||
        process.env.CLOUDFLARE_API_KEY
    );
}

function getCloudflarePromptText(value: unknown): string {
    if (typeof value === "string") {
        return value;
    }

    if (Array.isArray(value)) {
        return value.map(getCloudflarePromptText).join("\n");
    }

    if (value && typeof value === "object") {
        const obj = value as Record<string, unknown>;

        if (typeof obj.text === "string") {
            return obj.text;
        }

        if (typeof obj.content === "string") {
            return obj.content;
        }

        if (Array.isArray(obj.content)) {
            return obj.content.map(getCloudflarePromptText).join("\n");
        }

        if (typeof obj.prompt === "string") {
            return obj.prompt;
        }
    }

    return "";
}

function formatCloudflareMessages(prompt: unknown): string {
    if (typeof prompt === "string") {
        return prompt;
    }

    if (Array.isArray(prompt)) {
        return prompt
            .map((message) => {
                if (!message || typeof message !== "object") {
                    return "";
                }

                const role =
                    typeof (message as Record<string, unknown>).role === "string"
                        ? (message as Record<string, unknown>).role
                        : "user";
                const content = getCloudflarePromptText(
                    (message as Record<string, unknown>).content ?? message,
                );

                return content ? `${role}:\n${content}` : "";
            })
            .filter(Boolean)
            .join("\n\n");
    }

    return getCloudflarePromptText(prompt);
}

function getCloudflareModelConfig(modelName: string, apiKey?: string) {
    const cloudflareApiKey = getCloudflareToken(apiKey);
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    const defaultModel =
        process.env.CLOUDFLARE_MODEL ||
        "@cf/zai-org/glm-4.7-flash";

    if (!cloudflareApiKey) {
        throw new Error(
            "Cloudflare model selected but CLOUDFLARE_AUTH_TOKEN (or CLOUDFLARE_API_TOKEN/API_KEY) is missing.",
        );
    }

    if (!accountId) {
        throw new Error(
            "Cloudflare model selected but CLOUDFLARE_ACCOUNT_ID is missing.",
        );
    }

    const resolvedModel = ((): string => {
        if (
            modelName === "cloudflare" ||
            modelName === "glm" ||
            modelName === "@cf/zai-org/glm-4.7-flash"
        ) {
            return defaultModel;
        }
        return modelName.replace(/^cloudflare[:/]/, "").replace(/^@cf\//, "@cf/");
    })();

    return {
        specificationVersion: "v2",
        provider: "cloudflare",
        modelId: resolvedModel,
        supportedUrls: {},

        async doGenerate({ prompt }: { prompt: unknown }) {
            const fullPrompt = formatCloudflareMessages(prompt);

            const response = await fetch(
                `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${resolvedModel}`,
                {
                    method: "POST",
                    headers: {
                        Authorization: `Bearer ${cloudflareApiKey}`,
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ prompt: fullPrompt }),
                },
            );

            const data = await response.json();

            if (!response.ok || !data?.success) {
                const message =
                    data?.errors?.[0]?.message ||
                    data?.error?.message ||
                    `Cloudflare request failed with status ${response.status}`;
                throw new Error(message);
            }

            const raw =
                data?.result?.response ??
                data?.result?.text ??
                data?.result ??
                "";
            const text = typeof raw === "string" ? raw : JSON.stringify(raw);

            return {
                content: [{ type: "text", text }],
                finishReason: "stop",
                usage: {
                    inputTokens: undefined,
                    outputTokens: undefined,
                    totalTokens: undefined,
                },
                warnings: [],
                request: { body: fullPrompt },
                response: { body: data },
            };
        },

        async doStream() {
            throw new Error(
                "Streaming is not supported for Cloudflare direct model in this project.",
            );
        },
    } as any as ModelInstance;
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
    };

    if (
        modelName === "cloudflare" ||
        modelName.startsWith("cloudflare:") ||
        modelName.startsWith("cloudflare/") ||
        modelName.startsWith("@cf/")
    ) {
        return getCloudflareModelConfig(modelName, apiKey);
    }

    const factory = configs[modelName];
    if (!factory) {
        throw new Error(
            `Unknown model: ${modelName}. Supported: ${Object.keys(configs).join(", ")}, cloudflare, cloudflare:<model>, @cf/<model>`,
        );
    }
    return factory();
}
