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

function getCloudflareModelConfig(modelName: string, apiKey?: string) {
    const cloudflareApiKey = getCloudflareToken(apiKey);
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    const gateway = process.env.CLOUDFLARE_GATEWAY;
    const defaultModel =
        process.env.CLOUDFLARE_MODEL ||
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

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

    if (!gateway) {
        throw new Error(
            "Cloudflare model selected but CLOUDFLARE_GATEWAY is missing. Example: my-gateway",
        );
    }

    const resolvedModel =
        modelName === "cloudflare"
            ? defaultModel
            : modelName.replace(/^cloudflare[:/]/, "");

    return createOpenAI({
        apiKey: cloudflareApiKey,
        baseURL: `https://gateway.ai.cloudflare.com/v1/${accountId}/${gateway}`,
    })(resolvedModel);
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
        modelName.startsWith("cloudflare/")
    ) {
        return getCloudflareModelConfig(modelName, apiKey);
    }

    const factory = configs[modelName];
    if (!factory) {
        throw new Error(
            `Unknown model: ${modelName}. Supported: ${Object.keys(configs).join(", ")}, cloudflare, cloudflare:<model>`,
        );
    }
    return factory();
}
