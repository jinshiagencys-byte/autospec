import { z } from "zod";

export const magicStrings = {
    specPassed: "The spec passed",
    specFailed: "The spec failed",
};

export const testPlanSchema = z.object({
    arrayOfSpecs: z.array(
        z.union([
            z.string(),
            z.object({
                id: z.string().optional(),
                title: z.string(),
                description: z.string().optional(),
                priority: z.string().optional(),
                tags: z.array(z.string()).optional(),
            }),
        ]),
    ),
});

export const modelNameSchema = z.union([
    z.enum([
        "gpt-5.4",
        "claude-opus-4-6",
        "gemini-2.5-flash",
        "cloudflare",
        "opencode",
        "ollama",
    ]),
    z.string().startsWith("cloudflare:"),
    z.string().startsWith("cloudflare/"),
    z.string().startsWith("@cf/"),
    z.string().startsWith("opencode:"),
    z.string().startsWith("opencode/"),
    z.string().startsWith("ollama:"),
]);
export type ModelName = z.infer<typeof modelNameSchema>;

export type TestResult = {
    spec: string;
    status: "passed" | "failed";
    actions: ActionRecord[];
    totalInputTokens: number;
    totalOutputTokens: number;
    reason?: string;
};

export type ActionRecord = {
    tool: string;
    args: Record<string, unknown>;
    result: unknown;
};
