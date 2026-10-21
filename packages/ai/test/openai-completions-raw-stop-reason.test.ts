import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { Model } from "../src/types.ts";
import { isRetryableAssistantError } from "../src/utils/retry.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const mockState = vi.hoisted(() => ({
	chunks: [] as unknown[],
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: () => {
					const chunks = mockState.chunks;
					const stream = {
						async *[Symbol.asyncIterator]() {
							for (const chunk of chunks) yield chunk;
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}
	return { default: FakeOpenAI };
});

const model: Model<"openai-completions"> = {
	id: "test-model",
	name: "Test Model",
	api: "openai-completions",
	provider: "openai",
	baseUrl: "https://api.openai.com/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 4096,
};

const context = normalizeContext({
	messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
});

describe("OpenAI completions raw stop reasons", () => {
	beforeEach(() => {
		mockState.chunks = [];
	});

	it("preserves raw finish reasons for successful stops", async () => {
		mockState.chunks = [{ id: "chatcmpl-1", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }];

		const message = await streamOpenAICompletions(model, context, { apiKey: "test" }).result();

		expect(message.stopReason).toBe("stop");
		expect(message.rawStopReason).toBe("stop");
		expect(message.errorMessage).toBeUndefined();
	});

	it("preserves raw finish reasons for provider error stops", async () => {
		mockState.chunks = [{ id: "chatcmpl-2", choices: [{ index: 0, delta: {}, finish_reason: "content_filter" }] }];

		const message = await streamOpenAICompletions(model, context, { apiKey: "test" }).result();

		expect(message.stopReason).toBe("error");
		expect(message.rawStopReason).toBe("content_filter");
		expect(message.errorMessage).toBe("Provider finish_reason: content_filter");
		expect(isRetryableAssistantError(message)).toBe(false);
	});

	// #9002
	it("retries error stops and keeps the chunk's error payload", async () => {
		// OpenRouter's mid-stream error chunk: the 200 response has already started.
		mockState.chunks = [
			{ id: "gen-3", choices: [{ index: 0, delta: { content: "partial" } }] },
			{
				id: "gen-3",
				error: { code: 429, message: "Rate limit exceeded", metadata: { error_type: "rate_limit_exceeded" } },
				choices: [{ index: 0, delta: { content: "" }, finish_reason: "error" }],
			},
		];

		const message = await streamOpenAICompletions(model, context, { apiKey: "test" }).result();

		expect(message.stopReason).toBe("error");
		expect(message.rawStopReason).toBe("error");
		expect(message.errorMessage).toBe(
			'Provider finish_reason: error (server error): {"code":429,"message":"Rate limit exceeded","metadata":{"error_type":"rate_limit_exceeded"}}',
		);
		expect(isRetryableAssistantError(message)).toBe(true);
	});

	// #9002
	it("retries error stops without an error payload", async () => {
		mockState.chunks = [{ id: "chatcmpl-4", choices: [{ index: 0, delta: {}, finish_reason: "error" }] }];

		const message = await streamOpenAICompletions(model, context, { apiKey: "test" }).result();

		expect(message.errorMessage).toBe("Provider finish_reason: error (server error)");
		expect(isRetryableAssistantError(message)).toBe(true);
	});

	// #10752
	it("keeps an error payload sent on the choice", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-5",
				choices: [
					{
						index: 0,
						delta: {},
						finish_reason: "error",
						error: { type: "stall_after_content", message: "Upstream idle timeout after content delivered" },
					},
				],
			},
		];

		const message = await streamOpenAICompletions(model, context, { apiKey: "test" }).result();

		expect(message.errorMessage).toBe(
			'Provider finish_reason: error (server error): {"type":"stall_after_content","message":"Upstream idle timeout after content delivered"}',
		);
	});

	it("does not retry error stops whose payload reports exhausted quota", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-6",
				error: { code: "insufficient_quota", message: "You exceeded your current quota" },
				choices: [{ index: 0, delta: {}, finish_reason: "error" }],
			},
		];

		const message = await streamOpenAICompletions(model, context, { apiKey: "test" }).result();

		expect(message.stopReason).toBe("error");
		expect(isRetryableAssistantError(message)).toBe(false);
	});
});
