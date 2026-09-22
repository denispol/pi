// Governed-request final-send enforcement (fork: feat/governed-request-denial).
// Denial must be terminal: zero transport sends, no retry, no SSE fallback.
// Run from packages/ai: npx vitest --run test/openai-codex-governed-request.test.ts

import { afterEach, describe, expect, it, vi } from "vitest";
import { ProviderRequestDeniedError, stream as streamOpenAICodexResponses } from "../src/api/openai-codex-responses.ts";
import type { Model } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

afterEach(() => {
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
});

function mockToken(accountId = "acc_test"): string {
	const payload = Buffer.from(
		JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: accountId } }),
		"utf8",
	).toString("base64");
	return `aaa.${payload}.bbb`;
}

const MODEL: Model<"openai-codex-responses"> = {
	id: "gpt-5.1-codex",
	name: "GPT-5.1 Codex",
	api: "openai-codex-responses",
	provider: "openai-codex",
	baseUrl: "https://chatgpt.com/backend-api",
	reasoning: true,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 400000,
	maxTokens: 128000,
};

function testContext() {
	return normalizeContext({
		systemPrompt: "You are a helpful assistant.",
		messages: [{ role: "user", content: "Say hello", timestamp: Date.now() }],
	});
}

async function drain(stream: AsyncIterable<unknown>): Promise<unknown[]> {
	const events: unknown[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

class MockWebSocket {
	static sent: unknown[] = [];
	private listeners = new Map<string, Set<(event: unknown) => void>>();

	constructor() {
		queueMicrotask(() => this.dispatch("open", {}));
	}
	addEventListener(type: string, listener: (event: unknown) => void): void {
		let set = this.listeners.get(type);
		if (!set) {
			set = new Set();
			this.listeners.set(type, set);
		}
		set.add(listener);
	}
	removeEventListener(type: string, listener: (event: unknown) => void): void {
		this.listeners.get(type)?.delete(listener);
	}
	send(data: string): void {
		MockWebSocket.sent.push(JSON.parse(data));
	}
	close(): void {}
	private dispatch(type: string, event: unknown): void {
		for (const listener of this.listeners.get(type) ?? []) listener(event);
	}
}

describe("governed request denial", () => {
	it("carries a stable code and name (never match on message text)", () => {
		const err = new ProviderRequestDeniedError("ADMISSION_DENIED:E_BUDGET", "exhausted");
		expect(err).toBeInstanceOf(Error);
		expect(err.name).toBe("ProviderRequestDeniedError");
		expect(err.code).toBe("ADMISSION_DENIED:E_BUDGET");
	});

	it("SSE: denial sends zero requests and does not retry", async () => {
		const fetchMock = vi.fn(async () => new Response("must-not-send", { status: 500 }));
		const events = await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken(),
				transport: "sse",
				fetch: fetchMock,
				governRequest: () => {
					throw new ProviderRequestDeniedError("ADMISSION_DENIED:E_BUDGET", "exhausted");
				},
			}),
		);
		expect(fetchMock).not.toHaveBeenCalled();
		const errors = events.filter((e) => (e as { type?: string }).type === "error");
		expect(errors.length).toBe(1);
	});

	it("SSE: allowance invokes the governor with the sse envelope, then sends once", async () => {
		const seen: unknown[] = [];
		const fetchMock = vi.fn(
			async () =>
				new Response('data: {"type":"response.completed","response":{"status":"completed"}}\n\n', {
					status: 200,
					headers: { "content-type": "text/event-stream" },
				}),
		);
		await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken(),
				transport: "sse",
				maxRetries: 0,
				fetch: fetchMock,
				governRequest: (body, envelope) => {
					seen.push([typeof body, envelope]);
				},
			}),
		);
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(seen).toEqual([["object", { transport: "sse" }]]);
	});

	it("WS: denial runs zero socket.send and terminates without fallback send", async () => {
		MockWebSocket.sent = [];
		vi.stubGlobal("WebSocket", MockWebSocket);
		const fetchMock = vi.fn(async () => new Response("fallback-must-not-send", { status: 500 }));
		const events = await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken(),
				fetch: fetchMock,
				governRequest: () => {
					throw new ProviderRequestDeniedError("ADMISSION_DENIED:E_HISTORY", "replay-refused");
				},
			}),
		);
		expect(MockWebSocket.sent).toEqual([]);
		expect(fetchMock).not.toHaveBeenCalled();
		const errors = events.filter((e) => (e as { type?: string }).type === "error");
		expect(errors.length).toBe(1);
	});
});
