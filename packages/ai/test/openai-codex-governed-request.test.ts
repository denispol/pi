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
		queueMicrotask(() => {
			this.dispatch("message", {
				data: JSON.stringify({
					type: "response.completed",
					response: { status: "completed", end_turn: true },
				}),
			});
		});
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
		expect(seen).toEqual([
			["object", { transport: "sse", model: "gpt-5.1-codex", accountId: "acc_test", priorSends: 0 }],
		]);
	});

	it("WS: allowance carries the full pre-delta body in the envelope", async () => {
		MockWebSocket.sent = [];
		vi.stubGlobal("WebSocket", MockWebSocket);
		// Make send() complete a minimal response so the stream finishes.
		const seen: Array<{ transport: unknown; fullLen: number; sentLen: number }> = [];
		(MockWebSocket as unknown as { sent: unknown[] }).sent = [];
		await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken(),
				transport: "websocket",
				fetch: vi.fn(async () => new Response("unexpected", { status: 500 })),
				governRequest: (body, envelope) => {
					seen.push({
						transport: (envelope as { transport: string }).transport,
						fullLen: ((envelope as { fullBody?: { input?: unknown[] } }).fullBody?.input ?? []).length,
						sentLen: ((body as { input?: unknown[] }).input ?? []).length,
					});
				},
			}),
		).catch(() => []);
		expect(seen.length).toBeGreaterThanOrEqual(1);
		expect(seen[0].transport).toBe("websocket");
		expect(seen[0].fullLen).toBeGreaterThanOrEqual(seen[0].sentLen);
	});

	// C2 dispatch contract (Package N): after a possible inference-bearing
	// send there is no automatic second send. Repeated validation without a
	// send stays harmless; recovery resends need a fresh dispatch permission
	// (a new caller preparation, i.e. a new generation) — never silent reuse.
	// Loss-path socket: records the send, then either closes quietly (server
	// never answered) or answers with previous_response_not_found (the send
	// demonstrably executed server-side). Standalone (no private access).
	class LossyWebSocket {
		static mode: "close-quiet" | "previous-not-found" = "close-quiet";
		private listeners = new Map<string, Set<(event: unknown) => void>>();
		constructor() {
			queueMicrotask(() => this.emit("open", {}));
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
			queueMicrotask(() => {
				if (LossyWebSocket.mode === "previous-not-found") {
					this.emit("message", {
						data: JSON.stringify({ type: "error", code: "previous_response_not_found" }),
					});
				} else {
					this.emit("close", { code: 1006 });
				}
			});
		}
		close(): void {}
		private emit(type: string, event: unknown): void {
			for (const listener of this.listeners.get(type) ?? []) listener(event);
		}
	}

	describe("single-send cardinality", () => {
		it("WS: post-send loss with no response stops uncertain (no SSE second send)", async () => {
			MockWebSocket.sent = [];
			LossyWebSocket.mode = "close-quiet";
			vi.stubGlobal("WebSocket", LossyWebSocket);
			const fetchMock = vi.fn(async () => new Response("second-send-must-not-happen", { status: 500 }));
			const facts: unknown[] = [];
			const events = await drain(
				streamOpenAICodexResponses(MODEL, testContext(), {
					apiKey: mockToken(),
					transport: "websocket",
					fetch: fetchMock,
					governRequest: () => {},
					onDispatch: (fact) => facts.push(fact),
				}),
			);
			expect(MockWebSocket.sent.length).toBe(1);
			expect(fetchMock).not.toHaveBeenCalled();
			expect(facts.length).toBe(1);
			expect((facts[0] as { transport?: string }).transport).toBe("websocket");
			const errors = events.filter((e) => (e as { type?: string }).type === "error");
			expect(errors.length).toBe(1);
		});

		it("WS: previous_response_not_found proves the send and never auto-resends", async () => {
			MockWebSocket.sent = [];
			LossyWebSocket.mode = "previous-not-found";
			vi.stubGlobal("WebSocket", LossyWebSocket);
			const fetchMock = vi.fn(async () => new Response("second-send-must-not-happen", { status: 500 }));
			const events = await drain(
				streamOpenAICodexResponses(MODEL, testContext(), {
					apiKey: mockToken(),
					transport: "websocket",
					fetch: fetchMock,
					governRequest: () => {},
				}),
			);
			expect(MockWebSocket.sent.length).toBe(1);
			expect(fetchMock).not.toHaveBeenCalled();
			const errors = events.filter((e) => (e as { type?: string }).type === "error");
			expect(errors.length).toBe(1);
		});

		it("SSE: ambiguous 500 stops after one send (no automatic retry)", async () => {
			const fetchMock = vi.fn(async () => new Response("boom", { status: 500 }));
			const facts: unknown[] = [];
			const events = await drain(
				streamOpenAICodexResponses(MODEL, testContext(), {
					apiKey: mockToken(),
					transport: "sse",
					maxRetries: 3,
					fetch: fetchMock,
					governRequest: () => {},
					onDispatch: (fact) => facts.push(fact),
				}),
			);
			expect(fetchMock).toHaveBeenCalledTimes(1);
			expect(facts.length).toBe(1);
			const errors = events.filter((e) => (e as { type?: string }).type === "error");
			expect(errors.length).toBe(1);
		});

		it("SSE: definitive 429 still retries, with one dispatch fact per send", async () => {
			const fetchMock = vi
				.fn()
				.mockResolvedValueOnce(new Response("slow", { status: 429 }))
				.mockResolvedValueOnce(
					new Response('data: {"type":"response.completed","response":{"status":"completed"}}\n\n', {
						status: 200,
						headers: { "content-type": "text/event-stream" },
					}),
				);
			const facts: Array<{ attemptSeq?: number; transport?: string }> = [];
			const events = await drain(
				streamOpenAICodexResponses(MODEL, testContext(), {
					apiKey: mockToken(),
					transport: "sse",
					maxRetries: 3,
					fetch: fetchMock,
					governRequest: () => {},
					onDispatch: (fact) => facts.push(fact),
				}),
			);
			expect(fetchMock).toHaveBeenCalledTimes(2);
			expect(facts.map((f) => f.attemptSeq)).toEqual([1, 2]);
			const done = events.filter((e) => (e as { type?: string }).type === "done");
			expect(done.length).toBe(1);
		});

		it("WS: pre-send abort sends nothing", async () => {
			MockWebSocket.sent = [];
			vi.stubGlobal("WebSocket", MockWebSocket);
			const controller = new AbortController();
			controller.abort();
			const governed: unknown[] = [];
			const events = await drain(
				streamOpenAICodexResponses(MODEL, testContext(), {
					apiKey: mockToken(),
					transport: "websocket",
					fetch: vi.fn(async () => new Response("unexpected", { status: 500 })),
					signal: controller.signal,
					governRequest: () => {
						governed.push(true);
					},
				}),
			);
			expect(MockWebSocket.sent).toEqual([]);
			const errors = events.filter((e) => (e as { type?: string }).type === "error");
			expect(errors.length).toBe(1);
		});

		it("governor envelope binds trusted route identity with zero prior sends", async () => {
			MockWebSocket.sent = [];
			vi.stubGlobal("WebSocket", MockWebSocket);
			const seen: unknown[] = [];
			await drain(
				streamOpenAICodexResponses(MODEL, testContext(), {
					apiKey: mockToken(),
					transport: "websocket",
					fetch: vi.fn(async () => new Response("unexpected", { status: 500 })),
					governRequest: (_body, envelope) => {
						seen.push(envelope);
					},
				}),
			);
			expect(seen.length).toBeGreaterThanOrEqual(1);
			const first = seen[0] as Record<string, unknown>;
			expect(first["transport"]).toBe("websocket");
			expect(first["model"]).toBe("gpt-5.1-codex");
			expect(first["accountId"]).toBe("acc_test");
			expect(first["priorSends"]).toBe(0);
		});

		it("dispatch facts are deterministic per identical bytes", async () => {
			const runOnce = async () => {
				MockWebSocket.sent = [];
				vi.stubGlobal("WebSocket", MockWebSocket);
				const facts: Array<{ payloadHash?: string; byteLength?: number }> = [];
				await drain(
					streamOpenAICodexResponses(MODEL, testContext(), {
						apiKey: mockToken(),
						transport: "websocket",
						fetch: vi.fn(async () => new Response("unexpected", { status: 500 })),
						governRequest: () => {},
						onDispatch: (fact) => facts.push(fact),
					}),
				);
				return facts;
			};
			const first = await runOnce();
			const second = await runOnce();
			expect(first.length).toBe(1);
			expect(second.length).toBe(1);
			expect(first[0].payloadHash).toMatch(/^[0-9a-f]{16}$/);
			expect(first[0].payloadHash).toBe(second[0].payloadHash);
			expect(first[0].byteLength).toBeGreaterThan(0);
		});
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
