// Governed-request final-send enforcement (fork: feat/governed-request-denial).
// Denial must be terminal: zero transport sends, no retry, no SSE fallback.
// Run from packages/ai: npx vitest --run test/openai-codex-governed-request.test.ts
//
// Residual map (C2 acceptance):
// - Single-use dispatch: cardinality suite below (loss, previous-not-found,
//   429-as-send, ambiguous-500, pre-send abort). All are local receiver/fake
//   tests through the real stream() path; no live inference.
// - R-DELTA-CUT: exact-cut continuation test below drives two real streams
//   sharing one session (response.created -> continuation -> delta send).
// - R-RESPID: previous_response_not_found is terminal without resend; the
//   delta test links the cut to the actual preceding response id.
// - R-FIRST-SEND: envelope identity (model/accountId/priorSends) is emitted
//   and tested, but comparison against the selected authority stays
//   caller-side via the hook payload; accountId has no caller-side expected
//   value (capability record owns it in Package C).
// - Terminal-outcome correlation (completed vs failed/incomplete/late) is
//   the N/B integration slice, not this file.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	getOpenAICodexWebSocketDebugStats,
	hashDispatchBytes,
	ProviderRequestDeniedError,
	resetOpenAICodexWebSocketDebugStats,
	stream as streamOpenAICodexResponses,
} from "../src/api/openai-codex-responses.ts";
import type { DispatchFact } from "../src/api/openai-codex-responses.ts";
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
			[
				"object",
				{
					transport: "sse",
					model: "gpt-5.1-codex",
					accountId: "acc_test",
					priorSends: 0,
					authNamespace: "acct:acc_test",
					authRevision: 1,
				},
			],
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

		// A received 429 is a received generation request: the send counts
		// (the contract counts attempts, not accepted inference), so no
		// automatic second send follows. Recovery is a new generation with
		// fresh permission.
		it("SSE: received 429 stops after one send (no automatic retry)", async () => {
			const fetchMock = vi.fn(async () => new Response("slow", { status: 429 }));
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
			expect(fetchMock).toHaveBeenCalledTimes(1);
			expect(facts.map((f) => f.attemptSeq)).toEqual([1]);
			const errors = events.filter((e) => (e as { type?: string }).type === "error");
			expect(errors.length).toBe(1);
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

		// R-DELTA-CUT: the native cut is the exact suffix after the recorded
		// baseline (previous request + response items), linked to the actual
		// preceding response id on the same connection state.
		it("WS: continuation sends the exact delta linked to the preceding response", async () => {
			class ScriptedSocket {
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
						this.emit("message", {
							data: JSON.stringify({ type: "response.created", response: { id: "resp_1" } }),
						});
						this.emit("message", {
							data: JSON.stringify({
								type: "response.completed",
								response: { id: "resp_1", status: "completed", end_turn: true },
							}),
						});
					});
				}
				close(): void {}
				private emit(type: string, event: unknown): void {
					for (const listener of this.listeners.get(type) ?? []) listener(event);
				}
			}
			MockWebSocket.sent = [];
			vi.stubGlobal("WebSocket", ScriptedSocket);
			const seen: Array<{ sentLen: number; fullLen: number; prevId: unknown }> = [];
			const opts = {
				apiKey: mockToken(),
				transport: "auto" as const,
				sessionId: "delta-cut-test",
				fetch: vi.fn(async () => new Response("unexpected", { status: 500 })),
				governRequest: (body: unknown, envelope: unknown) => {
					const b = body as { input?: unknown[] };
					const e = envelope as { fullBody?: { input?: unknown[] } };
					seen.push({
						sentLen: b.input?.length ?? -1,
						fullLen: e.fullBody?.input?.length ?? -1,
						prevId: (b as { previous_response_id?: unknown }).previous_response_id,
					});
				},
			};
			const ctx1 = normalizeContext({
				systemPrompt: "You are a helpful assistant.",
				messages: [{ role: "user", content: "Say hello", timestamp: Date.now() }],
			});
			await drain(streamOpenAICodexResponses(MODEL, ctx1, opts));
			const ctx2 = normalizeContext({
				systemPrompt: "You are a helpful assistant.",
				messages: [
					{ role: "user", content: "Say hello", timestamp: Date.now() },
					{ role: "user", content: "And more", timestamp: Date.now() },
				],
			});
			await drain(streamOpenAICodexResponses(MODEL, ctx2, opts));
			expect(seen.length).toBe(2);
			expect(seen[0].prevId).toBeUndefined();
			// Second send is the exact delta: one new item, full body held in
			// the envelope, linked to the actual preceding response.
			expect(seen[1].prevId).toBe("resp_1");
			expect(seen[1].sentLen).toBe(1);
			expect(seen[1].fullLen).toBe(seen[1].sentLen + seen[0].sentLen);
		});

		it("a throwing dispatch listener never fails the send it observes", async () => {
			const fetchMock = vi.fn(
				async () =>
					new Response('data: {"type":"response.completed","response":{"status":"completed"}}\n\n', {
						status: 200,
						headers: { "content-type": "text/event-stream" },
					}),
			);
			const events = await drain(
				streamOpenAICodexResponses(MODEL, testContext(), {
					apiKey: mockToken(),
					transport: "sse",
					fetch: fetchMock,
					governRequest: () => {},
					onDispatch: () => {
						throw new Error("listener defect");
					},
				}),
			);
			expect(fetchMock).toHaveBeenCalledTimes(1);
			const done = events.filter((e) => (e as { type?: string }).type === "done");
			expect(done.length).toBe(1);
		});

		it("dispatch fact echoes the governor-returned identity and governed-bytes hash", async () => {
			let governed: unknown;
			const facts: DispatchFact[] = [];
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
					fetch: fetchMock,
					governRequest: (body) => {
						governed = body;
						return "caller-attempt-7";
					},
					onDispatch: (fact) => {
						facts.push(fact);
					},
				}),
			);
			expect(facts).toHaveLength(1);
			expect(facts[0]?.identity).toBe("caller-attempt-7");
			// Same reference the caller holds: byte-identical serialization.
			expect(facts[0]?.governHash).toBe(hashDispatchBytes(JSON.stringify(governed)));
		});

		it("void governor return leaves identity undefined but still hashes (pre-identity compat)", async () => {
			const facts: DispatchFact[] = [];
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
					fetch: fetchMock,
					governRequest: () => {},
					onDispatch: (fact) => {
						facts.push(fact);
					},
				}),
			);
			expect(facts).toHaveLength(1);
			expect(facts[0]?.identity).toBeUndefined();
			expect(typeof facts[0]?.governHash).toBe("string");
			expect((facts[0]?.governHash ?? "").length).toBeGreaterThan(0);
		});

		it("session readback retains the performed send until acknowledged", async () => {
			const sessionId = "b1-readback-session";
			resetOpenAICodexWebSocketDebugStats(sessionId);
			const facts: DispatchFact[] = [];
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
					sessionId,
					fetch: fetchMock,
					governRequest: () => "caller-attempt-3",
					onDispatch: (fact) => {
						facts.push(fact);
					},
				}),
			);
			const stats = getOpenAICodexWebSocketDebugStats(sessionId);
			expect(stats?.dispatchedRequests).toBe(1);
			expect(stats?.lastDispatchFact).toEqual(facts[0]);
		});

		it("cyrb53 dispatch-hash vector (evidence consumers port this)", () => {
			expect(hashDispatchBytes("thinking-adjuster-b1")).toBe("7f57b39bd8a6636d");
		});

		it("dispatch-hash v1 fixtures (UTF-8 lanes, empty/Unicode/JSON)", () => {
			expect(hashDispatchBytes("")).toBe("488bdcb81aee8d83");
			expect(hashDispatchBytes("héllo→世界")).toBe("50ec19137e5f54ef");
			expect(hashDispatchBytes('{"a":1}')).toBe("de3034b77cc1d846");
		});

		it("dispatch facts are deterministic per identical bytes and carry v1", async () => {
			const runOnce = async () => {
				MockWebSocket.sent = [];
				vi.stubGlobal("WebSocket", MockWebSocket);
				const facts: Array<{ v?: number; payloadHash?: string; byteLength?: number }> = [];
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
			expect(first[0].v).toBe(1);
			expect(second[0].v).toBe(1);
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

describe("C-ID account authority (shared/planning#565)", () => {
	beforeEach(async () => {
		const { resetSelection } = await import("../src/auth/authority.ts");
		resetSelection("openai-codex");
	});

	async function selectAccount(accountId: string) {
		const { noteSelectedNamespace, deriveNamespace } = await import("../src/auth/authority.ts");
		return noteSelectedNamespace("openai-codex", deriveNamespace(mockToken(accountId)));
	}

	it("F-CID-1: send under a non-selected account fails closed with zero sends", async () => {
		await selectAccount("acc_selected");
		MockWebSocket.sent = [];
		vi.stubGlobal("WebSocket", MockWebSocket);
		const fetchMock = vi.fn(async () => new Response("must-not-send", { status: 500 }));
		const events = await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken("acc_other"),
				transport: "websocket",
				fetch: fetchMock,
				governRequest: () => {},
			}),
		);
		const errors = events.filter((e) => (e as { type?: string }).type === "error");
		expect(errors.length).toBe(1);
		expect(String((errors[0] as { error?: { errorMessage?: string } }).error?.errorMessage)).toMatch(
			/account-mismatch/,
		);
		expect(MockWebSocket.sent.length).toBe(0);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("F-CID-2: stale selection after an account switch is denied", async () => {
		await selectAccount("acc_A");
		await selectAccount("acc_B"); // operator switches account
		const fetchMock = vi.fn(async () => new Response("must-not-send", { status: 500 }));
		const events = await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken("acc_A"),
				transport: "sse",
				fetch: fetchMock,
				governRequest: () => {},
			}),
		);
		const errors = events.filter((e) => (e as { type?: string }).type === "error");
		expect(errors.length).toBe(1);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("F-CID-5: same-account rotation keeps working (no false denial)", async () => {
		await selectAccount("acc_A");
		await selectAccount("acc_A"); // refresh: same namespace, no revision bump
		const facts: unknown[] = [];
		MockWebSocket.sent = [];
		vi.stubGlobal("WebSocket", MockWebSocket);
		const events = await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken("acc_A"),
				transport: "websocket",
				fetch: vi.fn(async () => new Response("unexpected", { status: 500 })),
				governRequest: () => {},
				onDispatch: (fact) => facts.push(fact),
			}),
		);
		const errors = events.filter((e) => (e as { type?: string }).type === "error");
		expect(errors.length).toBe(0);
		expect(MockWebSocket.sent.length).toBe(1);
		expect(facts.length).toBe(1);
	});

	it("F-CID-4: effort outside provider-declared levels is denied pre-send", async () => {
		await selectAccount("acc_A");
		const fetchMock = vi.fn(async () => new Response("must-not-send", { status: 500 }));
		const events = await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken("acc_A"),
				transport: "sse",
				fetch: fetchMock,
				governRequest: () => {},
				reasoningEffort: "ultra",
			}),
		);
		const errors = events.filter((e) => (e as { type?: string }).type === "error");
		expect(errors.length).toBe(1);
		expect(String((errors[0] as { error?: { errorMessage?: string } }).error?.errorMessage)).toMatch(
			/first-request-settings/,
		);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("dispatch facts carry the account namespace and selection revision", async () => {
		const binding = await selectAccount("acc_A");
		const facts: Array<{ authNamespace?: string; authRevision?: number }> = [];
		MockWebSocket.sent = [];
		vi.stubGlobal("WebSocket", MockWebSocket);
		await drain(
			streamOpenAICodexResponses(MODEL, testContext(), {
				apiKey: mockToken("acc_A"),
				transport: "websocket",
				fetch: vi.fn(async () => new Response("unexpected", { status: 500 })),
				governRequest: () => {},
				onDispatch: (fact) => facts.push(fact),
			}),
		);
		expect(facts.length).toBe(1);
		expect(facts[0].authNamespace).toBe("acct:acc_A");
		expect(facts[0].authRevision).toBe(binding?.selectionRevision);
	});
});

describe("C-ID refresh vs replacement (shared/planning#565)", () => {
	it("same-account rotation passes through; account change at refresh throws", async () => {
		const { assertSameAccountRefresh } = await import("../src/auth/oauth/openai-codex.ts");
		const prev = { type: "oauth", access: "a", refresh: "r", expires: 1, accountId: "acc_A" } as const;
		const same = { type: "oauth", access: "a2", refresh: "r2", expires: 2, accountId: "acc_A" } as const;
		expect(assertSameAccountRefresh(prev, same)).toBe(same);
		const other = { type: "oauth", access: "a3", refresh: "r3", expires: 3, accountId: "acc_B" } as const;
		expect(() => assertSameAccountRefresh(prev, other)).toThrow(/replaced during refresh/);
	});
});
