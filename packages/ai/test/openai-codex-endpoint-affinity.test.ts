import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	closeOpenAICodexWebSocketSessions,
	resetOpenAICodexWebSocketDebugStats,
	streamSimple,
} from "../src/api/openai-codex-responses.ts";
import type { Message } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";
import {
	codexModel,
	createCodexWebSocketHarness,
	syntheticContext,
	syntheticToken,
} from "./fixtures/codex-websocket.ts";

let harness: ReturnType<typeof createCodexWebSocketHarness>;
let messages: Message[];

beforeEach(() => {
	messages = [];
	harness = createCodexWebSocketHarness();
	vi.stubGlobal("WebSocket", harness.WebSocket);
	vi.stubGlobal(
		"fetch",
		vi.fn(() => Promise.reject(new Error("Unexpected network request"))),
	);
});

afterEach(() => {
	closeOpenAICodexWebSocketSessions();
	resetOpenAICodexWebSocketDebugStats();
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

async function request(
	turn: number,
	endpoint = codexModel.baseUrl,
	account = "account-a",
	sessionId = "affinity",
	signature = "synthetic",
) {
	const context = {
		systemPrompt: "Synthetic conversation only.",
		messages: [...messages, { role: "user" as const, content: `Synthetic turn ${turn}`, timestamp: turn }],
	};
	const response = await streamSimple({ ...codexModel, baseUrl: endpoint }, normalizeContext(context), {
		apiKey: syntheticToken(account, signature),
		sessionId,
		transport: "auto",
		timeoutMs: 1000,
	}).result();
	expect(response.stopReason).toBe("stop");
	expect(response.responseId).toBeDefined();
	messages = [...context.messages, response];
	return response;
}

describe("Codex cached socket owner", () => {
	// Endpoint isolation from #6539 must preserve the current per-account pool.
	it("isolates endpoints and retains each endpoint's continuation when returning", async () => {
		const first = await request(1);
		await request(2, "https://second.synthetic.invalid/backend-api");
		await request(3);

		expect(harness.connections).toHaveLength(2);
		expect(harness.connections.map((connection) => connection.url)).toEqual([
			"wss://first.synthetic.invalid/backend-api/codex/responses",
			"wss://second.synthetic.invalid/backend-api/codex/responses",
		]);
		expect(harness.connections[1].bodies[0].previous_response_id).toBeUndefined();
		expect(harness.connections[0].bodies[1].previous_response_id).toBe(first.responseId);
	});

	it("retains A's socket and continuation across A to B to A", async () => {
		const first = await request(1);
		await request(2, codexModel.baseUrl, "account-b");
		await request(3);

		expect(harness.connections).toHaveLength(2);
		expect(harness.connections.map((connection) => connection.account)).toEqual(["account-a", "account-b"]);
		expect(harness.connections[1].bodies[0].previous_response_id).toBeUndefined();
		expect(harness.connections[0].bodies[1].previous_response_id).toBe(first.responseId);
	});

	it("keeps same-account token refresh and normalized endpoint reuse", async () => {
		const first = await request(1);
		const response = await request(
			2,
			"https://FIRST.synthetic.invalid:443/backend-api/",
			"account-a",
			"affinity",
			"refreshed",
		);

		expect(response.stopReason).toBe("stop");
		expect(harness.connections).toHaveLength(1);
		expect(harness.connections[0].bodies[1].previous_response_id).toBe(first.responseId);
		expect(harness.connections[0].bodies[1].prompt_cache_key).toBe("affinity");
	});

	it("does not share sockets or continuation between Pi sessions", async () => {
		await request(1);
		await request(2, codexModel.baseUrl, "account-a", "another-session");
		expect(harness.connections).toHaveLength(2);
		expect(harness.connections[1].bodies[0].previous_response_id).toBeUndefined();
	});

	it("does not let an earlier lease's cancellation evict its same-owner replacement", async () => {
		harness.enqueue("hold", "hold");
		const controller = new AbortController();
		const context = syntheticContext();
		const options = { apiKey: syntheticToken(), sessionId: "replacement", transport: "auto" as const };
		const earlier = streamSimple(codexModel, context, { ...options, signal: controller.signal }).result();
		const replacement = streamSimple(codexModel, context, options).result();
		await vi.waitFor(() => {
			expect(harness.connections).toHaveLength(2);
			expect(harness.connections[0].bodies).toHaveLength(1);
			expect(harness.connections[1].bodies).toHaveLength(1);
		});
		controller.abort();
		expect((await earlier).stopReason).toBe("aborted");
		harness.connections[1].emit({
			type: "response.completed",
			response: {
				id: "replacement-response",
				status: "completed",
				usage: { input_tokens: 5, output_tokens: 0, total_tokens: 5 },
			},
		});
		const completed = await replacement;
		expect(completed.stopReason).toBe("stop");
		const next = await streamSimple(
			codexModel,
			normalizeContext({
				messages: [...context.messages, completed, { role: "user", content: "Continue", timestamp: 2 }],
			}),
			options,
		).result();
		expect(next.stopReason).toBe("stop");
		expect(harness.connections).toHaveLength(2);
		expect(harness.connections[1].bodies[1].previous_response_id).toBe("replacement-response");
	});

	it("does not let cancellation on an old endpoint evict the new endpoint's entry", async () => {
		harness.enqueue("hold");
		const controller = new AbortController();
		const old = streamSimple(codexModel, syntheticContext(), {
			apiKey: syntheticToken(),
			sessionId: "affinity",
			transport: "auto",
			signal: controller.signal,
		}).result();
		await vi.waitFor(() => expect(harness.connections[0]?.bodies).toHaveLength(1));
		const second = await request(2, "https://second.synthetic.invalid/backend-api");
		controller.abort();
		expect((await old).stopReason).toBe("aborted");
		await request(3, "https://second.synthetic.invalid/backend-api");

		expect(harness.connections).toHaveLength(2);
		expect(harness.connections[1].bodies[1].previous_response_id).toBe(second.responseId);
	});

	it("expires one endpoint without removing a different endpoint's live entry", async () => {
		vi.useFakeTimers();
		await request(1);
		await vi.advanceTimersByTimeAsync(2 * 60_000);
		const second = await request(2, "https://second.synthetic.invalid/backend-api");
		await vi.advanceTimersByTimeAsync(3 * 60_000 + 1);
		expect(harness.connections[0].closes).toContainEqual({ code: 1000, reason: "idle_timeout" });
		await request(3, "https://second.synthetic.invalid/backend-api");
		expect(harness.connections).toHaveLength(2);
		expect(harness.connections[1].bodies[1].previous_response_id).toBe(second.responseId);
	});
});
