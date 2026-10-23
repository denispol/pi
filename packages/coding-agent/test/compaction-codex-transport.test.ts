import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import {
	createAssistantMessageEventStream,
	fauxAssistantMessage,
	type RetryCallbacks,
	type RetryPolicy,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import {
	closeOpenAICodexWebSocketSessions,
	resetOpenAICodexWebSocketDebugStats,
	streamSimple,
} from "@earendil-works/pi-ai/api/openai-codex-responses";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	codexModel,
	createCodexWebSocketHarness,
	syntheticContext,
	syntheticSseResponse,
	syntheticToken,
} from "../../ai/test/fixtures/codex-websocket.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { completeSummarization } from "../src/core/compaction/compaction.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

let harness: ReturnType<typeof createCodexWebSocketHarness>;
let fetchMock: ReturnType<typeof vi.fn>;
let attemptedOptions: SimpleStreamOptions[];
const retry: RetryPolicy = { enabled: true, maxRetries: 1, baseDelayMs: 1, maxAgentDelayMs: 1 };
const cleanups: Array<() => void> = [];

beforeEach(() => {
	harness = createCodexWebSocketHarness();
	attemptedOptions = [];
	fetchMock = vi.fn(async () => syntheticSseResponse("Recovered summary"));
	vi.stubGlobal("WebSocket", harness.WebSocket);
	vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
	while (cleanups.length) cleanups.pop()?.();
	closeOpenAICodexWebSocketSessions();
	resetOpenAICodexWebSocketDebugStats();
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

const actualStream: StreamFn = (_model, context, options) => {
	attemptedOptions.push(options ?? {});
	return streamSimple(codexModel, context, { ...options, maxRetries: 0, timeoutMs: 50 });
};

function summarize(options: SimpleStreamOptions = {}, policy = retry, callbacks?: RetryCallbacks) {
	return completeSummarization(
		codexModel,
		syntheticContext(),
		{
			apiKey: syntheticToken(),
			transport: "auto",
			...options,
		},
		actualStream,
		policy,
		callbacks,
	);
}

async function createSdkSession(transport: "auto" | "sse", inputTokens = 100) {
	const directory = mkdtempSync(join(tmpdir(), "pi-summary-transport-"));
	const agentDir = join(directory, "agent");
	mkdirSync(agentDir);
	const model = { ...codexModel, provider: "synthetic-summary-provider" };
	const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
	await authStorage.modify(model.provider, async () => ({ type: "api_key", key: syntheticToken() }));
	const registry = await createModelRegistry(authStorage, join(agentDir, "models.json"));
	registry.registerProvider(model.provider, {
		api: model.api,
		streamSimple: (requestModel, context, options) =>
			streamSimple({ ...requestModel, api: "openai-codex-responses" }, context, options),
	});
	const sessionManager = SessionManager.inMemory(directory);
	const { session } = await createAgentSession({
		cwd: directory,
		agentDir,
		model,
		sessionManager,
		modelRuntime: getModelRuntime(registry),
		settingsManager: SettingsManager.inMemory({
			transport,
			compaction: { reserveTokens: 4096, keepRecentTokens: 1 },
			retry,
		}),
	});
	cleanups.push(() => {
		session.dispose();
		registry.unregisterProvider(model.provider);
		rmSync(directory, { recursive: true, force: true });
	});
	const previous = fauxAssistantMessage("Previous synthetic reply");
	sessionManager.appendMessage({ role: "user", content: "Earlier synthetic question", timestamp: 1 });
	sessionManager.appendMessage({
		...previous,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: { ...previous.usage, input: inputTokens, totalTokens: inputTokens },
	});
	session.agent.state.messages = sessionManager.buildSessionContext().messages;
	return { session, sessionManager };
}

describe("Codex summary transport recovery (#11202)", () => {
	// #10781: exercise AgentSession -> SDK -> the actual Codex adapter, not a replaced agent stream function.
	it.each(["sse", "auto"] as const)(
		"routes manual split-turn compaction through configured %s transport",
		async (transport) => {
			const { session, sessionManager } = await createSdkSession(transport);
			if (transport === "auto") harness.enqueue("partial-disconnect");
			const result = await session.compact();
			expect(result.summary).toContain("Recovered summary");
			expect(result.summary).not.toContain("Incomplete");
			expect(sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(1);
			expect(harness.connections).toHaveLength(transport === "sse" ? 0 : 1);
			expect(fetchMock).toHaveBeenCalledTimes(1);
		},
	);

	it("recovers automatic pre-prompt compaction and persists only its final summary", async () => {
		const { session, sessionManager } = await createSdkSession("auto", codexModel.contextWindow - 4096 + 100);
		harness.enqueue("partial-disconnect");
		await session.prompt("Continue the synthetic conversation");
		const entries = sessionManager.getEntries().filter((entry) => entry.type === "compaction");
		expect(entries).toHaveLength(1);
		expect(entries[0].summary).toContain("Recovered summary");
		expect(entries[0].summary).not.toContain("Incomplete");
		expect(harness.connections).toHaveLength(2); // Failed uncached summary, then ordinary cached turn.
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it("uses the SSE fixture successfully without opening a WebSocket", async () => {
		const result = await summarize({ transport: "sse" });
		expect(result.stopReason).toBe("stop");
		expect(result.content).toEqual([expect.objectContaining({ type: "text", text: "Recovered summary" })]);
		expect(harness.connections).toHaveLength(0);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it("pins the provider's post-start transport diagnostic without replaying inside the provider", async () => {
		harness.enqueue("disconnect");
		const result = await streamSimple(codexModel, syntheticContext(), {
			apiKey: syntheticToken(),
			transport: "auto",
			cacheRetention: "none",
		}).result();
		expect(result.stopReason).toBe("error");
		expect(result.diagnostics).toEqual([
			expect.objectContaining({
				type: "provider_transport_failure",
				details: {
					configuredTransport: "auto",
					eventsEmitted: true,
					phase: "after_message_stream_start",
					requestBytes: expect.any(Number),
				},
			}),
		]);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("retries a partial automatic WebSocket summary on SSE, without retaining the partial text", async () => {
		harness.enqueue("partial-disconnect");
		const result = await summarize();
		expect(result.stopReason).toBe("stop");
		expect(result.content).toEqual([expect.objectContaining({ type: "text", text: "Recovered summary" })]);
		expect(harness.connections).toHaveLength(1);
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(attemptedOptions.map((options) => options.transport)).toEqual(["auto", "sse"]);
		expect(attemptedOptions[0]).not.toBe(attemptedOptions[1]);
		expect(harness.connections[0].bodies[0].prompt_cache_key).toBeUndefined();

		await summarize();
		expect(harness.connections).toHaveLength(2);
		expect(fetchMock).toHaveBeenCalledTimes(1);
		expect(attemptedOptions[2].transport).toBe("auto");
	});

	it("also recovers from an idle timeout after response.created", async () => {
		vi.useFakeTimers();
		harness.enqueue("idle");
		const pending = summarize();
		await vi.runAllTimersAsync();
		expect((await pending).stopReason).toBe("stop");
		expect(harness.connections).toHaveLength(1);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it("retains the provider's existing fallback before the first WebSocket event", async () => {
		harness.enqueue("prestart-disconnect");
		const result = await summarize();
		expect(result.stopReason).toBe("stop");
		expect(attemptedOptions).toHaveLength(1);
		expect(harness.connections).toHaveLength(1);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it.each(["websocket", "websocket-cached"] as const)("does not replace explicit %s with SSE", async (transport) => {
		harness.enqueue("disconnect");
		const result = await summarize({ transport });
		expect(result.stopReason).toBe("stop");
		expect(harness.connections).toHaveLength(2);
		expect(fetchMock).not.toHaveBeenCalled();
		expect(attemptedOptions.map((options) => options.transport)).toEqual([transport, transport]);
	});

	it.each([
		{ ...retry, enabled: false },
		{ ...retry, maxRetries: 0 },
	])("does not add attempts when retry policy forbids them: %j", async (policy) => {
		harness.enqueue("disconnect");
		expect((await summarize({}, policy)).stopReason).toBe("error");
		expect(harness.connections).toHaveLength(1);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("does not turn a provider refusal into a transport fallback", async () => {
		harness.enqueue("api-error");
		const result = await summarize();
		expect(result.stopReason).toBe("error");
		expect(result.errorMessage).toContain("Synthetic refusal");
		expect(result.diagnostics).toBeUndefined();
		expect(attemptedOptions).toHaveLength(1);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("does not apply Codex fallback metadata to another provider API", async () => {
		harness.enqueue("disconnect");
		const failure = await streamSimple(codexModel, syntheticContext(), {
			apiKey: syntheticToken(),
			transport: "auto",
			cacheRetention: "none",
		}).result();
		const attempts: Array<SimpleStreamOptions | undefined> = [];
		await completeSummarization(
			codexModel,
			syntheticContext(),
			{ transport: "auto" },
			(_model, _context, options) => {
				attempts.push(options);
				const stream = createAssistantMessageEventStream();
				stream.end({ ...failure, api: "openai-completions" });
				return stream;
			},
			retry,
		);
		expect(attempts.map((options) => options?.transport)).toEqual(["auto", "auto"]);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("cancels during backoff without starting the fallback attempt", async () => {
		harness.enqueue("disconnect");
		const controller = new AbortController();
		const result = await summarize({ signal: controller.signal }, retry, {
			onRetryScheduled: () => {
				controller.abort();
			},
		});
		expect(result.stopReason).toBe("aborted");
		expect(attemptedOptions).toHaveLength(1);
		expect(fetchMock).not.toHaveBeenCalled();
	});

	it("cancels an SSE fallback without starting another attempt", async () => {
		harness.enqueue("disconnect");
		const controller = new AbortController();
		fetchMock.mockImplementationOnce(async () => {
			controller.abort();
			throw new Error("Request was aborted");
		});
		const result = await summarize({ signal: controller.signal }, { ...retry, maxRetries: 3 });
		expect(result.stopReason).toBe("aborted");
		expect(attemptedOptions).toHaveLength(2);
		expect(harness.connections).toHaveLength(1);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it("keeps repeated SSE failures inside the existing summary retry budget", async () => {
		harness.enqueue("disconnect");
		fetchMock.mockImplementation(async () => new Response("Synthetic overloaded", { status: 503 }));
		const result = await summarize({}, { ...retry, maxRetries: 2 });
		expect(result.stopReason).toBe("error");
		expect(attemptedOptions.map((options) => options.transport)).toEqual(["auto", "sse", "sse"]);
		expect(harness.connections).toHaveLength(1);
		expect(fetchMock).toHaveBeenCalledTimes(2);
	});

	it("does not evict the ordinary caller's cached socket or continuation", async () => {
		const context = syntheticContext();
		const ordinaryOptions: SimpleStreamOptions = {
			apiKey: syntheticToken(),
			transport: "auto",
			sessionId: "ordinary",
		};
		const first = await streamSimple(codexModel, context, ordinaryOptions).result();
		expect(first.stopReason).toBe("stop");
		harness.enqueue("partial-disconnect");
		expect((await summarize({ sessionId: "ordinary" })).stopReason).toBe("stop");
		const next = await streamSimple(
			codexModel,
			{
				...context,
				messages: [...context.messages, first, { role: "user", content: "Continue", timestamp: 2 }],
			},
			ordinaryOptions,
		).result();
		expect(next.stopReason).toBe("stop");
		expect(harness.connections).toHaveLength(2);
		expect(harness.connections[0].bodies[1].previous_response_id).toBe(first.responseId);
		expect(harness.connections[0].bodies[1].prompt_cache_key).toBe("ordinary");
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});
});
