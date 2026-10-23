import type { Model, TranscriptContext } from "../../src/types.ts";
import { normalizeContext } from "../../src/utils/transcript.ts";

export const codexModel: Model<"openai-codex-responses"> = {
	id: "synthetic-codex",
	name: "Synthetic Codex",
	api: "openai-codex-responses",
	provider: "openai-codex",
	baseUrl: "https://first.synthetic.invalid/backend-api",
	reasoning: true,
	input: ["text"],
	contextWindow: 100000,
	maxTokens: 8192,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

export function syntheticToken(account = "account-a", signature = "synthetic"): string {
	const payload = Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: account } }));
	return `synthetic.${payload.toString("base64url")}.${signature}`;
}

export function syntheticContext(): TranscriptContext {
	return normalizeContext({
		systemPrompt: "Synthetic conversation only.",
		messages: [{ role: "user", content: "Synthetic turn", timestamp: 1 }],
	});
}

function textEvents(text: string, id: string): Record<string, unknown>[] {
	return [
		{
			type: "response.output_item.added",
			item: { type: "message", id, role: "assistant", status: "in_progress", content: [] },
		},
		{ type: "response.content_part.added", part: { type: "output_text", text: "" } },
		{ type: "response.output_text.delta", delta: text },
		{
			type: "response.output_item.done",
			item: {
				type: "message",
				id,
				role: "assistant",
				status: "completed",
				content: [{ type: "output_text", text }],
			},
		},
	];
}

function completion(id: string): Record<string, unknown> {
	return {
		type: "response.completed",
		response: {
			id,
			status: "completed",
			output: [],
			usage: { input_tokens: 5, output_tokens: 3, total_tokens: 8 },
		},
	};
}

export function syntheticSseResponse(text = "Complete summary"): Response {
	const events = [...textEvents(text, "sse-message"), completion("sse-response")];
	return new Response(`${events.map((event) => `data: ${JSON.stringify(event)}`).join("\n\n")}\n\n`, {
		headers: { "content-type": "text/event-stream" },
	});
}

type Behavior =
	| "complete"
	| "disconnect"
	| "partial-disconnect"
	| "idle"
	| "hold"
	| "prestart-disconnect"
	| "api-error";

interface SyntheticBody {
	input?: unknown[];
	previous_response_id?: string;
	prompt_cache_key?: string;
}

export function createCodexWebSocketHarness() {
	const connections: SyntheticWebSocket[] = [];
	const behaviors: Behavior[] = [];
	let sequence = 0;

	class SyntheticWebSocket extends EventTarget {
		readyState = 0;
		readonly url: string;
		readonly account: string | undefined;
		readonly bodies: SyntheticBody[] = [];
		readonly closes: Array<{ code?: number; reason?: string }> = [];

		constructor(url: string, options?: { headers?: Record<string, string> }) {
			super();
			this.url = url;
			this.account = options?.headers?.["chatgpt-account-id"];
			connections.push(this);
			queueMicrotask(() => {
				this.readyState = 1;
				this.dispatchEvent(new Event("open"));
			});
		}

		emit(event: Record<string, unknown>): void {
			this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify(event) }));
		}

		send(body: string): void {
			this.bodies.push(JSON.parse(body) as SyntheticBody);
			const behavior = behaviors.shift() ?? "complete";
			queueMicrotask(() => {
				if (behavior === "prestart-disconnect") {
					this.dispatchEvent(new Event("error"));
					return;
				}
				this.emit({ type: "response.created", response: { id: `created-${++sequence}`, status: "in_progress" } });
				if (behavior === "idle" || behavior === "hold") return;
				if (behavior === "api-error") {
					this.emit({ type: "error", error: { code: "invalid_request", message: "Synthetic refusal" } });
					return;
				}
				if (behavior !== "disconnect") {
					for (const event of textEvents(
						behavior === "partial-disconnect" ? "Incomplete" : "Complete summary",
						`msg-${sequence}`,
					))
						this.emit(event);
				}
				if (behavior === "disconnect" || behavior === "partial-disconnect") {
					queueMicrotask(() => this.dispatchEvent(new Event("error")));
				} else {
					this.emit(completion(`response-${sequence}`));
				}
			});
		}

		close(code?: number, reason?: string): void {
			this.readyState = 3;
			this.closes.push({ code, reason });
		}
	}

	return { WebSocket: SyntheticWebSocket, connections, enqueue: (...next: Behavior[]) => behaviors.push(...next) };
}
