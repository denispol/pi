/**
 * C-ID account authority (shared/planning#565).
 *
 * The auth owner (auth resolution) establishes, per provider, which account
 * namespace is currently selected and at which selection revision. Every
 * governed send verifies the key material it actually uses against that
 * record before the first byte leaves. A wrong account, a stale selection,
 * or a mid-preparation swap fails closed with zero sends.
 *
 * Vocabulary:
 * - namespace: nonsecret account identity. OAuth account id when the
 *   credential carries one, else the JWT account claim, else a truncated
 *   SHA-256 fingerprint of the key (proves replacement, reveals nothing).
 * - selectionRevision: per-provider monotonic counter. Bumps ONLY when the
 *   namespace changes (account replacement), never on same-account token
 *   rotation, so legitimate refresh keeps working.
 *
 * Single-process scope: two live sessions under different accounts on the
 * same provider in one process cannot be disambiguated here and are refused
 * rather than guessed (fail-closed, documented).
 */

import type { createHash } from "node:crypto";
import type { OAuthCredential } from "./types.ts";

export interface SelectionRecord {
	namespace: string;
	selectionRevision: number;
	modelId?: string;
	providerId?: string;
}

const registry = new Map<string, SelectionRecord & { modelId: string; providerId: string }>();

type ProcessWithCryptoBuiltinModule = typeof process & {
	getBuiltinModule?: (id: "node:crypto") => { createHash: typeof createHash };
};

// process.getBuiltinModule (repo pattern, see openai-codex-responses.ts zlib)
// keeps this module browser-bundle-safe: bundlers see no node:crypto import
// edge, yet under Node the digest is identical to createHash("sha256"). Test
// environments that shadow globalThis.crypto are unaffected. Non-Node
// runtimes fail closed rather than emit a weak fingerprint.
function fingerprintKey(key: string): string {
	const crypto =
		typeof process === "undefined"
			? undefined
			: (process as ProcessWithCryptoBuiltinModule).getBuiltinModule?.("node:crypto");
	if (!crypto) {
		throw new Error("node:crypto unavailable: key fingerprinting requires a Node-compatible runtime");
	}
	return `keyfp:${crypto.createHash("sha256").update(key, "utf8").digest("hex").slice(0, 16)}`;
}

/**
 * Derive the nonsecret namespace for key material. Prefers an explicit
 * credential account id (selection-time truth), then the JWT claim, then a
 * key fingerprint (replacement detection without secret exposure).
 */
export function deriveNamespace(keyMaterial: string | undefined, credentialAccountId?: string | null): string | null {
	if (typeof credentialAccountId === "string" && credentialAccountId.length > 0) {
		return `acct:${credentialAccountId}`;
	}
	if (typeof keyMaterial !== "string" || keyMaterial.length === 0) return null;
	try {
		const parts = keyMaterial.split(".");
		if (parts.length === 3) {
			const payload = JSON.parse(atob(parts[1]));
			const claim = payload?.["https://api.openai.com/auth"]?.chatgpt_account_id;
			if (typeof claim === "string" && claim.length > 0) return `acct:${claim}`;
		}
	} catch {
		// Not a JWT: fall through to the fingerprint.
	}
	return fingerprintKey(keyMaterial);
}

/**
 * Record the currently selected namespace for a provider. Called by auth
 * resolution (the owner) after every successful resolution. A namespace
 * change bumps the revision; anything else keeps it.
 */
export function noteSelectedNamespace(
	providerId: string,
	namespace: string | null,
	modelId?: string,
): SelectionRecord | null {
	if (namespace === null) return null;
	const prior = registry.get(providerId);
	if (prior && prior.namespace === namespace) {
		if (modelId !== undefined) prior.modelId = modelId;
		return { namespace: prior.namespace, selectionRevision: prior.selectionRevision };
	}
	const next: SelectionRecord & { modelId: string; providerId: string } = {
		namespace,
		selectionRevision: (prior?.selectionRevision ?? 0) + 1,
		modelId: modelId ?? prior?.modelId ?? "",
		providerId,
	};
	registry.set(providerId, next);
	return { namespace: next.namespace, selectionRevision: next.selectionRevision };
}

/** Current selection record for a provider, if any was established. */
export function currentSelection(providerId: string): SelectionRecord | null {
	const r = registry.get(providerId);
	return r ? { namespace: r.namespace, selectionRevision: r.selectionRevision } : null;
}

/** Test/owner reset for one provider. Never used on the send path. */
export function resetSelection(providerId: string): void {
	registry.delete(providerId);
}

/**
 * C-ID refresh-vs-replacement (F-CID-5): a refresh rotates tokens for the
 * SAME account. If the refreshed credential names a different account, the
 * stored credential was replaced out from under the session — refuse here
 * rather than letting the new account masquerade as a routine rotation.
 *
 * Lives here (not in the oauth module) so callers in the browser-reachable
 * graph never pull in the Node-only OAuth callback stack.
 */
export function assertSameAccountRefresh(previous: OAuthCredential, next: OAuthCredential): OAuthCredential {
	const before = typeof previous.accountId === "string" ? previous.accountId : null;
	const after = typeof next.accountId === "string" ? next.accountId : null;
	if (before !== null && after !== null && before !== after) {
		throw new Error(
			`OpenAI Codex credential replaced during refresh (account changed); re-login instead of silently following the switch`,
		);
	}
	return next;
}
