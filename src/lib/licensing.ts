// Commercial licensing & access control for exported artifacts (datasets, Q-tables, policies).
// Pure client-side: license terms are stamped into every export with a tamper-evident
// fingerprint, optionally signed with the publisher's ECDSA P-256 key (Web Crypto), and
// exports can be locked behind an access key for proprietary work.

export type LicenseKind = "proprietary" | "internal" | "evaluation" | "cc-by-4.0" | "mit";

export interface LicenseConfig {
  kind: LicenseKind;
  organization: string;
  contact: string;
  /** Comma/newline separated list of people or teams allowed to use the artifact. */
  authorizedUsers: string;
  /** ISO date (yyyy-mm-dd) after which the license is void; empty = perpetual. */
  expiresAt: string;
  notes: string;
  /** When true, downloads are blocked until the correct access key is entered. */
  exportLock: boolean;
  accessKey: string;
}

export interface LicenseSignature {
  algorithm: "ECDSA-P256-SHA256";
  /** First 64 bits of SHA-256 over the public key, grouped as xxxx-xxxx-xxxx-xxxx. */
  keyId: string;
  publicKey: JsonWebKey;
  /** Base64 raw (r||s) signature over the license basis + fingerprint + contact. */
  value: string;
}

export interface LicenseBlock {
  kind: LicenseKind;
  name: string;
  terms: string;
  organization: string | null;
  contact: string | null;
  authorizedUsers: string[];
  issuedAt: string;
  expiresAt: string | null;
  notes: string | null;
  /** FNV-1a over the license terms + the policy checksum — detects edited license blocks. */
  fingerprint: string;
  /** Present when the publisher signed the export — proves origin, not just integrity. */
  signature?: LicenseSignature;
}

export interface SigningKey {
  keyId: string;
  label: string;
  createdAt: string;
  privateKey: CryptoKey;
  publicJwk: JsonWebKey;
}

export const DEFAULT_LICENSE: LicenseConfig = {
  kind: "proprietary",
  organization: "",
  contact: "",
  authorizedUsers: "",
  expiresAt: "",
  notes: "",
  exportLock: false,
  accessKey: "",
};

export const LICENSE_INFO: Record<LicenseKind, { name: string; terms: string }> = {
  proprietary: {
    name: "Proprietary — All rights reserved",
    terms:
      "Confidential. Use, reproduction and redistribution are prohibited except by the authorized users listed below. Derived models and evaluations inherit these terms.",
  },
  internal: {
    name: "Internal research use only",
    terms:
      "May be used and modified within the issuing organization for research purposes. No external distribution, publication or commercial deployment without written approval.",
  },
  evaluation: {
    name: "Evaluation license (non-commercial)",
    terms:
      "Granted for benchmarking and evaluation only. No production use, no redistribution, no training of derivative commercial models. Results may be published with attribution.",
  },
  "cc-by-4.0": {
    name: "Creative Commons Attribution 4.0",
    terms: "Free to share and adapt for any purpose, including commercial, provided appropriate credit is given to the issuing organization.",
  },
  mit: {
    name: "MIT License",
    terms: "Permission is granted, free of charge, to use, copy, modify and distribute, subject to inclusion of this notice. Provided as-is, without warranty.",
  },
};

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function parseUsers(raw: string): string[] {
  return raw
    .split(/[,\n;]+/)
    .map((u) => u.trim())
    .filter(Boolean)
    .slice(0, 50);
}

export function isExpired(cfg: LicenseConfig): boolean {
  if (!cfg.expiresAt) return false;
  const t = Date.parse(`${cfg.expiresAt}T23:59:59`);
  return Number.isFinite(t) && t < Date.now();
}

type BasisFields = Omit<LicenseBlock, "fingerprint" | "name" | "contact" | "signature">;

function licenseBasis(b: BasisFields, artifactChecksum: string): string {
  return [b.kind, b.terms, b.organization ?? "", b.authorizedUsers.join("|"), b.expiresAt ?? "", b.notes ?? "", b.issuedAt, artifactChecksum].join("\u0000");
}

function fingerprintOf(b: BasisFields, artifactChecksum: string): string {
  return fnv1a(licenseBasis(b, artifactChecksum));
}

/** Builds the license block stamped into an export. `artifactChecksum` ties it to the payload. */
export function buildLicenseBlock(cfg: LicenseConfig, artifactChecksum: string): LicenseBlock {
  const info = LICENSE_INFO[cfg.kind];
  const users = parseUsers(cfg.authorizedUsers);
  const issuedAt = new Date().toISOString();
  const fingerprint = fingerprintOf(
    {
      kind: cfg.kind,
      terms: info.terms,
      organization: cfg.organization.trim() || null,
      authorizedUsers: users,
      issuedAt,
      expiresAt: cfg.expiresAt || null,
      notes: cfg.notes.trim() || null,
    },
    artifactChecksum,
  );
  return {
    kind: cfg.kind,
    name: info.name,
    terms: info.terms,
    organization: cfg.organization.trim() || null,
    contact: cfg.contact.trim() || null,
    authorizedUsers: users,
    issuedAt,
    expiresAt: cfg.expiresAt || null,
    notes: cfg.notes.trim() || null,
    fingerprint,
  };
}

// ---------------------------------------------------------------------------------------------
// Publisher signing (ECDSA P-256 / SHA-256 via Web Crypto)
// ---------------------------------------------------------------------------------------------

const EC = { name: "ECDSA", namedCurve: "P-256" } as const;
const SIG = { name: "ECDSA", hash: "SHA-256" } as const;
const KEY_FILE_FORMAT = "offline-learning-lab/signing-key";

function subtle(): SubtleCrypto {
  const s = typeof crypto !== "undefined" ? crypto.subtle : undefined;
  if (!s) throw new Error("Signing needs a secure (https) context with Web Crypto available.");
  return s;
}

const toB64 = (bytes: Uint8Array) => btoa(String.fromCharCode(...bytes));
const fromB64 = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const pickPublic = (j: JsonWebKey): JsonWebKey => ({ kty: "EC", crv: "P-256", x: j.x, y: j.y });

/** Normalizes a user-typed key ID ("A1B2 c3d4-…") to lowercase hex for comparison. */
export function normalizeKeyId(id: string): string {
  return id.toLowerCase().replace(/[^0-9a-f]/g, "");
}

async function keyIdOf(jwk: JsonWebKey): Promise<string> {
  const data = new TextEncoder().encode(`P-256.${jwk.x ?? ""}.${jwk.y ?? ""}`);
  const h = new Uint8Array(await subtle().digest("SHA-256", data));
  const hex = [...h.slice(0, 8)].map((b) => b.toString(16).padStart(2, "0")).join("");
  return hex.match(/.{4}/g)!.join("-");
}

export async function generateSigningKey(label: string): Promise<SigningKey> {
  const pair = (await subtle().generateKey(EC, true, ["sign", "verify"])) as CryptoKeyPair;
  const publicJwk = pickPublic(await subtle().exportKey("jwk", pair.publicKey));
  return {
    keyId: await keyIdOf(publicJwk),
    label: label.trim() || "Publisher key",
    createdAt: new Date().toISOString(),
    privateKey: pair.privateKey,
    publicJwk,
  };
}

/** Serializes the key (including the PRIVATE half) so the publisher can back it up and reuse it. */
export async function exportSigningKeyFile(key: SigningKey): Promise<string> {
  const privateJwk = await subtle().exportKey("jwk", key.privateKey);
  return JSON.stringify(
    { format: KEY_FILE_FORMAT, version: 1, label: key.label, keyId: key.keyId, createdAt: key.createdAt, publicKey: key.publicJwk, privateKey: privateJwk },
    null,
    2,
  );
}

export async function importSigningKeyFile(raw: string): Promise<SigningKey> {
  let doc: Record<string, unknown>;
  try {
    doc = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    throw new Error("Key file is not valid JSON.");
  }
  if (doc?.format !== KEY_FILE_FORMAT) throw new Error("Not an Offline Learning Lab signing-key file.");
  const priv = doc.privateKey as JsonWebKey | undefined;
  if (!priv || priv.kty !== "EC" || priv.crv !== "P-256" || !priv.d || !priv.x || !priv.y) throw new Error("Key file has no usable P-256 private key.");
  let privateKey: CryptoKey;
  try {
    privateKey = await subtle().importKey("jwk", priv, EC, true, ["sign"]);
  } catch {
    throw new Error("The private key in this file is corrupted.");
  }
  const publicJwk = pickPublic(priv);
  return {
    keyId: await keyIdOf(publicJwk),
    label: typeof doc.label === "string" && doc.label.trim() ? doc.label.trim().slice(0, 80) : "Imported key",
    createdAt: typeof doc.createdAt === "string" ? doc.createdAt : new Date().toISOString(),
    privateKey,
    publicJwk,
  };
}

function signedMessage(b: LicenseBlock, artifactChecksum: string): Uint8Array {
  const basis = licenseBasis(
    { kind: b.kind, terms: String(b.terms ?? ""), organization: b.organization ?? null, authorizedUsers: b.authorizedUsers, issuedAt: String(b.issuedAt ?? ""), expiresAt: b.expiresAt ?? null, notes: b.notes ?? null },
    artifactChecksum,
  );
  return new TextEncoder().encode(`${basis}\u0000${b.fingerprint}\u0000${b.contact ?? ""}`);
}

/** Returns a copy of the block with a publisher signature over terms, users, expiry, contact and weights checksum. */
export async function signLicenseBlock(block: LicenseBlock, artifactChecksum: string, key: SigningKey): Promise<LicenseBlock> {
  const sig = new Uint8Array(await subtle().sign(SIG, key.privateKey, signedMessage(block, artifactChecksum) as BufferSource));
  return { ...block, signature: { algorithm: "ECDSA-P256-SHA256", keyId: key.keyId, publicKey: key.publicJwk, value: toB64(sig) } };
}

// ---------------------------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------------------------

export interface VerifyCheck {
  label: string;
  ok: boolean;
  detail: string;
}

export interface VerifyResult {
  status: "valid" | "tampered" | "expired" | "unlicensed" | "invalid" | "untrusted";
  summary: string;
  checks: VerifyCheck[];
  license: LicenseBlock | null;
  /** Signer info when the export carries a signature. */
  signer?: { keyId: string; verified: boolean; matchesExpected: boolean | null } | null;
}

/** Verifies an exported policy file: Q-table checksum, license fingerprint, canonical terms and expiry. */
export function verifyLicensedExport(raw: string): VerifyResult {
  let doc: Record<string, unknown>;
  try {
    doc = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return { status: "invalid", summary: "Not a valid JSON file.", checks: [], license: null };
  }
  if (doc?.format !== "offline-learning-lab/policy") {
    return { status: "invalid", summary: "Not an Offline Learning Lab policy export.", checks: [], license: null };
  }
  const checks: VerifyCheck[] = [];
  const policy = doc.policy as { qValues?: unknown } | undefined;
  const recon = doc.reconstruction as { checksum?: { value?: string } } | undefined;
  const stored = recon?.checksum?.value ?? "";
  const actual = Array.isArray(policy?.qValues) ? fnv1a(JSON.stringify(policy!.qValues)) : "";
  const qOk = !!actual && actual === stored;
  checks.push({ label: "Q-table checksum", ok: qOk, detail: qOk ? `matches ${stored}` : `stored ${stored || "—"}, computed ${actual || "—"}` });

  const lic = doc.license as LicenseBlock | undefined;
  if (!lic || typeof lic.fingerprint !== "string" || !(lic.kind in LICENSE_INFO)) {
    return {
      status: qOk ? "unlicensed" : "tampered",
      summary: qOk ? "Weights are intact, but the file carries no license block." : "Weights were modified after export and no license block is present.",
      checks,
      license: null,
    };
  }
  const users = Array.isArray(lic.authorizedUsers) ? lic.authorizedUsers.map(String) : [];
  const fp = fingerprintOf(
    { kind: lic.kind, terms: String(lic.terms ?? ""), organization: lic.organization ?? null, authorizedUsers: users, issuedAt: String(lic.issuedAt ?? ""), expiresAt: lic.expiresAt ?? null, notes: lic.notes ?? null },
    stored,
  );
  const fpOk = fp === lic.fingerprint;
  checks.push({ label: "License fingerprint", ok: fpOk, detail: fpOk ? `matches ${lic.fingerprint}` : `stamped ${lic.fingerprint}, recomputed ${fp}` });
  const termsOk = lic.terms === LICENSE_INFO[lic.kind].terms;
  checks.push({ label: "Standard terms", ok: termsOk, detail: termsOk ? `${LICENSE_INFO[lic.kind].name} — unaltered` : "terms text differs from the canonical license" });
  const expT = lic.expiresAt ? Date.parse(`${lic.expiresAt}T23:59:59`) : NaN;
  const notExpired = !lic.expiresAt || !Number.isFinite(expT) || expT >= Date.now();
  checks.push({ label: "Expiry", ok: notExpired, detail: lic.expiresAt ? `${notExpired ? "valid until" : "expired on"} ${lic.expiresAt}` : "perpetual" });

  const intact = qOk && fpOk && termsOk;
  const status: VerifyResult["status"] = !intact ? "tampered" : notExpired ? "valid" : "expired";
  const summary =
    status === "valid"
      ? "Authentic: weights and license terms are exactly as issued."
      : status === "expired"
        ? "Authentic, but the license has expired — the artifact may no longer be used."
        : !qOk && fpOk
          ? "Weights were modified after export — the license no longer covers this file."
          : "License terms were edited after export — do not rely on the stated permissions.";
  return { status, summary, checks, license: { ...lic, authorizedUsers: users } };
}

/**
 * Full verification including the publisher signature. When `expectedKeyId` is given, the file
 * must be signed by exactly that key — this is what defeats deliberate re-fingerprinting.
 */
export async function verifyLicensedExportWithSignature(raw: string, expectedKeyId = ""): Promise<VerifyResult> {
  const base = verifyLicensedExport(raw);
  if (!base.license) return base;
  const lic = base.license;
  const doc = JSON.parse(raw) as { reconstruction?: { checksum?: { value?: string } } };
  const stored = doc.reconstruction?.checksum?.value ?? "";
  const expected = normalizeKeyId(expectedKeyId);
  const checks = [...base.checks];
  let { status, summary } = base;
  const sig = lic.signature;

  if (!sig || typeof sig.value !== "string" || !sig.publicKey || typeof sig.keyId !== "string") {
    checks.push({ label: "Publisher signature", ok: false, detail: "not signed — origin can't be proven" });
    if (expected && status !== "tampered") {
      status = "untrusted";
      summary = "File is not signed, so it can't be confirmed to come from the expected publisher.";
    } else if (status === "valid") {
      summary = "Intact, but unsigned — the fingerprint proves integrity, not who issued it.";
    }
    return { ...base, status, summary, checks, signer: null };
  }

  let verified = false;
  try {
    const pub = await subtle().importKey("jwk", pickPublic(sig.publicKey), EC, false, ["verify"]);
    const idOk = (await keyIdOf(sig.publicKey)) === sig.keyId;
    verified = idOk && (await subtle().verify(SIG, pub, fromB64(sig.value) as BufferSource, signedMessage(lic, stored) as BufferSource));
  } catch {
    verified = false;
  }
  const matchesExpected = expected ? normalizeKeyId(sig.keyId) === expected : null;
  checks.push({ label: "Publisher signature", ok: verified, detail: verified ? `ECDSA P-256 · key ${sig.keyId}` : "signature does not match the contents" });
  if (expected) {
    checks.push({ label: "Expected publisher", ok: !!matchesExpected, detail: matchesExpected ? "key ID matches" : `signed by ${sig.keyId}` });
  }

  if (!verified) {
    status = "tampered";
    summary = "Signature check failed — the license, contact or weights changed after signing, or the signature was forged.";
  } else if (expected && !matchesExpected && status !== "tampered") {
    status = "untrusted";
    summary = "Validly signed, but by a different key than the publisher you expected — treat it as unofficial.";
  } else if (status === "valid") {
    summary = expected ? "Authentic and signed by the expected publisher — terms and weights are exactly as issued." : `Authentic and signed by key ${sig.keyId}. Compare this key ID with the one your publisher shared.`;
  }
  return { ...base, status, summary, checks, signer: { keyId: sig.keyId, verified, matchesExpected } };
}
