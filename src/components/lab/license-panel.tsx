// Commercial licensing & access control panel: configures the license stamped into every
// export and optionally locks downloads behind an access key for proprietary artifacts.

import { useState } from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import {
  exportSigningKeyFile,
  generateSigningKey,
  importSigningKeyFile,
  isExpired,
  parseUsers,
  verifyLicensedExportWithSignature,
  LICENSE_INFO,
  type LicenseConfig,
  type LicenseKind,
  type SigningKey,
  type VerifyResult,
} from "@/lib/licensing";

const KINDS: LicenseKind[] = ["proprietary", "internal", "evaluation", "cc-by-4.0", "mit"];

export function LicensePanel({
  cfg,
  onChange,
  signingKey,
  onSigningKeyChange,
}: {
  cfg: LicenseConfig;
  onChange: (c: LicenseConfig) => void;
  signingKey: SigningKey | null;
  onSigningKeyChange: (k: SigningKey | null) => void;
}) {
  const set = (patch: Partial<LicenseConfig>) => onChange({ ...cfg, ...patch });
  const users = parseUsers(cfg.authorizedUsers);
  const expired = isExpired(cfg);

  return (
    <div className="space-y-3">
      <div>
        <label className="mb-1.5 block text-[11px] text-muted-foreground">License</label>
        <Select value={cfg.kind} onValueChange={(v) => set({ kind: v as LicenseKind })}>
          <SelectTrigger className="h-9 px-3 text-[12.5px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {KINDS.map((k) => (
              <SelectItem key={k} value={k}>
                {LICENSE_INFO[k].name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="mt-1.5 text-[10.5px] leading-relaxed text-muted-foreground">{LICENSE_INFO[cfg.kind].terms}</p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label htmlFor="lic-org" className="mb-1.5 block text-[11px] text-muted-foreground">Organization</label>
          <input
            id="lic-org"
            value={cfg.organization}
            onChange={(e) => set({ organization: e.target.value })}
            placeholder="Acme Robotics Research"
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-[12.5px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>
        <div>
          <label htmlFor="lic-contact" className="mb-1.5 block text-[11px] text-muted-foreground">Licensing contact</label>
          <input
            id="lic-contact"
            value={cfg.contact}
            onChange={(e) => set({ contact: e.target.value })}
            placeholder="licensing@acme.com"
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-[12.5px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
        </div>
      </div>

      <div>
        <label htmlFor="lic-users" className="mb-1.5 block text-[11px] text-muted-foreground">
          Authorized users <span className="text-muted-foreground/70">(comma-separated)</span>
        </label>
        <textarea
          id="lic-users"
          value={cfg.authorizedUsers}
          onChange={(e) => set({ authorizedUsers: e.target.value })}
          placeholder="r.chen@acme.com, controls-team, j.okafor@acme.com"
          rows={2}
          className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-[12.5px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <p className="mt-1 text-[10.5px] text-muted-foreground">
          {users.length === 0 ? "No users listed — the license applies to the organization as a whole." : `${users.length} authorized user${users.length === 1 ? "" : "s"} will be named in every export.`}
        </p>
      </div>

      <div>
        <label htmlFor="lic-exp" className="mb-1.5 block text-[11px] text-muted-foreground">License expires</label>
        <input
          id="lic-exp"
          type="date"
          value={cfg.expiresAt}
          onChange={(e) => set({ expiresAt: e.target.value })}
          className="h-9 w-full rounded-md border border-input bg-background px-3 font-mono text-[12px] text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        {expired && <p className="mt-1 text-[10.5px] font-medium text-destructive">This license date has passed — exports are blocked until you extend or clear it.</p>}
      </div>

      <div>
        <label htmlFor="lic-notes" className="mb-1.5 block text-[11px] text-muted-foreground">
          Additional terms <span className="text-muted-foreground/70">(optional)</span>
        </label>
        <textarea
          id="lic-notes"
          value={cfg.notes}
          onChange={(e) => set({ notes: e.target.value })}
          placeholder="e.g. Not for deployment on flight hardware. Cite contract #RL-2207 when publishing results."
          rows={2}
          maxLength={500}
          className="w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-[12.5px] text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <p className="mt-1 flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
          <span>Stamped into every export and covered by the fingerprint.</span>
          <span className="tabular-nums">{cfg.notes.length}/500</span>
        </p>
      </div>

      <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-surface/60 px-3 py-2.5">
        <span className="text-[11.5px] leading-snug text-foreground">
          Lock exports behind an access key
          <span className="block text-[10.5px] text-muted-foreground">Downloads require the key — protects proprietary policies on shared machines</span>
        </span>
        <Switch checked={cfg.exportLock} onCheckedChange={(v) => set({ exportLock: v })} aria-label="Lock exports behind an access key" />
      </label>

      {cfg.exportLock && (
        <ExportKeyField value={cfg.accessKey} onChange={(accessKey) => set({ accessKey })} />
      )}

      <p className="text-[10.5px] leading-relaxed text-muted-foreground">
        Every exported model embeds this license with a tamper-evident fingerprint tied to the Q-table checksum. Sign exports with a publisher key so recipients can also prove who issued them.
      </p>

      <SigningKeyPanel signingKey={signingKey} onChange={onSigningKeyChange} organization={cfg.organization} />

      <ExportVerifier ownKeyId={signingKey?.keyId ?? ""} />
    </div>
  );
}

const STATUS_STYLE: Record<VerifyResult["status"], { label: string; cls: string }> = {
  valid: { label: "Authentic", cls: "border-success/40 bg-success/10 text-success" },
  expired: { label: "Expired", cls: "border-warning/40 bg-warning/10 text-warning" },
  tampered: { label: "Tampered", cls: "border-destructive/40 bg-destructive/10 text-destructive" },
  unlicensed: { label: "No license", cls: "border-warning/40 bg-warning/10 text-warning" },
  invalid: { label: "Unreadable", cls: "border-destructive/40 bg-destructive/10 text-destructive" },
  untrusted: { label: "Wrong signer", cls: "border-warning/40 bg-warning/10 text-warning" },
};

function SigningKeyPanel({ signingKey, onChange, organization }: { signingKey: SigningKey | null; onChange: (k: SigningKey | null) => void; organization: string }) {
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  const create = async () => {
    setBusy(true);
    try {
      const k = await generateSigningKey(organization ? `${organization.trim()} publisher key` : "Publisher key");
      onChange(k);
      toast.success(`Created signing key ${k.keyId} — back it up to keep using it.`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Could not create a signing key.");
    } finally {
      setBusy(false);
    }
  };

  const backup = async () => {
    if (!signingKey) return;
    try {
      const text = await exportSigningKeyFile(signingKey);
      const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `signing-key-${signingKey.keyId}.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("Key backup downloaded — store it privately; anyone holding it can sign as you.");
    } catch {
      toast.error("Could not export the key.");
    }
  };

  const restore = (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    file
      .text()
      .then(importSigningKeyFile)
      .then((k) => {
        onChange(k);
        toast.success(`Loaded signing key ${k.keyId}`);
      })
      .catch((e) => toast.error(e instanceof Error ? e.message : "Could not load that key file."))
      .finally(() => setBusy(false));
  };

  const copyId = () => {
    if (!signingKey) return;
    navigator.clipboard
      ?.writeText(signingKey.keyId)
      .then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => toast.error("Clipboard unavailable — copy the key ID manually."));
  };

  const btn = "h-8 rounded-md border border-border bg-card px-3 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50";

  return (
    <div className="space-y-2.5 border-t border-border pt-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11.5px] font-medium text-foreground">Publisher signature</span>
        <span className={`rounded-full border px-2.5 py-0.5 text-[10px] font-semibold ${signingKey ? "border-success/40 bg-success/10 text-success" : "border-border bg-surface text-muted-foreground"}`}>
          {signingKey ? "signing on" : "unsigned"}
        </span>
      </div>
      {signingKey ? (
        <div className="space-y-2 rounded-lg border border-border bg-surface/60 p-3">
          <div className="text-[11px] text-foreground">{signingKey.label}</div>
          <div className="flex items-center justify-between gap-2">
            <code className="truncate font-mono text-[12px] text-foreground">{signingKey.keyId}</code>
            <button type="button" onClick={copyId} className={btn}>
              {copied ? "Copied" : "Copy ID"}
            </button>
          </div>
          <p className="text-[10.5px] leading-relaxed text-muted-foreground">
            Every export is signed with ECDSA P-256. Share this key ID with recipients so they can confirm files really came from you. The private key stays in this tab — back it up or it's lost on reload.
          </p>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={backup} className={btn}>Back up key</button>
            <button type="button" onClick={() => onChange(null)} className={`${btn} hover:text-destructive`}>Stop signing</button>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-[10.5px] leading-relaxed text-muted-foreground">
            The fingerprint catches edits but can be recalculated by anyone. A signature can only be produced with your private key, so a forged license fails verification.
          </p>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={create} disabled={busy} className="h-8 rounded-md bg-primary px-3 text-[11px] font-semibold text-primary-foreground shadow-sm transition-all hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50">
              {busy ? "Working…" : "Create signing key"}
            </button>
            <label className={`${btn} inline-flex cursor-pointer items-center`}>
              Load key backup
              <input type="file" accept="application/json,.json" className="sr-only" onChange={(e) => { restore(e.target.files?.[0]); e.target.value = ""; }} />
            </label>
          </div>
        </div>
      )}
    </div>
  );
}

function ExportVerifier({ ownKeyId }: { ownKeyId: string }) {
  const [result, setResult] = useState<VerifyResult | null>(null);
  const [fileName, setFileName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [expectedKey, setExpectedKey] = useState("");
  const [checking, setChecking] = useState(false);

  const readFile = (file: File | undefined) => {
    if (!file) return;
    if (file.size > 25 * 1024 * 1024) {
      setFileName(file.name);
      setResult({ status: "invalid", summary: "File is larger than 25 MB — not a policy export.", checks: [], license: null });
      return;
    }
    setFileName(file.name);
    setChecking(true);
    file
      .text()
      .then((t) => verifyLicensedExportWithSignature(t, expectedKey))
      .then(setResult)
      .catch(() => setResult({ status: "invalid", summary: "Could not read the file.", checks: [], license: null }))
      .finally(() => setChecking(false));
  };

  const style = result ? STATUS_STYLE[result.status] : null;
  const lic = result?.license;

  return (
    <div className="space-y-2.5 border-t border-border pt-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11.5px] font-medium text-foreground">Verify an exported model</span>
        {result && (
          <button
            type="button"
            onClick={() => { setResult(null); setFileName(""); }}
            className="rounded-md px-2 py-1 text-[10.5px] text-muted-foreground transition-colors hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            Clear
          </button>
        )}
      </div>
      <div>
        <label htmlFor="lic-expected" className="mb-1.5 flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
          <span>Expected publisher key ID <span className="text-muted-foreground/70">(optional)</span></span>
          {ownKeyId && (
            <button type="button" onClick={() => setExpectedKey(ownKeyId)} className="rounded px-1.5 py-0.5 text-[10.5px] text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              Use mine
            </button>
          )}
        </label>
        <input
          id="lic-expected"
          value={expectedKey}
          onChange={(e) => setExpectedKey(e.target.value)}
          placeholder="e.g. 3f9a-1c02-7be4-d815"
          className="h-9 w-full rounded-md border border-input bg-background px-3 font-mono text-[12px] text-foreground placeholder:font-sans placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
      </div>
      <label
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); readFile(e.dataTransfer.files[0]); }}
        className={`flex cursor-pointer flex-col items-center gap-1 rounded-lg border border-dashed px-3 py-3.5 text-center transition-colors ${dragging ? "border-primary bg-primary/5" : "border-border bg-surface/40 hover:border-primary/50"}`}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground"><path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z" /><path d="M9 12l2 2 4-4" /></svg>
        <span className="text-[11.5px] text-foreground">{checking ? "Verifying…" : fileName || "Drop a policy-*.json file or click to choose"}</span>
        <span className="text-[10px] text-muted-foreground">Checks weights, license fingerprint and publisher signature locally</span>
        <input type="file" accept="application/json,.json" className="sr-only" onChange={(e) => { readFile(e.target.files?.[0]); e.target.value = ""; }} />
      </label>

      {result && style && (
        <div className="space-y-2 rounded-lg border border-border bg-card p-3">
          <div className="flex items-start gap-2">
            <span className={`shrink-0 rounded-full border px-2.5 py-0.5 text-[10px] font-semibold ${style.cls}`}>{style.label}</span>
            <p className="text-[11px] leading-snug text-foreground">{result.summary}</p>
          </div>
          {result.checks.length > 0 && (
            <ul className="space-y-1">
              {result.checks.map((c) => (
                <li key={c.label} className="flex items-start gap-2 text-[10.5px]">
                  <span className={`mt-px font-bold ${c.ok ? "text-success" : "text-destructive"}`}>{c.ok ? "✓" : "✗"}</span>
                  <span className="text-foreground">{c.label}</span>
                  <span className="ml-auto truncate font-mono text-muted-foreground" title={c.detail}>{c.detail}</span>
                </li>
              ))}
            </ul>
          )}
          {lic && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 border-t border-border pt-2 text-[10.5px]">
              <dt className="text-muted-foreground">License</dt><dd className="text-foreground">{lic.name}</dd>
              <dt className="text-muted-foreground">Issued by</dt><dd className="text-foreground">{lic.organization ?? "—"}</dd>
              <dt className="text-muted-foreground">Issued</dt><dd className="font-mono text-foreground">{String(lic.issuedAt).slice(0, 10)}</dd>
              <dt className="text-muted-foreground">Signed by</dt><dd className="font-mono text-foreground">{result.signer ? `${result.signer.keyId}${result.signer.verified ? "" : " (invalid)"}` : "unsigned"}</dd>
              <dt className="text-muted-foreground">Users</dt><dd className="text-foreground">{lic.authorizedUsers.length ? lic.authorizedUsers.join(", ") : "whole organization"}</dd>
              {lic.notes && (<><dt className="text-muted-foreground">Terms</dt><dd className="break-words text-foreground">{lic.notes}</dd></>)}
            </dl>
          )}
        </div>
      )}
    </div>
  );
}

function ExportKeyField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [show, setShow] = useState(false);
  return (
    <div>
      <label htmlFor="lic-key" className="mb-1.5 block text-[11px] text-muted-foreground">Access key</label>
      <div className="flex items-center gap-2">
        <input
          id="lic-key"
          type={show ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Set a key (min 4 characters)"
          className="h-9 w-full rounded-md border border-input bg-background px-3 font-mono text-[12.5px] text-foreground placeholder:font-sans placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
        <button
          type="button"
          onClick={() => setShow((s) => !s)}
          className="h-9 shrink-0 rounded-md border border-border bg-card px-3 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {show ? "Hide" : "Show"}
        </button>
      </div>
      {value.length > 0 && value.length < 4 && <p className="mt-1 text-[10.5px] text-warning">Key must be at least 4 characters to arm the lock.</p>}
    </div>
  );
}
