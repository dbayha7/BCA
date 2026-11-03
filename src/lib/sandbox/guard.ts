// Static pre-check for user-written reward / agent code (layer 1 of 2).
//
// This is a LEXICAL filter, not a parser and not a security boundary on its own. It rejects
// oversized code and code that spells clearly dangerous names (module loading, runtime code
// compilation, constructor/prototype access, the worker global, network and storage APIs),
// and it rejects escape sequences that could spell those names indirectly. Code that builds
// names at runtime (e.g. string concatenation used as a computed property) is NOT caught here;
// that is what layer 2 is for: the code only ever runs inside a dedicated Web Worker that has
// removed the network/storage globals and disabled runtime compilation (see harden.ts).
// Taken together this is defense in depth for a single-user, browser-only lab, not a hardened
// multi-tenant sandbox.

export const MAX_CODE_BYTES = 20 * 1024;

export interface GuardIssue {
  /** 1-based line of the offending token. */
  line: number;
  /** 1-based column of the offending token. */
  column: number;
  message: string;
  token?: string;
}

export type GuardResult = { ok: true } | ({ ok: false } & GuardIssue);

interface Rule {
  tokens: string[];
  why: string;
}

const RULES: Rule[] = [
  { tokens: ["import", "require", "importScripts"], why: "loading modules or scripts is not allowed in the sandbox" },
  { tokens: ["eval", "Function", "WebAssembly"], why: "compiling code at runtime is not allowed in the sandbox" },
  { tokens: ["setTimeout", "setInterval"], why: "timers are unavailable in the sandbox (string arguments would run as code)" },
  { tokens: ["constructor", "__proto__", "prototype"], why: "constructor/prototype access is blocked (it can reach built-ins and the Function constructor)" },
  { tokens: ["globalThis", "self", "postMessage", "onmessage"], why: "the worker's global scope and message channel are off limits" },
  {
    tokens: ["fetch", "XMLHttpRequest", "WebSocket", "EventSource", "BroadcastChannel", "Worker", "SharedWorker", "navigator", "sendBeacon", "WebTransport"],
    why: "network and messaging APIs are blocked in the sandbox",
  },
  { tokens: ["indexedDB", "caches", "localStorage", "sessionStorage"], why: "storage APIs are blocked in the sandbox" },
];

const BLOCKED = new Map<string, string>();
for (const r of RULES) for (const t of r.tokens) BLOCKED.set(t, r.why);

/** Every name the static guard rejects (exported for docs/tests). */
export const BLOCKED_TOKENS: readonly string[] = [...BLOCKED.keys()];

// Keywords after which a "/" starts a regular-expression literal rather than a division.
const REGEX_AFTER_WORD = new Set(["return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw", "case", "do", "else", "yield", "await"]);
// Keywords whose "( … )" is followed by a statement, so a "/" after the ")" starts a regex.
const PAREN_STATEMENT = new Set(["if", "while", "for", "with"]);

// Escapes allowed inside strings/templates. Everything else is rejected: \u and \x can spell any
// character, and "identity" escapes (\c → "c") or line continuations can split a blocked name
// so that "con\structor" evaluates to "constructor".
const SAFE_ESCAPES = new Set(["n", "t", "r", "b", "f", "v", "0", "'", '"', "`", "\\", "$"]);

const isIdStart = (c: string) => /[A-Za-z_$]/.test(c) || c.charCodeAt(0) > 127;
const isIdPart = (c: string) => /[A-Za-z0-9_$]/.test(c) || c.charCodeAt(0) > 127;

export interface LexResult {
  /** The source with comments blanked out (newlines kept), so indices and lines still line up. */
  masked: string;
  /** Index of a backslash found outside any string/template/regex (an identifier escape or garbage). */
  strayBackslash: number | null;
  /** First escape sequence in a string/template that is not in the safe list. */
  badEscape: { index: number; seq: string } | null;
  /** First structural problem (unbalanced bracket, unterminated literal), as a syntax-error hint. */
  structure: { index: number; message: string } | null;
}

/**
 * Tiny JavaScript lexer: strips comments and tracks strings, template literals (with nested
 * `${…}` expressions), regex literals and bracket balance. Strings, templates and regex bodies are
 * KEPT in the masked output so they are scanned too (a computed member like obj["constructor"]
 * must be caught). When the regex-vs-division heuristic is unsure it errs toward "regex", which
 * only ever causes MORE text to be scanned; the residual blind spot is a contrived regex literal
 * that the heuristic reads as division and that contains "//" — layer 2 still applies there.
 */
export function lexSource(src: string): LexResult {
  const out = src.split("");
  const n = src.length;
  const stack: { ch: string; index: number; kw?: string }[] = [];
  let structure: LexResult["structure"] = null;
  let strayBackslash: number | null = null;
  let badEscape: LexResult["badEscape"] = null;
  const checkEscape = (j: number) => {
    const e = src[j + 1] ?? "";
    if (!badEscape && !SAFE_ESCAPES.has(e)) badEscape = { index: j, seq: e === "\n" ? "\\⏎" : `\\${e}` };
  };
  const fail = (index: number, message: string) => {
    if (!structure) structure = { index, message };
  };
  let regexOK = true;
  let lastWord = "";
  let i = 0;

  // Reads template text from j until the closing backtick (returns index after it) or "${" (pushes and returns after it).
  const readTemplateChunk = (j: number, start: number): number => {
    while (j < n) {
      const ch = src[j];
      if (ch === "\\") {
        checkEscape(j);
        j += 2;
        continue;
      }
      if (ch === "`") return j + 1;
      if (ch === "$" && src[j + 1] === "{") {
        stack.push({ ch: "${", index: j });
        regexOK = true;
        return j + 2;
      }
      j++;
    }
    fail(start, "unterminated template literal");
    return n;
  };

  while (i < n) {
    const c = src[i];
    const d = src[i + 1];
    if (c === "/" && d === "/") {
      while (i < n && src[i] !== "\n") out[i++] = " ";
      continue;
    }
    if (c === "/" && d === "*") {
      const end = src.indexOf("*/", i + 2);
      if (end === -1) fail(i, "unterminated /* comment */");
      const stop = end === -1 ? n : end + 2;
      for (; i < stop; i++) if (src[i] !== "\n") out[i] = " ";
      continue;
    }
    if (c === '"' || c === "'") {
      const start = i++;
      while (i < n && src[i] !== c && src[i] !== "\n") {
        if (src[i] === "\\") {
          checkEscape(i);
          i += 2;
        } else i++;
      }
      if (i >= n || src[i] === "\n") fail(start, "unterminated string");
      else i++;
      regexOK = false;
      lastWord = "";
      continue;
    }
    if (c === "`") {
      const before = stack.length;
      i = readTemplateChunk(i + 1, i);
      if (stack.length === before) regexOK = false;
      lastWord = "";
      continue;
    }
    if (c === "/") {
      if (regexOK) {
        let j = i + 1;
        let inClass = false;
        let closed = false;
        while (j < n && src[j] !== "\n") {
          const ch = src[j];
          if (ch === "\\") {
            j += 2;
            continue;
          }
          if (ch === "[") inClass = true;
          else if (ch === "]") inClass = false;
          else if (ch === "/" && !inClass) {
            closed = true;
            break;
          }
          j++;
        }
        if (closed) {
          j++;
          while (j < n && isIdPart(src[j])) j++;
          i = j;
          regexOK = false;
          lastWord = "";
          continue;
        }
        fail(i, "unterminated regular expression");
      }
      i++;
      regexOK = true;
      lastWord = "";
      continue;
    }
    if (c === "\\") {
      if (strayBackslash === null) strayBackslash = i;
      i++;
      continue;
    }
    if (isIdStart(c)) {
      let j = i + 1;
      while (j < n && isIdPart(src[j])) j++;
      const word = src.slice(i, j);
      regexOK = REGEX_AFTER_WORD.has(word);
      lastWord = word;
      i = j;
      continue;
    }
    if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(d ?? ""))) {
      let j = i + 1;
      while (j < n && /[0-9A-Za-z_.]/.test(src[j])) j++;
      i = j;
      regexOK = false;
      lastWord = "";
      continue;
    }
    if (c === "(" || c === "[" || c === "{") {
      stack.push({ ch: c, index: i, kw: c === "(" ? lastWord : undefined });
      regexOK = true;
      lastWord = "";
      i++;
      continue;
    }
    if (c === ")" || c === "]" || c === "}") {
      const top = stack.pop();
      const want = c === ")" ? "(" : c === "]" ? "[" : "{";
      if (c === "}" && top?.ch === "${") {
        // closes a template expression: continue reading the template text
        const before = stack.length;
        i = readTemplateChunk(i + 1, top.index);
        if (stack.length === before) regexOK = false;
        lastWord = "";
        continue;
      }
      if (!top) fail(i, `unexpected '${c}' with no matching '${want}'`);
      else if (top.ch !== want) {
        // Usually the opener is the culprit (a missing ')' or ']'), so point at it.
        fail(top.index, `'${top.ch}' is never closed (found '${c}' on line ${positionOf(src, i).line} instead)`);
      }
      regexOK = c === "}" ? true : c === ")" ? PAREN_STATEMENT.has(top?.kw ?? "") : false;
      lastWord = "";
      i++;
      continue;
    }
    if (!/\s/.test(c)) {
      // operators and punctuation: a regex may follow
      regexOK = true;
      lastWord = "";
    }
    i++;
  }
  if (stack.length && !structure) {
    const top = stack[stack.length - 1];
    fail(top.index, top.ch === "${" ? "unterminated template expression '${'" : `'${top.ch}' is never closed`);
  }
  return { masked: out.join(""), strayBackslash, badEscape, structure };
}

/** 1-based line/column of a character index. */
export function positionOf(src: string, index: number): { line: number; column: number } {
  let line = 1;
  let lineStart = 0;
  const end = Math.min(index, src.length);
  for (let k = 0; k < end; k++) {
    if (src.charCodeAt(k) === 10) {
      line++;
      lineStart = k + 1;
    }
  }
  return { line, column: end - lineStart + 1 };
}

const issue = (src: string, index: number, message: string, token?: string): GuardResult => {
  const { line, column } = positionOf(src, index);
  return { ok: false, line, column, message: `Line ${line}: ${message}`, token };
};

export interface GuardOptions {
  /** Name of the function the code must define, e.g. "reward" or "act". */
  requireFunction?: string;
  /** Size limit in UTF-8 bytes (default 20 KB). */
  maxBytes?: number;
}

/** Layer-1 static check. Returns the FIRST problem (by position) with a readable message and line. */
export function checkUserCode(code: unknown, opts: GuardOptions = {}): GuardResult {
  if (typeof code !== "string") return { ok: false, line: 1, column: 1, message: "Line 1: code must be text." };
  if (!code.trim()) return { ok: false, line: 1, column: 1, message: "Line 1: the editor is empty." };

  const maxBytes = opts.maxBytes ?? MAX_CODE_BYTES;
  const enc = new TextEncoder();
  const bytes = enc.encode(code).length;
  if (bytes > maxBytes) {
    // locate the character where the limit is crossed
    let acc = 0;
    let at = code.length;
    for (let k = 0; k < code.length; k++) {
      const cp = code.codePointAt(k) ?? 0;
      acc += cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
      if (cp >= 0x10000) k++;
      if (acc > maxBytes) {
        at = k;
        break;
      }
    }
    return issue(code, at, `code is ${(bytes / 1024).toFixed(1)} KB; the limit is ${(maxBytes / 1024).toFixed(0)} KB.`);
  }

  const lex = lexSource(code);
  const { masked } = lex;
  const found: { index: number; result: GuardResult }[] = [];

  if (lex.strayBackslash !== null) {
    found.push({ index: lex.strayBackslash, result: issue(code, lex.strayBackslash, "a backslash outside a string is not allowed (identifier escapes can spell blocked names).", "\\") });
  }
  if (lex.badEscape) {
    const { index, seq } = lex.badEscape;
    found.push({ index, result: issue(code, index, `escape sequence '${seq}' is not allowed in strings (only \\n \\t \\r \\b \\f \\v \\0 \\\\ and escaped quotes; others can spell blocked names indirectly).`, seq) });
  }
  let m: RegExpExecArray | null;
  const word = /[A-Za-z_$][A-Za-z0-9_$]*/g;
  while ((m = word.exec(masked))) {
    const prev = m.index > 0 ? masked[m.index - 1] : "";
    if (prev && isIdPart(prev)) continue; // part of a longer identifier
    const why = BLOCKED.get(m[0]);
    if (why) {
      found.push({ index: m.index, result: issue(code, m.index, `\`${m[0]}\` is not allowed — ${why}.`, m[0]) });
      break;
    }
  }
  if (found.length) {
    found.sort((a, b) => a.index - b.index);
    return found[0].result;
  }

  if (opts.requireFunction) {
    const name = opts.requireFunction;
    const decl = new RegExp(`(^|[^A-Za-z0-9_$])(function\\s+${name}\\s*\\(|(const|let|var)\\s+${name}\\s*=)`);
    if (!decl.test(masked)) {
      return { ok: false, line: 1, column: 1, message: `Line 1: define a function named \`${name}\`, e.g. \`function ${name}(…) { … }\`.` };
    }
  }
  return { ok: true };
}

/**
 * Best-effort location for a SyntaxError reported by the engine (V8's `new Function` errors carry
 * no position). Only consulted AFTER the engine has confirmed a syntax error, so heuristic misses
 * can never block valid code.
 */
export function locateSyntaxError(code: string): { line: number; column: number; hint: string } | null {
  const { structure } = lexSource(code);
  if (!structure) return null;
  const { line, column } = positionOf(code, structure.index);
  return { line, column, hint: structure.message };
}
