"""EXPLORATORY stand-ins (plan 4.2): the user's choice of which stand-in forms carry plan 4.2's mark, recorded in
runs/shift_explore/standins/approval.json (+ .sha256) before the first launch (spec change C3).

The spec's changes F1-F3 read density, distance to data and two-head disagreement from the row's recorded command. All
eight forms are trained, matched and scored either way; the approval says which five carry the mark:
  'approve F1-F3'  the re-referenced forms (density, distance, two_head) with range_hinge and classifier_ratio;
  'decline F1-F3'  the plan's literal forms (density_plan, distance_plan, two_head_plan) with the same two.
The other three are reported. Every run phase refuses without a valid approval of the CURRENT spec and code manifest
(standins.require_approval), so a new spec or manifest voids it.

Two modes, as Stage 2's approve.py:
  (a) terminal: standard input is a terminal; the user types the first 16 characters of spec.json's SHA-256, then
      'approve F1-F3' or 'decline F1-F3'. No agent message counts.
        cd /home/user/bca && /tmp/claude-0/bcavenv/bin/python <standins>/approve.py terminal
  (b) chat: a file holding the user's chat message VERBATIM, written by the orchestrating agent only from a genuine user
      turn (this program cannot check that). The message must hold the first 16 characters of spec.json's SHA-256,
      the words 'accept chat mode' (the user accepts that chat mode is weaker than typing at a terminal) and exactly one
      of 'approve F1-F3' / 'decline F1-F3'.
        cd /home/user/bca && /tmp/claude-0/bcavenv/bin/python <standins>/approve.py chat --message-file <file>
  show: prints the spec's path, SHA-256, its 16-character prefix, the two choices and whether an approval exists.

Both modes refuse when the spec, the gate files, the room import or the code manifest do not check out
(standins.begin), when approval.json exists, or when any stand-in run output exists (the choice comes before any run).
Writes only approval.json and its sidecar. Reads no ThrusterChain data and no room-test outcome.
"""
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import standins as S  # noqa: E402

CHAT_WEAKER = ("Chat mode is weaker than terminal typing: the message reaches this program through a file written by "
               "the orchestrating agent, so the program cannot tell a genuine user turn from a forged one. It is valid "
               "only because the user accepted it in the same message ('accept chat mode').")


def _run_outputs(out):
    out = Path(out)
    return [str(p) for p in sorted(out.glob("seed*"))] + [str(out / n) for n in ("marks.json",) if (out / n).exists()]


def _common(out=None):
    spec, spec_sha, man_sha = S.begin(out)
    path = S.approval_path(out)
    if path.exists():
        raise SystemExit(f"refused: {path} exists already")
    found = _run_outputs(out or S.OUT)
    if found:
        raise SystemExit("refused: stand-in run outputs exist; the choice must be made before any run: " + ", ".join(found))
    return spec, spec_sha, man_sha, path


def _write(path, body, spec_sha, man_sha, out=None):
    S.write_json(path, body, out)
    try:
        S.require_approval(spec_sha, man_sha, out)       # the file just written must verify
    except SystemExit:
        path.unlink()
        path.with_name(path.name + ".sha256").unlink()
        raise
    return body


def _body(choice, spec_sha, man_sha, **fields):
    return dict(kind="standins-approval", label=S.LABEL, spec_path=str(S.SPEC_PATH), spec_sha256=spec_sha,
                code_manifest_sha256=man_sha, mark_forms=choice, mark_families=list(S.MARK_FORMS[choice]["mark"]),
                reported_families=list(S.MARK_FORMS[choice]["reported"]), meaning=S.MARK_FORMS[choice]["meaning"],
                approved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields)


def approve_terminal(stdin=None, input_fn=input, out=None, log=print):
    stdin = sys.stdin if stdin is None else stdin
    spec, spec_sha, man_sha, path = _common(out)
    if not (hasattr(stdin, "isatty") and stdin.isatty()):
        raise SystemExit("refused: terminal approval must be typed by the user at a terminal (standard input is not a "
                         "terminal); no agent message counts as approval")
    log(f"Stand-ins spec: {S.SPEC_PATH}\nSHA-256: {spec_sha}\nChoices: 'approve F1-F3' (re-referenced forms carry the "
        f"mark) or 'decline F1-F3' (the plan's literal forms carry the mark)")
    typed = input_fn("Type the first 16 characters of the spec's SHA-256: ").strip()
    if typed != spec_sha[:S.APPROVAL_PREFIX_LEN]:
        raise SystemExit("not approved: the typed characters do not match")
    typed_choice = input_fn("Type 'approve F1-F3' or 'decline F1-F3': ").strip()
    choice = {v: k for k, v in S.CHOICE_PHRASES.items()}.get(typed_choice)
    if choice is None:
        raise SystemExit("not approved: the choice must be exactly 'approve F1-F3' or 'decline F1-F3'")
    try:
        tty = os.ttyname(stdin.fileno())
    except (OSError, AttributeError, ValueError):
        tty = None
    body = _body(choice, spec_sha, man_sha, mode="terminal", typed=typed, typed_choice=typed_choice,
                 stdin_isatty=True, tty=tty,
                 statement="Chosen by the user at a terminal after reading the spec with this SHA-256.")
    _write(path, body, spec_sha, man_sha, out)
    log(f"approval written: {path} (mark forms: {choice})")
    return body


def approve_chat(message_file, out=None, log=print):
    spec, spec_sha, man_sha, path = _common(out)
    mf = Path(message_file)
    if not mf.exists():
        raise SystemExit(f"refused: no message file at {mf}")
    raw = mf.read_bytes()
    try:
        message = raw.decode("utf8")
    except UnicodeDecodeError:
        raise SystemExit("refused: the message file is not UTF-8 text") from None
    prefix = spec_sha[:S.APPROVAL_PREFIX_LEN]
    choice = S.choice_in_message(message)
    if prefix not in message or S.CHAT_ACCEPT_PHRASE not in message.lower() or choice is None:
        raise SystemExit(f"not approved: the message must hold the spec prefix {prefix}, the words "
                         f"'{S.CHAT_ACCEPT_PHRASE}' and exactly one of 'approve F1-F3' / 'decline F1-F3'")
    body = _body(choice, spec_sha, man_sha, mode="chat", message_verbatim=message,
                 message_sha256=hashlib.sha256(message.encode("utf8")).hexdigest(),
                 message_file=str(mf.resolve()), message_file_sha256=hashlib.sha256(raw).hexdigest(),
                 stdin_isatty=bool(hasattr(sys.stdin, "isatty") and sys.stdin.isatty()), chat_mode_caveat=CHAT_WEAKER,
                 statement=("Chosen in chat: the message above, written verbatim to the message file by the "
                            "orchestrating agent from the user's own turn, holds this spec's 16-character prefix, "
                            "accepts chat mode and names one choice."))
    _write(path, body, spec_sha, man_sha, out)
    log(f"approval written (chat mode): {path} (mark forms: {choice})")
    return body


def show(out=None, log=print):
    spec, spec_sha, man_sha = S.begin(out)
    log(f"Stand-ins spec: {S.SPEC_PATH} (version {spec.get('version')})\nSHA-256: {spec_sha}\n"
        f"First 16 characters: {spec_sha[:S.APPROVAL_PREFIX_LEN]}\nCode manifest SHA-256: {man_sha}\n"
        + "\n".join(f"'{S.CHOICE_PHRASES[k]}': {v['meaning']}" for k, v in S.MARK_FORMS.items())
        + f"\nApproval file present: {S.approval_path(out).exists()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("terminal")
    c = sub.add_parser("chat")
    c.add_argument("--message-file", required=True)
    sub.add_parser("show")
    a = ap.parse_args(argv)
    if a.mode == "terminal":
        approve_terminal()
    elif a.mode == "chat":
        approve_chat(a.message_file)
    else:
        show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
