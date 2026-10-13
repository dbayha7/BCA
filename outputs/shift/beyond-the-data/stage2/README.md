# Stage 2: approve the pre-registration from your own computer

This folder holds the Stage 2 pre-registration (the flags claim, with BCA) exactly as the cloud session will run it.
It also holds a small script, `approve_local.py`, that lets you approve it at a terminal on your own computer. This is
the only way of approving that this version accepts.

**The fingerprint you are approving** (the SHA-256 of `prereg.json`, version 6):

    9b38c7c48a303cfd67e5db6f137e3a162997bb533d86b3f7647e0467b91ca3b1

Its first 16 characters are `9b38c7c48a303cfd`. The script checks the files and then prints the fingerprint itself. If
it prints a different one, stop and tell Claude.

## Your steps

1. Get the branch:

       git fetch origin && git checkout claude/main-bca-connection-x4o46s && git pull

2. Read `PREREGISTRATION.md` in this folder. Section 9 explains this approval. Section 14 lists the defaults you
   accept by approving. Section 18 says what changed in version 6.
3. From the root of the repository, in a terminal, run:

       python3 outputs/shift/beyond-the-data/stage2/approve_local.py

   On Windows: `py outputs\shift\beyond-the-data\stage2\approve_local.py`. Any Python 3.8 or later works, and nothing
   needs installing. To look without approving, add `--check`.
4. Compare the fingerprint it prints with the one above. Then type its first 16 characters and press Enter.
5. Commit and push the two new files. The script prints these commands:

       git add outputs/shift/beyond-the-data/stage2/approval_9b38c7c48a303cfd.json outputs/shift/beyond-the-data/stage2/approval_9b38c7c48a303cfd.json.sha256
       git commit -m "Stage 2: approve pre-registration 9b38c7c48a303cfd (local terminal)"
       git push origin claude/main-bca-connection-x4o46s

   After the push, do not delete, edit or re-add these files.
6. Tell Claude in the chat that you have pushed.
7. When Claude says the import is done, open `import_9b38c7c48a303cfd.json` in this folder on GitHub. Check that the commit it
   names is the one you pushed in step 5.

## What happens next

The cloud session pulls your commit and runs `approve.py import`. Import reads git itself: the one commit that added
your file, its author and committer, and the file as stored in that commit. It refuses unless all of these hold:
- the file approves exactly this fingerprint, with the right 16 characters;
- it is unchanged since your commit;
- exactly one commit on the branch added it;
- neither the commit's author nor its committer is the cloud session's own git identity;
- the files in your commit are the ones that will run;
- nothing pinned has changed since.

Import then writes its record, `import_9b38c7c48a303cfd.json`, into this folder, and Claude commits and pushes it. **No BCA run
starts until that record is on GitHub.** Every BCA run checks again, with git, that your commit is still on the branch
with your file unchanged.

## What the approval file records

- the fingerprint, and the 16 characters you typed;
- the mode, `local-terminal`, and the time in UTC;
- your computer's name and your user name on it;
- the git commit you were on;
- a statement that the approval covers this exact pre-registration only.

The file is committed to the repository, so these details become part of its history.

## If the script refuses

It writes nothing. The usual reasons:
- **"No pre-registration at ..."**: run it from the root of the repository, after `git pull`.
- **"does not match"**: a file differs from its pinned copy. Pull again, and do not edit the files here. If the message
  mentions Windows line endings, your git converted them: check `git config core.autocrlf` and check the branch out
  again.
- **"Standard input is not a terminal"**: run it yourself in a terminal window, with nothing piped into it.
- **"already exists"**: you have already approved this fingerprint. It is never overwritten.

## If Claude sends you a new fingerprint

This happens if something pinned changes before your approval is imported. Pull the branch again, run the script
again, and commit and push the new file it writes. Each fingerprint has its own approval file. Leave any earlier
`approval_....json` where it is; it approves only the earlier version and does not block the new one.

## If you change your mind

Before you push, delete the two files. Claude imports your approval only after you say you have pushed, so if you have
pushed and changed your mind, say so instead.

## What this cannot prove

- The script checks only that its input comes from a terminal, not who is typing.
- Git's author and committer names are whatever the person making the commit has set, and git does not check them.
- Anyone able to push to the repository could make a commit under your name, and that includes an agent with push
  access.
- The cloud session's runs-side approval file is not committed, so an agent could write one by hand. The checks make
  such a forgery visible on GitHub, but they cannot prevent it.

So this approval rests on the orchestrating agent following the procedure, and on your check in step 7. Its strength
is that the approval is typed on your machine, arrives through your push, and the import record that names your
commit is on GitHub. A signed commit shows as verified. Section 9 of `PREREGISTRATION.md` has the full statement.

## Files here

| File | What it is |
|---|---|
| `prereg.json`, `prereg.json.sha256` | the pre-registration you approve, and its fingerprint; byte-identical to the cloud session's copy |
| `PREREGISTRATION.md` | the same pre-registration in plain language |
| `approve_local.py` | the approval script |
| `code_manifest.json`, `code_manifest.json.sha256` | the fingerprint of every Stage 2 script; `prereg.json` pins it |
| `code/` | a copy of every pinned script, as it will run |
| `README.md` | this page; it is not pinned, so it can carry the fingerprint |
| `approval_<16 characters>.json` (+ `.sha256`) | your approval, once you have made it; one per fingerprint |
| `import_<16 characters>.json` (+ `.sha256`) | the cloud session's record of the import, naming your commit |

Some files mention paths such as `/home/user/bca/...` or `/tmp/claude-0/...`. Those are where the files live in the
cloud session. They are records only; nothing on your computer needs them.
