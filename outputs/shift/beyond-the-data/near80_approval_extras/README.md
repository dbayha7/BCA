# Files for checking the near-80 approval on a fresh clone

`runs/` is not in git, but the near-80 code manifest pins 8 files under `runs/shift_explore/room/code/` (the room
check's code, made by the earlier ThrusterChain checks). A machine where version 7's restore ran already has them.
On any other machine (for example a laptop with a fresh clone at `/home/user/bca`), copy them into place first:

    cp -r /home/user/bca/outputs/shift/beyond-the-data/near80_approval_extras/runs /home/user/bca/

They are byte-identical copies; the approval script checks each one against its pinned SHA-256, so a wrong copy is
refused. This folder is not part of any pre-registration and changes nothing that runs.
