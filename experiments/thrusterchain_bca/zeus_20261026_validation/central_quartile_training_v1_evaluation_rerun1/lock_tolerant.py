"""common.atomic_json/atomic_bytes, tolerant of a momentary Windows lock on the mounted drive.

The first evaluation attempt stopped when replacing progress.json raised
PermissionError while another reader held the file. Bytes written are
identical to common's writers; only the retry is new.
"""
import json
import os
from pathlib import Path
import time

from common import digest

LOCK_ATTEMPTS = 40


def serialized(value):
    """Exactly the bytes common.atomic_json writes."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def lock_tolerant_bytes(path, value, *, attempts=LOCK_ATTEMPTS, stats=None, sleep=time.sleep):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    for attempt in range(attempts):
        try:
            with temporary.open("wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            break
        except PermissionError:
            if stats is not None:
                stats['lock_errors'] += 1
            if attempt + 1 == attempts:
                raise
            sleep(min(.05 * 2 ** attempt, 1.))
    return {"path": path.name, "bytes": len(value), "sha256": digest(value)}


def lock_tolerant_json(path, value, **options):
    return lock_tolerant_bytes(path, serialized(value), **options)
