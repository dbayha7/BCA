"""Stdlib-only exact content identity, separate from MessagePack wire ordering."""
import hashlib
import json


def content_identity(actual, expected):
    """Compare full sorted path/shape/dtype/raw-leaf-hash descriptors exactly."""
    if actual != expected:
        raise ValueError("Restored state content differs from committed numeric leaf descriptors")
    encoded = json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def count(value):
        if isinstance(value, dict):
            if set(value) == {"dtype", "shape", "sha256"}:
                return 1
            return sum(count(child) for child in value.values())
        if isinstance(value, list):
            return sum(count(child) for child in value)
        return 0

    return {"canonical_state_leaves_sha256": hashlib.sha256(encoded).hexdigest(),
            "numeric_leaves": count(actual), "exact_content_match": True}


def mapping_order_audit(original, restored):
    """Record all map-order differences without changing either data tree."""
    different = []

    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                raise ValueError("Decoded state map key sets differ at " + path)
            if list(a) != list(b):
                different.append({"path": path, "original_order": list(a), "restored_order": list(b)})
            for key in sorted(a):
                visit(a[key], b[key], path + "/" + str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                raise ValueError("Decoded sequence lengths differ at " + path)
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "/" + str(index))

    visit(original, restored, "state")
    return {"changed_mapping_order_count": len(different), "changed_mapping_orders": different,
            "scope": "Order observation only; content equality is checked independently and exactly"}
