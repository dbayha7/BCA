// useState that survives reloads: loads once from the lab's local key-value store, then saves
// (throttled) whenever the value changes. Falls back to plain in-memory state when storage
// is unavailable.

import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { kv } from "./store";
import { recordError } from "./error-log";

export function usePersistentState<T>(key: string, initial: T, delayMs = 400): [T, Dispatch<SetStateAction<T>>, boolean] {
  const [value, setValue] = useState<T>(initial);
  const [loaded, setLoaded] = useState(false);
  const dirty = useRef(false);

  useEffect(() => {
    let alive = true;
    kv.get<T>(key)
      .then((saved) => {
        // Keep anything the user changed while the load was in flight.
        if (alive && saved !== undefined && !dirty.current) setValue(saved);
      })
      .catch(() => undefined)
      .finally(() => alive && setLoaded(true));
    return () => {
      alive = false;
    };
  }, [key]);

  useEffect(() => {
    if (!loaded) return;
    const id = window.setTimeout(() => {
      kv.set(key, value).catch((e) => recordError("storage", e, `saving ${key}`));
    }, delayMs);
    return () => window.clearTimeout(id);
  }, [key, value, loaded, delayMs]);

  const set: Dispatch<SetStateAction<T>> = (v) => {
    dirty.current = true;
    setValue(v);
  };
  return [value, set, loaded];
}
