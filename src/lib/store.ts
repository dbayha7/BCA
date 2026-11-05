// Local persistence for the lab: one IndexedDB database in this browser profile.
// Nothing here leaves the machine. Typed arrays (Q-tables) are stored as-is via
// structured clone. Every call fails soft: when IndexedDB is unavailable (private
// windows in some browsers, blocked site data) the lab keeps working unsaved and the
// header says so.

import type { Checkpoint } from "@/components/lab/checkpoint-history";
import type { ExperimentRecord } from "./experiments";
import type { ErrorEntry } from "./error-log";

const DB_NAME = "offline-learning-lab";
const DB_VERSION = 1;
export const STORES = ["kv", "experiments", "checkpoints", "uploads", "errors"] as const;
type StoreName = (typeof STORES)[number];

let dbPromise: Promise<IDBDatabase> | null = null;

export function storageSupported(): boolean {
  try {
    return typeof indexedDB !== "undefined" && indexedDB !== null;
  } catch {
    return false;
  }
}

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    if (!storageSupported()) {
      reject(new Error("IndexedDB is not available in this browser."));
      return;
    }
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, DB_VERSION);
    } catch (e) {
      reject(e instanceof Error ? e : new Error("Could not open local storage."));
      return;
    }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("kv")) db.createObjectStore("kv");
      if (!db.objectStoreNames.contains("experiments")) db.createObjectStore("experiments", { keyPath: "id" });
      if (!db.objectStoreNames.contains("checkpoints")) db.createObjectStore("checkpoints", { keyPath: "id" });
      if (!db.objectStoreNames.contains("uploads")) db.createObjectStore("uploads");
      if (!db.objectStoreNames.contains("errors")) db.createObjectStore("errors", { keyPath: "id" });
    };
    req.onsuccess = () => {
      const db = req.result;
      // Another tab upgraded the schema: drop this handle so the next call reopens.
      db.onversionchange = () => {
        db.close();
        dbPromise = null;
      };
      resolve(db);
    };
    req.onerror = () => reject(req.error ?? new Error("Could not open local storage."));
    req.onblocked = () => reject(new Error("Local storage is blocked by another open tab of the lab."));
  });
  dbPromise.catch(() => {
    dbPromise = null;
  });
  return dbPromise;
}

function run<T>(store: StoreName, mode: IDBTransactionMode, body: (s: IDBObjectStore) => IDBRequest<T> | undefined): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode);
        const req = body(t.objectStore(store));
        let result: T;
        if (req) req.onsuccess = () => (result = req.result);
        t.oncomplete = () => resolve(result);
        t.onerror = () => reject(t.error ?? new Error("Local storage write failed."));
        t.onabort = () => reject(t.error ?? new Error("Local storage transaction aborted (disk full?)."));
      }),
  );
}

export const kv = {
  get<T>(key: string): Promise<T | undefined> {
    return run<T | undefined>("kv", "readonly", (s) => s.get(key) as IDBRequest<T | undefined>);
  },
  set<T>(key: string, value: T): Promise<void> {
    return run<IDBValidKey>("kv", "readwrite", (s) => s.put(value, key)).then(() => undefined);
  },
  del(key: string): Promise<void> {
    return run<undefined>("kv", "readwrite", (s) => s.delete(key)).then(() => undefined);
  },
};

export const experimentsStore = {
  all(): Promise<ExperimentRecord[]> {
    return run<ExperimentRecord[]>("experiments", "readonly", (s) => s.getAll() as IDBRequest<ExperimentRecord[]>).then((r) => r ?? []);
  },
  get(id: string): Promise<ExperimentRecord | undefined> {
    return run<ExperimentRecord | undefined>("experiments", "readonly", (s) => s.get(id) as IDBRequest<ExperimentRecord | undefined>);
  },
  put(rec: ExperimentRecord): Promise<void> {
    return run<IDBValidKey>("experiments", "readwrite", (s) => s.put(rec)).then(() => undefined);
  },
  delete(id: string): Promise<void> {
    return run<undefined>("experiments", "readwrite", (s) => s.delete(id)).then(() => undefined);
  },
};

export const checkpointsStore = {
  all(): Promise<Checkpoint[]> {
    return run<Checkpoint[]>("checkpoints", "readonly", (s) => s.getAll() as IDBRequest<Checkpoint[]>).then((r) => (r ?? []).sort((a, b) => b.createdAt - a.createdAt));
  },
  put(cp: Checkpoint): Promise<void> {
    return run<IDBValidKey>("checkpoints", "readwrite", (s) => s.put(cp)).then(() => undefined);
  },
  delete(id: string): Promise<void> {
    return run<undefined>("checkpoints", "readwrite", (s) => s.delete(id)).then(() => undefined);
  },
};

export interface StoredUpload {
  name: string;
  text: string;
  size: number;
  savedAt: number;
}

export const uploadsStore = {
  get(key: string): Promise<StoredUpload | undefined> {
    return run<StoredUpload | undefined>("uploads", "readonly", (s) => s.get(key) as IDBRequest<StoredUpload | undefined>);
  },
  put(key: string, value: StoredUpload): Promise<void> {
    return run<IDBValidKey>("uploads", "readwrite", (s) => s.put(value, key)).then(() => undefined);
  },
};

export const errorsStore = {
  all(): Promise<ErrorEntry[]> {
    return run<ErrorEntry[]>("errors", "readonly", (s) => s.getAll() as IDBRequest<ErrorEntry[]>).then((r) => (r ?? []).sort((a, b) => b.at - a.at));
  },
  add(e: ErrorEntry): Promise<void> {
    return run<IDBValidKey>("errors", "readwrite", (s) => s.put(e)).then(() => undefined);
  },
  clear(): Promise<void> {
    return run<undefined>("errors", "readwrite", (s) => s.clear()).then(() => undefined);
  },
};

/** Removes every saved experiment, checkpoint, upload, error and setting of the lab. */
export async function clearAllLabData(): Promise<void> {
  for (const name of STORES) await run<undefined>(name, "readwrite", (s) => s.clear());
}

export interface StorageInfo {
  supported: boolean;
  usage?: number;
  quota?: number;
  persisted?: boolean;
}

export async function storageInfo(): Promise<StorageInfo> {
  const supported = storageSupported();
  const info: StorageInfo = { supported };
  try {
    if (navigator.storage?.estimate) {
      const e = await navigator.storage.estimate();
      info.usage = e.usage;
      info.quota = e.quota;
    }
    if (navigator.storage?.persisted) info.persisted = await navigator.storage.persisted();
  } catch {
    /* estimates are best-effort */
  }
  return info;
}

/** Asks the browser not to evict the lab's data under storage pressure. */
export async function requestPersistence(): Promise<boolean> {
  try {
    return (await navigator.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}
