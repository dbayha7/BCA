// Resumable random stream for training. Same algorithm and sequence as rl.ts `mulberry32`,
// but the internal state can be read back, so a saved session resumes the exact minibatch
// stream it would have drawn had the page never been closed.

export interface ResumableRng {
  (): number;
  /** Current internal state; pass it to `resumableRng` to continue the same stream. */
  state(): number;
}

export function resumableRng(seedOrState: number): ResumableRng {
  let a = seedOrState >>> 0;
  const next = (() => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }) as ResumableRng;
  next.state = () => a >>> 0;
  return next;
}
