# Local and cluster ownership, September 29

This is an execution allocation within the frozen 20-pair OOD protocol. It adds
no seed, training update, candidate, horizon, tolerance, or resource allowance.

- Local owns TD3+BC Hopper/Walker seeds 202609171–175. The currently immutable
  worker is hardcoded to TD3+BC Hopper 202609171 and its existing ledger stays
  intact. Future local workers must enforce this TD3-only ownership and reuse
  that ledger; its old broad header is not permission to run ReBRAC locally.
- Cluster owns ReBRAC Hopper/Walker seeds 202609171–175. This dispatch executes
  only accepted ReBRAC Hopper 202609171, under an exclusive cluster OOD lease.
  Its ledger permits only that pair plus its existing engineering allowance.
- Each host allocation is at most 18,395,680 environment calls: 17,920,000
  outcomes, 384,000 collection, 71,680 repeats and 20,000 engineering. Their sum
  is the existing 36,791,360-call v2 allowance. Including the closed ancestor's
  1,298,353 calls gives 38,089,713, below the original 39,998,400 ceiling.
- The current single cluster pair's maximum is 1,847,568 calls, including up to
  10,000 engineering calls; its other pair caps are zero. Further dispatches must
  debit the remaining ReBRAC allocation and preserve all prior reservations.

Every cluster simulator call, including the constructor, is durably reserved
before physics. Uncertain calls remain charged. A remote ledger does not create
an additional allocation. No outcome is repeated following a restart.

The cluster does not have live access to the local ancestor ledger. It uses an
explicitly named, hash-pinned attestation taken immediately before dispatch;
the active local worker continues its original live ancestor checks. This is
not represented as remote live filesystem verification. A later alteration of
the closed ancestor invalidates combined accounting and requires stopping and
reviewing the campaign. The old ledger remains read-only on both execution paths.

Cluster work uses a separate scheduled CPU allocation, not a login-node worker
or a third GPU. The existing GPU training job continues unchanged. Successful
query and engineering checks are not OOD results; independent saved-outcome
verification remains necessary.
