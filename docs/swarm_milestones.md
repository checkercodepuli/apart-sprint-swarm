# Swarm extension milestones

This branch follows the Part A swarm-alignment plan and splits implementation into reviewable milestones.

## Milestone M1 — least-privilege tamper tool

Goal: enforce secure-by-default transcript access on the agent side while keeping monitor access explicit and auditable.

Status: in progress on branch `milestone/m1-tamper-tool`.

Scope:
- agent writes to own reasoning/transcript store only
- agent reads are limited to public outcomes
- monitor reads of monitored transcripts are explicit and logged. But, no agent should be able to read monitor's activity logs or it's actions. If an agent accesses a monitors' logs or a monitor allows access, monitor confirgured to montior the agent's monitor should be able to act in disabling the monitor for collusion and/or replace with a new monitor. 
- unauthorized accesses are visible in the access log
- access grants/revocations are recorded with approver metadata

## Milestone M2 — anchored evidence chain

Goal: harden tamper detection with keyed hashes and periodic anchor checkpoints written outside the editable transcript store.

Scope:
- keyed hash for evidence records
- periodic anchor writes to a private monitor-side
- Access audit records log to be kept separately. Monitors can see all agent access audit records. Monitor of monitors( or hierarchy of chains in future) should see all monitor actions, any changes on itself and it's agents being monitored  

log
- find-first-broken-record verification
- monitor-side tamper detection for post-hoc rewrite of monitor records

## Milestone M3 — monitor input switching

Goal: replay the same episode under three monitor views:
- agent transcript view
- chain view
- anchored-chain view

Scope:
- runtime switch for monitor input source
- same seed replay and per-condition comparison
- report on false gaps and hold/constraint differences

## Milestone M4 — swarm-size benchmark

Goal: validate robustness at 5, 10 and 20 sellers.

Scope:
- default market scaffolding for 5/10/20 sellers
- sample runs under hard cap budgets
- summary metrics by seed and condition
- summary of misses by monitor and metrics of monitor of monitor

## Milestone M5 — observability ablations

Goal: quantify the monitor’s reliability under sampling and gateway-only visibility.

Scope:
- add `sampled` preset
- add `egress_only` preset
- compare action timing, missing verdict rate and harm detection

## Milestone M6 — prereg + analysis + report

Goal: close the sprint with reproducible analysis and a clean write-up.

Scope:
- `docs/PREREG_SWARM.md`
- `paper/analysis/swarm_study.py`
- final metrics and appendices
- dual-use / ethics note
