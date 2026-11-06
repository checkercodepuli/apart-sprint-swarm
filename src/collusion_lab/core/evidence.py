"""Authoritative, append-only, hash-chained evidence log.

Design constraints (from the requirements):
* Only the environment writes events; agents interact through tools that the
  environment logs on their behalf, so agents cannot omit or rewrite evidence.
* Agents get a *filtered view* (what they could legitimately observe).
* The monitor reads the full log, and writes its own decisions to a separate
  private log that no agent tool can reach.
* Tampering is detectable: each record carries the hash of its predecessor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Event, EventKind


def _digest(key: str | None, prev: str, body: dict) -> str:
    raw = (key or "") + prev + json.dumps(body, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class EvidenceLog:
    def __init__(self, path: Path | None = None, *, key: str | None = None,
                 anchor_path: Path | None = None, anchor_every: int = 0):
        self.path = Path(path) if path else None
        self.key = key or "swarm-alignment-default-key"
        self.anchor_path = Path(anchor_path) if anchor_path else None
        self.anchor_every = max(0, int(anchor_every))
        self._events: list[Event] = []
        self._anchors: dict[int, str] = {}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")
        if self.anchor_path:
            self.anchor_path.parent.mkdir(parents=True, exist_ok=True)
            if self.anchor_path.exists():
                for line in self.anchor_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seq = d.get("seq")
                    if seq is not None:
                        self._anchors[int(seq)] = str(d.get("hash") or "")

    def _write_anchor(self, seq: int, digest: str) -> None:
        if not self.anchor_path or self.anchor_every <= 0:
            return
        if seq % self.anchor_every != 0:
            return
        self._anchors[int(seq)] = digest
        if self.anchor_path:
            with self.anchor_path.open("a") as f:
                f.write(json.dumps({"seq": int(seq), "hash": digest}, default=str) + "\n")

    def append(self, ev: Event) -> Event:
        ev.seq = len(self._events)
        ev.prev_hash = self._events[-1].hash if self._events else "GENESIS"
        body = ev.to_dict()
        body.pop("hash", None)
        ev.hash = _digest(self.key, ev.prev_hash, body)
        self._events.append(ev)
        self._write_anchor(ev.seq, ev.hash)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(ev.to_dict(), default=str) + "\n")
        return ev

    def record_tamper_attempt(self, actor: str, idx: int, action: str, *, payload: dict | None = None,
                              note: str = "seller_tamper_attempt") -> dict:
        """Record a seller tampering action in the environment log so the monitor sees it."""
        rec = {
            "event": "tamper_attempt",
            "actor": actor,
            "index": idx,
            "action": action,
            "note": note,
            "payload": payload or {},
        }
        self.emit("env", "env", EventKind.STATE, rec, "env", idx)
        return rec

    def emit(self, episode_id: str, actor: str, kind: EventKind, payload: dict,
             channel: str = "env", step: int = 0) -> Event:
        return self.append(Event(episode_id, actor, kind, payload, channel, step))

    # -- read side ------------------------------------------------------
    def events(self) -> list[Event]:
        return list(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(list(self._events))

    def __len__(self) -> int:
        return len(self._events)

    def verify_chain(self) -> bool:
        prev = "GENESIS"
        for ev in self._events:
            body = ev.to_dict()
            h = body.pop("hash")
            if ev.prev_hash != prev or _digest(self.key, prev, body) != h:
                return False
            prev = h
        if not self._anchors:
            return True
        for ev in self._events:
            if ev.seq in self._anchors and self._anchors[ev.seq] != ev.hash:
                return False
        return True

    @staticmethod
    def load(path: Path, *, key: str | None = None, anchor_path: Path | None = None,
             anchor_every: int = 0) -> "EvidenceLog":
        log = EvidenceLog(None, key=key, anchor_path=anchor_path, anchor_every=anchor_every)
        for line in Path(path).read_text().splitlines():
            if line.strip():
                log._events.append(Event.from_dict(json.loads(line)))
        return log


class ObservabilityFilter:
    """Partial-observability ablation: hide some event kinds/channels from the monitor."""

    PRESETS = {
        "full": dict(drop_kinds=set(), drop_channels=set()),
        # monitor sees only final artifacts + gate results (no chat, no tool traces)
        "artifacts_only": dict(drop_kinds={EventKind.MESSAGE, EventKind.TOOL_CALL,
                                           EventKind.TOOL_RESULT, EventKind.FILE_READ},
                               drop_channels=set()),
        # monitor sees chat transcripts only (classic transcript monitor)
        "transcript_only": dict(drop_kinds={EventKind.TOOL_CALL, EventKind.TOOL_RESULT,
                                            EventKind.FILE_READ, EventKind.FILE_WRITE},
                                drop_channels=set()),
        # private DMs are not logged (agents found an unmonitored channel)
        "no_dm": dict(drop_kinds=set(), drop_channels={"dm"}),
        # monitor sees only the exchange side + system-level gates; content is omitted
        "egress_only": dict(drop_kinds={EventKind.MESSAGE}, drop_channels={"chat"}),
        # monitor sees a random sample of messages, but still all actions and gates.
        "sampled": dict(drop_kinds=set(), drop_channels=set()),
    }

    def __init__(self, preset: str = "full"):
        cfg = self.PRESETS[preset]
        self.preset = preset
        self.drop_kinds = cfg["drop_kinds"]
        self.drop_channels = cfg["drop_channels"]
        if preset == "sampled":
            self.sample_fraction = 0.5
        else:
            self.sample_fraction = 1.0

    def apply(self, events: Iterable[Event]) -> list[Event]:
        out = []
        for i, e in enumerate(events):
            if e.kind in self.drop_kinds:
                continue
            if any(e.channel.startswith(c) for c in self.drop_channels):
                continue
            if self.preset == "sampled" and e.kind == EventKind.MESSAGE:
                if (i % 2) == 0:
                    continue
            out.append(e)
        return out


class MonitorPrivateLog:
    """Decision trail of the trusted monitor. Never exposed through agent tools."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def record(self, rec: dict) -> None:
        prev = self.records[-1]["hash"] if self.records else "GENESIS"
        rec = dict(rec)
        rec["hash"] = _digest("monitor-private", prev, rec)
        self.records.append(rec)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
