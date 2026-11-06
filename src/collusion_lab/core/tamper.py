from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AccessRecord:
    actor: str
    action: str
    resource: str
    target: str | None
    allowed: bool
    reason: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "action": self.action,
            "resource": self.resource,
            "target": self.target,
            "allowed": self.allowed,
            "reason": self.reason,
            "ts": self.ts,
        }


class TamperTool:
    """Least-privilege access-control guard for agent transcripts and outcomes.

    Design goals for the Part A tamper experiment:
    * Agents can only write their own reasoning/transcript records.
    * Agents cannot read other agents' transcripts, inputs or hidden state.
    * Agents can read only the outcome store for public results.
    * Monitors are the only principals with transcript read access and audit-log access.
    * Every read/write is logged with actor, resource, action, timestamp and allow/deny.
    * Unauthorized read attempts become visible to the monitor as a tamper-and-access anomaly.
    """

    DEFAULT_POLICY = {
        "agent": {
            "read": {"outcome"},
            "write": {"transcript", "reasoning"},
            "deny": {"input", "transcript", "reasoning_other", "audit_log"},
        },
        "monitor": {
            "read": {"input", "transcript", "outcome", "reasoning", "audit_log"},
            "write": {"monitor_private"},
            "deny": set(),
        },
        "env": {
            "read": {"input", "transcript", "outcome", "reasoning", "audit_log"},
            "write": {"input", "outcome", "transcript", "reasoning", "audit_log"},
            "deny": set(),
        },
    }

    def __init__(self, *, actor_roles: dict[str, str] | None = None):
        self.actor_roles = actor_roles or {}
        self.store: dict[str, Any] = defaultdict(dict)
        self.audit: list[AccessRecord] = []

    def _role_for(self, actor: str) -> str:
        return self.actor_roles.get(actor, "agent")

    def _resource_kind(self, resource: str) -> str:
        return resource.split(":", 1)[0] if ":" in resource else resource

    def _owner(self, resource: str) -> str | None:
        if ":" not in resource:
            return None
        kind, owner = resource.split(":", 1)
        if kind in {"transcript", "reasoning", "outcome"}:
            return owner
        return None

    def _authorizes(self, actor: str, action: str, resource: str) -> tuple[bool, str]:
        kind = self._resource_kind(resource)
        role = self._role_for(actor)
        policy = self.DEFAULT_POLICY.get(role, self.DEFAULT_POLICY["agent"])
        allowed = kind in policy.get(action, set())

        if role == "agent":
            owner = self._owner(resource)
            if action == "write":
                if kind in {"transcript", "reasoning"} and owner in {None, actor}:
                    return True, "own transcript record"
                return False, "agents write only their own transcript/reasoning"
            if action == "read":
                if kind == "outcome":
                    return True, "agents can read only public outcomes"
                return False, "agents cannot read inputs, transcripts or other agent state"

        if role == "monitor":
            if kind in policy.get(action, set()):
                return True, "monitor has explicit read access"
            return False, "monitor access is not granted for this resource"

        if role == "env":
            return True, "environment is authoritative"

        return False, "role has no access"

    def _log(self, actor: str, action: str, resource: str, target: str | None, allowed: bool, reason: str) -> AccessRecord:
        rec = AccessRecord(actor=actor, action=action, resource=resource, target=target,
                          allowed=allowed, reason=reason, ts=time.time())
        self.audit.append(rec)
        return rec

    def read(self, actor: str, resource: str, *, target: str | None = None, default: Any = None) -> Any:
        allowed, reason = self._authorizes(actor, "read", resource)
        rec = self._log(actor, "read", resource, target, allowed, reason)
        if not allowed:
            return default
        key = resource if resource in self.store else (resource if resource.endswith(":") else resource)
        value = self.store.get(resource, default)
        if value is None and target is not None:
            value = self.store.get(target, default)
        return value

    def write(self, actor: str, resource: str, value: Any, *, target: str | None = None) -> bool:
        allowed, reason = self._authorizes(actor, "write", resource)
        self._log(actor, "write", resource, target, allowed, reason)
        if not allowed:
            return False
        self.store[resource] = value
        if target is not None:
            self.store[target] = value
        return True

    def monitor_access(self, actor: str, resource: str, *, target: str | None = None, default: Any = None) -> Any:
        if self._role_for(actor) != "monitor":
            return default
        allowed, reason = self._authorizes(actor, "read", resource)
        self._log(actor, "read", resource, target, allowed, reason)
        if not allowed:
            return default
        return self.store.get(resource, default)

    def unauthorized_reads(self, resource: str | None = None) -> list[AccessRecord]:
        out: list[AccessRecord] = []
        for rec in self.audit:
            if rec.action != "read" or rec.allowed:
                continue
            if resource is not None and rec.resource != resource:
                continue
            out.append(rec)
        return out
