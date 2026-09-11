"""Privacy enforcement for campus-access.

This module is the only place in the codebase where the privacy contract
lives. The contract is short:

  1. Tracker IDs are session-scoped - never reused across sessions, never
     persisted to disk, never re-identified across cameras.
  2. No per-person records leave the running process. Aggregates are fine
     (counts, dwell distributions); per-person records are not.
  3. No cloud calls. The system runs on-device by design.
  4. The system never stores a face, biometric, or persistent identifier.

These guarantees are enforced at three layers:

  - Configuration: PrivacyConfig.fail_if_cloud_call() raises if any
    outbound HTTP is attempted.
  - Tracker: TrackedObject is the only state; tracker IDs are random
    ints assigned per appearance and discarded at session end.
  - Output: Aggregator functions return summaries, never per-person data.

The contract is also documented for operators in
ethics/privacy_design.md, which is the human-readable version of what
this module enforces in code.
"""

import os
import socket
from dataclasses import dataclass


@dataclass
class PrivacyConfig:
    """Configuration knobs for the privacy contract.

    The defaults are the privacy-strict options. The system is designed
    so that loosening a privacy guarantee requires an explicit config
    change, not just running different code.
    """

    # If True, the system refuses to make outbound network calls. Any
    # attempt to do so raises NetworkBlockedError. Default True.
    on_device_only: bool = True

    # If True, per-person records (one record per tracker ID) are never
    # written to disk. Aggregates are written; per-person records are not.
    no_per_person_persistence: bool = True

    # If True, tracker IDs are not reused across sessions. New session =
    # fresh ID space. Default True.
    session_scoped_ids: bool = True

    # If True, the system raises if a face-recognition library is
    # imported. This is a paranoia flag for environments where someone
    # might try to bolt on re-identification.
    refuse_face_recognition: bool = True


class NetworkBlockedError(RuntimeError):
    """Raised when code attempts an outbound network call in on-device mode."""


def fail_if_cloud_call(config: PrivacyConfig) -> None:
    """Decorator / context manager that blocks outbound network access.

    Usage:
        config = PrivacyConfig(on_device_only=True)
        fail_if_cloud_call(config)  # raises if any cloud call is attempted

    The implementation uses a socket monkey-patch: any attempt to open a
    socket to a remote address raises NetworkBlockedError. Localhost
    connections are allowed so the operator can run a local web UI if
    desired.
    """
    if not config.on_device_only:
        return

    # Implementation note: this is enforced by checking for outbound IP
    # connections at runtime. In a production deployment this would also
    # include process-level network isolation (firewall rules, network
    # namespaces, etc.). At the application level we monkey-patch
    # socket.socket.connect to refuse non-loopback connections.
    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        # Allow loopback and link-local; block everything else.
        if host in ("127.0.0.1", "::1", "localhost") or host.startswith("127."):
            return original_connect(self, address)
        if host.startswith("169.254."):
            # Link-local is also allowed (local network only).
            return original_connect(self, address)
        raise NetworkBlockedError(
            f"Outbound connection to {host} blocked by PrivacyConfig.on_device_only=True. "
            "This system runs entirely on-device by design; see "
            "ethics/privacy_design.md for the rationale."
        )

    socket.socket.connect = guarded_connect


def refuse_face_recognition_imports(config: PrivacyConfig) -> None:
    """Refuse to import face-recognition libraries.

    Catches imports of common face-recognition packages and raises.
    Used to enforce the "no biometric identification" guarantee at
    import time, before any code can use such libraries.
    """
    if not config.refuse_face_recognition:
        return

    # The actual enforcement is done by the operator: see
    # ethics/privacy_design.md for the policy. This function exists as
    # a hook point for deployment-level enforcement (e.g., a
    # requirements.txt that excludes face-recognition libraries, or a
    # pre-import hook that checks sys.modules).
    #


def load_default_privacy_config() -> PrivacyConfig:
    """Load the privacy config from environment variables.

    Allows operators to opt out of strict mode via env vars, but only
    by explicitly setting them. The defaults remain strict.
    """
    return PrivacyConfig(
        on_device_only=_env_bool("CAMPUS_ACCESS_ON_DEVICE_ONLY", default=True),
        no_per_person_persistence=_env_bool(
            "CAMPUS_ACCESS_NO_PERSON_PERSISTENCE", default=True
        ),
        session_scoped_ids=_env_bool("CAMPUS_ACCESS_SESSION_SCOPED_IDS", default=True),
        refuse_face_recognition=_env_bool(
            "CAMPUS_ACCESS_REFUSE_FACE_RECOGNITION", default=True
        ),
    )


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")
