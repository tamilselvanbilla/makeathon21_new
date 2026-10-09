"""The only module permitted to make network calls.

Everything that leaves the device passes through `OnlineGateway.lookup`, which
accepts plain text queries only (never audio buffers), refuses anything the
privacy policy does not classify as a real-time factual lookup, and only talks
to hosts on `ALLOWED_HOSTS`. All reasoning about the result happens locally.
"""

from dataclasses import dataclass
from urllib.parse import urlparse

from .brain.policy import is_allowed_cloud_lookup
from .telemetry import log_event

# Providers (weather, news, search) are added here together with their host.
ALLOWED_HOSTS: frozenset[str] = frozenset()


class LookupUnavailable(Exception):
    """Raised when a lookup cannot or must not be performed."""


@dataclass(frozen=True)
class LookupResult:
    source: str
    text: str


def check_host_allowed(url: str) -> None:
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise LookupUnavailable(f"Host '{host}' is not on the online allowlist.")


class OnlineGateway:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def lookup(self, query: str) -> LookupResult:
        if not isinstance(query, str):
            raise TypeError("Only text queries may leave the device.")
        if not self.enabled:
            raise LookupUnavailable("Online lookups are switched off.")
        if not is_allowed_cloud_lookup(query):
            raise LookupUnavailable("This request is not a permitted factual lookup.")
        log_event("online_lookup_refused", reason="no_provider")
        raise LookupUnavailable("No online lookup provider is configured yet.")
