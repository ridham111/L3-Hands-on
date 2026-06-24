"""TLS bootstrap: verify HTTPS against the OS trust store.

Corporate TLS-inspection proxies re-sign traffic with an internal root CA that
the OS trusts but Python's bundled certifi does not, causing
`CERTIFICATE_VERIFY_FAILED: self-signed certificate in chain`. truststore makes
Python trust the same roots the machine does. Opt out with
ONBOARDING_DISABLE_TRUSTSTORE=1.
"""
from __future__ import annotations

import os


def _inject() -> None:
    if os.getenv("ONBOARDING_DISABLE_TRUSTSTORE", "").strip() in ("1", "true", "True"):
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


_inject()
