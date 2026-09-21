"""Network hardening shared by every external client."""

DEFAULT_TIMEOUT = 30.0


def apply_timeout(client, seconds: float = DEFAULT_TIMEOUT):
    """Give a client that has no timeout support (alpaca-py's requests session sends none, so a stalled connection would
    hang the bot forever) a default per-request timeout. Returns the client. Explicit per-call timeouts still win."""
    session = getattr(client, "_session", None)
    if session is not None and not getattr(session, "_timeout_applied", False):
        original = session.request

        def request(method, url, **kwargs):
            """Send the request with the default timeout unless one was given."""
            kwargs.setdefault("timeout", seconds)
            return original(method, url, **kwargs)

        session.request = request
        session._timeout_applied = True
    return client
