"""Login details for the portal, kept in a secure credential store and nowhere else: not in files, .env, code, logs,
git, prompts or Telegram. Two backends, chosen by `PORTFOLIO_SECRET_BACKEND` (default `keyring`):

* `keyring` (default, for a Windows PC): Windows Credential Manager, via the `keyring` package.
* `aws` (for the AWS deployment, where there is no Windows Credential Manager): one JSON secret in AWS Secrets
  Manager, encrypted at rest with a KMS key, named by `PORTFOLIO_SECRET_NAME` (default
  `ai-trading-bot/integrated-portfolio`). Needs `boto3` and an IAM role/user allowed
  `secretsmanager:GetSecretValue`, `PutSecretValue`, `CreateSecret`, `DeleteSecret` on that one secret.

(The OTP still comes through the trading bot's own Telegram bot, configured in .env, on both backends.)"""
import json
import os
import re
from dataclasses import dataclass, fields
from typing import Dict, List

SERVICE = "ai-trading-bot/integrated-portfolio"
RULES = {
    "mobile": (re.compile(r"[6-9]\d{9}"), "a 10-digit Indian mobile number"),
    "mpin": (re.compile(r"\d{4,6}"), "4 to 6 digits"),
    "customer_id": (re.compile(r"\d*"), "digits, or empty when the mobile number has one customer ID"),
}


class MissingCredentials(Exception):
    """Setup has not been run (or was only partly completed) for this backend."""


class InsecureStore(Exception):
    """The credential store would keep secrets in a plain file or nowhere at all."""


@dataclass(frozen=True, repr=False)
class Credentials:
    mobile: str
    mpin: str
    customer_id: str

    def __repr__(self) -> str:  # never let a secret reach a log line or a traceback
        return "Credentials(<hidden>)"


def problems(values: Dict[str, str]) -> List[str]:
    """What is wrong with each field, as '<field>: expected <format>'; empty when all are valid. Values are never echoed."""
    return [f"{name}: expected {hint}" for name, (pattern, hint) in RULES.items()
            if not pattern.fullmatch(values.get(name, ""))]


def backend_name() -> str:
    return os.getenv("PORTFOLIO_SECRET_BACKEND", "keyring").strip().lower()


def _backend():
    name = backend_name()
    if name == "aws":
        return _AwsSecretsBackend()
    if name == "keyring":
        return _KeyringBackend()
    raise ValueError(f"PORTFOLIO_SECRET_BACKEND must be 'keyring' or 'aws', got {name!r}")


class _KeyringBackend:
    """Windows Credential Manager (or whatever secure OS store `keyring` finds); refuses an insecure fallback."""

    def _store(self):
        import keyring

        backend = keyring.get_keyring()
        kind = f"{type(backend).__module__}.{type(backend).__name__}"
        if kind.startswith(("keyring.backends.fail", "keyrings.alt", "keyring.backends.null")):
            raise InsecureStore(f"no secure credential store available ({kind})")
        return keyring

    def save(self, values: Dict[str, str]) -> None:
        store = self._store()
        for name, value in values.items():
            store.set_password(SERVICE, name, value)

    def load(self) -> Dict[str, str]:
        store = self._store()
        return {f: v for f in values_fields() if (v := store.get_password(SERVICE, f)) is not None}

    def forget(self) -> None:
        import keyring.errors

        store = self._store()
        for f in values_fields():
            try:
                store.delete_password(SERVICE, f)
            except keyring.errors.PasswordDeleteError:
                pass


class _AwsSecretsBackend:
    """AWS Secrets Manager: one JSON secret holding all three fields, encrypted at rest by AWS-managed (or your own)
    KMS key. `boto3` picks up credentials the normal AWS way (an EC2 instance role, in production; `aws configure`,
    locally), so none of that touches this code."""

    def __init__(self):
        self.name = os.getenv("PORTFOLIO_SECRET_NAME", SERVICE)

    def _client(self):
        import boto3

        # No explicit region/credentials here: boto3 resolves them itself (the EC2 instance's role and region, or
        # `aws configure`'s profile locally), which is the standard, correct way and needs nothing from this code.
        return boto3.client("secretsmanager")

    def save(self, values: Dict[str, str]) -> None:
        client = self._client()
        payload = json.dumps({**self.load(), **values})
        try:
            client.put_secret_value(SecretId=self.name, SecretString=payload)
        except client.exceptions.ResourceNotFoundException:
            client.create_secret(Name=self.name, SecretString=payload,
                                 Description="Integrated India login for the read-only portfolio agent")

    def load(self) -> Dict[str, str]:
        client = self._client()
        try:
            body = client.get_secret_value(SecretId=self.name)["SecretString"]
        except client.exceptions.ResourceNotFoundException:
            return {}
        try:
            return {k: v for k, v in json.loads(body).items() if k in values_fields()}
        except (json.JSONDecodeError, AttributeError):
            return {}

    def forget(self) -> None:
        client = self._client()
        try:
            client.delete_secret(SecretId=self.name, ForceDeleteWithoutRecovery=True)
        except client.exceptions.ResourceNotFoundException:
            pass


def values_fields() -> List[str]:
    return [f.name for f in fields(Credentials)]


def save(creds: Credentials) -> None:
    """Validate and store every field; nothing is stored if any field is invalid."""
    values = {f: getattr(creds, f) for f in values_fields()}
    wrong = problems(values)
    if wrong:
        raise ValueError("; ".join(wrong))
    _backend().save(values)


def load() -> Credentials:
    """Read the stored credentials; raises MissingCredentials if setup has not been completed."""
    values = _backend().load()
    missing = sorted(f for f in values_fields() if f not in values)
    if missing:
        raise MissingCredentials(f"not set up: {', '.join(missing)} (run: python -m src.portfolio setup)")
    return Credentials(**values)


def forget() -> None:
    """Delete the stored credentials (missing ones are ignored)."""
    _backend().forget()
