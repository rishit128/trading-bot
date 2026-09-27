"""python -m src.portfolio <command>

  setup      store the login details in Windows Credential Manager (asked for here, never saved anywhere else)
  check      show what is set up, without revealing it
  forget     delete the stored login details (--profile also deletes the saved browser profile)

To run the agent: `python main.py --portfolio` (OTP on Telegram), or send /portfolio to the running trading bot.
"""
import argparse
import getpass
import shutil
import sys
from typing import Optional

from src.portfolio import vault
from src.portfolio.runs import RunStore


def _ask(prompt: str, secret: bool, current: str = "") -> str:
    keep = " (Enter keeps the saved one)" if current else ""
    value = (getpass.getpass if secret else input)(f"{prompt}{keep}: ").strip()
    return value or current


def setup(store: RunStore) -> int:
    try:
        old = vault.load()
    except vault.MissingCredentials:
        old = None
    print("Stored in Windows Credential Manager only. Nothing you type here is shown, logged or written to a file.")
    values = {
        "mobile": _ask("Mobile number registered with Integrated", True, old.mobile if old else ""),
        "mpin": _ask("MPIN", True, old.mpin if old else ""),
        "customer_id": _ask("Customer ID (only if several are linked to the mobile; else Enter)", False,
                            old.customer_id if old else ""),
    }
    wrong = vault.problems(values)
    if wrong:
        print("Not saved:\n  " + "\n  ".join(wrong))
        return 1
    vault.save(vault.Credentials(**values))
    if store.clear_mpin_block():
        print("The earlier MPIN failure is cleared.")
    print("Saved. Next: python main.py --portfolio --show  (or /portfolio to the running trading bot on Telegram)")
    return 0


def check(store: RunStore) -> int:
    try:
        creds = vault.load()
    except (vault.MissingCredentials, vault.InsecureStore) as e:
        print(e)
        return 1
    import keyring

    print(f"credential store  {type(keyring.get_keyring()).__name__}")
    print(f"mobile            ******{creds.mobile[-4:]}")
    print(f"MPIN              {len(creds.mpin)} digits")
    print(f"customer ID       {'set' if creds.customer_id else 'not set (picked automatically)'}")
    print(f"browser profile   {'present' if store.profile.exists() else 'none yet'}")
    blocked = store.mpin_blocked()
    if blocked:
        print(f"BLOCKED           the MPIN step failed on {blocked}; run setup with the right MPIN")
    return 0


def forget(store: RunStore, profile: bool) -> int:
    vault.forget()
    if profile and store.profile.exists():
        shutil.rmtree(store.profile)
    print("Deleted the stored login details" + (" and the browser profile." if profile else "."))
    return 0


def main(argv=None, store: Optional[RunStore] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.portfolio", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    sub.add_parser("check")
    f = sub.add_parser("forget")
    f.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    store = store or RunStore()
    if args.command == "setup":
        return setup(store)
    if args.command == "check":
        return check(store)
    return forget(store, args.profile)


if __name__ == "__main__":
    sys.exit(main())
