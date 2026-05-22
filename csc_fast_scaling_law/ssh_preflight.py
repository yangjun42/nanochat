#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PreflightStatus:
    ok: bool
    code: str
    message: str
    valid_after: datetime | None = None
    valid_before: datetime | None = None


def _format_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _extract_validity(text: str) -> tuple[datetime, datetime] | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Valid: from "):
            continue
        try:
            raw_after, raw_before = stripped.removeprefix("Valid: from ").split(" to ", maxsplit=1)
        except ValueError:
            return None
        return datetime.fromisoformat(raw_after), datetime.fromisoformat(raw_before)
    return None


def assess_certificate_text(
    text: str,
    *,
    now: datetime | None = None,
    warn_seconds: int = 15 * 60,
) -> PreflightStatus:
    validity = _extract_validity(text)
    if validity is None:
        return PreflightStatus(
            ok=False,
            code="missing_validity",
            message="SSH certificate preflight could not find a 'Valid: from ... to ...' line.",
        )

    valid_after, valid_before = validity
    current = now or datetime.now()
    if current < valid_after:
        return PreflightStatus(
            ok=False,
            code="certificate_not_yet_valid",
            message=(
                "CSC SSH user certificate is not valid yet; "
                f"validity starts at {_format_timestamp(valid_after)}."
            ),
            valid_after=valid_after,
            valid_before=valid_before,
        )
    if current >= valid_before:
        return PreflightStatus(
            ok=False,
            code="expired_certificate",
            message=f"CSC SSH user certificate expired at {_format_timestamp(valid_before)}.",
            valid_after=valid_after,
            valid_before=valid_before,
        )
    if (valid_before - current).total_seconds() <= warn_seconds:
        return PreflightStatus(
            ok=True,
            code="certificate_expiring_soon",
            message=f"CSC SSH user certificate expires at {_format_timestamp(valid_before)}.",
            valid_after=valid_after,
            valid_before=valid_before,
        )
    return PreflightStatus(
        ok=True,
        code="ok",
        message=f"CSC SSH user certificate is valid until {_format_timestamp(valid_before)}.",
        valid_after=valid_after,
        valid_before=valid_before,
    )


def assess_certificate_file(
    cert_path: Path,
    *,
    now: datetime | None = None,
    warn_seconds: int = 15 * 60,
) -> PreflightStatus:
    if not cert_path.exists():
        return PreflightStatus(
            ok=False,
            code="missing_certificate",
            message=f"CSC SSH user certificate is missing: {cert_path}.",
        )

    result = subprocess.run(
        ["ssh-keygen", "-L", "-f", str(cert_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return PreflightStatus(
            ok=False,
            code="unreadable_certificate",
            message=f"Could not inspect CSC SSH user certificate {cert_path}: {detail}",
        )
    return assess_certificate_text(result.stdout, now=now, warn_seconds=warn_seconds)


def _ssh_agent_summary() -> str:
    result = subprocess.run(
        ["ssh-add", "-l"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    detail = (result.stdout or result.stderr).strip()
    if result.returncode == 0:
        return f"ssh-agent identities available: {detail}"
    if "no identities" in detail.lower():
        return "ssh-agent has no identities; direct cert+key SSH may still work if the certificate is valid."
    return f"ssh-agent status unavailable: {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight local CSC SSH certificate state before Roihu SSH.")
    parser.add_argument(
        "--cert",
        type=Path,
        default=Path.home() / ".ssh" / "id_ed25519-cert.pub",
        help="OpenSSH user certificate path.",
    )
    parser.add_argument(
        "--warn-seconds",
        type=int,
        default=15 * 60,
        help="Emit an expiring-soon warning within this many seconds of certificate expiry.",
    )
    args = parser.parse_args()

    status = assess_certificate_file(args.cert.expanduser(), warn_seconds=args.warn_seconds)
    print(status.message, file=sys.stderr)
    print(_ssh_agent_summary(), file=sys.stderr)
    return 0 if status.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
