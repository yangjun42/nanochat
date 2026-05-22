from __future__ import annotations

from datetime import datetime

from csc_fast_scaling_law.ssh_preflight import assess_certificate_text


EXPIRED_CERT = """/Users/yangju/.ssh/id_ed25519-cert.pub:
        Type: ssh-ed25519-cert-v01@openssh.com user certificate
        Public key: ED25519-CERT SHA256:IsllEn8QlbT+HOGfCWrllHbxfNSlWot/oX0NBsqetgk
        Signing CA: ED25519 SHA256:3DfRjpLKSPJ54Y8V/qLJRINvD1McizTh5DYLEXJo37U (using ssh-ed25519)
        Key ID: "yangju@helsinki.fi"
        Serial: 1779340696225096883
        Valid: from 2026-05-21T08:17:16 to 2026-05-22T08:18:16
        Principals:
                yangjun1
"""


def test_ssh_preflight_flags_expired_csc_user_certificate() -> None:
    status = assess_certificate_text(
        EXPIRED_CERT,
        now=datetime.fromisoformat("2026-05-22T08:34:43"),
    )

    assert not status.ok
    assert status.code == "expired_certificate"
    assert status.valid_before == datetime.fromisoformat("2026-05-22T08:18:16")
    assert "expired at 2026-05-22T08:18:16" in status.message


def test_ssh_preflight_warns_when_certificate_is_about_to_expire() -> None:
    status = assess_certificate_text(
        EXPIRED_CERT,
        now=datetime.fromisoformat("2026-05-22T08:10:00"),
        warn_seconds=600,
    )

    assert status.ok
    assert status.code == "certificate_expiring_soon"
    assert "expires at 2026-05-22T08:18:16" in status.message
