from __future__ import annotations

from typing import TypedDict


def test_with_structured_output_typed_dict_contract(chat):
    class ContactInfo(TypedDict):
        name: str
        email: str
        phone: str

    runnable = chat.with_structured_output(ContactInfo)

    text = (
        "Datos de contacto:\n"
        "Nombre: Ana Pérez\n"
        "Email: ana.perez@example.com\n"
        "Teléfono: +58 412-123-4567\n"
    )

    out = runnable.invoke(text)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"name", "email", "phone"}
    assert isinstance(out["name"], str) and out["name"].strip()
    assert isinstance(out["email"], str) and "@" in out["email"]
    assert isinstance(out["phone"], str) and any(ch.isdigit() for ch in out["phone"])

