from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


def test_with_structured_output_pydantic_contract(chat):
    class ContactModel(BaseModel):
        name: str = Field(..., min_length=1)
        email: EmailStr
        phone: str = Field(..., min_length=5)

    runnable = chat.with_structured_output(ContactModel)

    text = (
        "Por favor extrae los datos.\n"
        "Nombre: Carlos Rodríguez\n"
        "Correo: carlos.rodriguez@example.com\n"
        "Teléfono: 0412-000-1111\n"
    )

    out = runnable.invoke(text)

    assert isinstance(out, ContactModel)
    assert out.name.strip() == "Carlos Rodríguez"
    assert str(out.email) == "carlos.rodriguez@example.com"
    assert any(ch.isdigit() for ch in out.phone)

