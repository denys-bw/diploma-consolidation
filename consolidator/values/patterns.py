from __future__ import annotations


def value_shape(value: str) -> str:
    shape: list[str] = []
    previous = ""
    for character in value:
        if character.isdigit():
            token = "9"
        elif character.isalpha():
            token = "A" if character.isupper() else "a"
        elif character.isspace():
            token = " "
        else:
            token = character
        if token in ("A", "a", " ") and token == previous:
            continue
        shape.append(token)
        previous = token
    return "".join(shape)
