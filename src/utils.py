import re


def reformat_text(text: str) -> str:
    """
    Reformats the the text to remove newline if any character except . is before the newline

    Parameter
    ----------
    text: str
        Text that needs to be reformatted.

    Returns
    ----------
    reformatted_text: str
        Reformatted text.
    """
    reformatted_text = re.sub(r"(?<!\.)\n", " ", text)
    return reformatted_text
