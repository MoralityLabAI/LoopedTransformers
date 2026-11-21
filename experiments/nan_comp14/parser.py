"""Anchored bounded-prefix label grammar; never scan prose for a label."""
import re


def label(text):
    if not isinstance(text, str) or len(text) > 64:
        return None
    text = text.strip()
    for pattern in (r'([A-P])\.?', r'\[([A-P])\]\.?', r'\*([A-P])\*\.?',
                    r'\*\*([A-P])\*\*\.?', r'`([A-P])`\.?'):
        found = re.fullmatch(pattern, text)
        if found:
            return found.group(1)
    return None
