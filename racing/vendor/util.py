"""Repair Unity locale decimals without modifying strings or JSON arrays."""
import re

# Whole strings are consumed so apparent keys inside image/text strings are skipped.
_TOKENS = re.compile(
    r'(?P<key>"[^"\\]*(?:\\.[^"\\]*)*")'
    r'(?:(?P<prefix>\s*:\s*)(?P<number>-?\d+,\d+(?:[eE][+-]?\d+)?)'
    r'(?=\s*(?:,\s*"|})))?'
)


def replace_float_notation(value):
    def replace(match):
        if match.group("number") is None: return match.group(0)
        return match.group("key") + match.group("prefix") + match.group("number").replace(",", ".", 1)
    return _TOKENS.sub(replace, value)
