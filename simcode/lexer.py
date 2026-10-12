"""Stage 2 - Lexical analysis: source text -> token list (regex based scanner)."""
import re
from collections import namedtuple

from .errors import LexError

Token = namedtuple("Token", "type value line col")

KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for", "goto", "if", "int", "long", "register",
    "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while",
}

_OPS = ["<<=", ">>=", "++", "--", "->", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
        "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
        "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~", "?", ":", "."]

_MASTER = re.compile("|".join([
    r"(?P<WS>[ \t\r\f\v]+)",
    r"(?P<NL>\n)",
    r"(?P<NUMBER>0[xX][0-9a-fA-F]+[uUlL]*"
    r"|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlL]?"
    r"|\d+[eE][+-]?\d+[fFlL]?"
    r"|\d+[uUlL]*)",
    r'(?P<STRING>"(?:\\.|[^"\\\n])*")',
    r"(?P<CHAR>'(?:\\.|[^'\\\n])+')",
    r"(?P<IDENT>[A-Za-z_]\w*)",
    "(?P<OP>" + "|".join(re.escape(o) for o in _OPS) + ")",
    r"(?P<DELIM>[;,(){}\[\]])",
    r"(?P<MISMATCH>.)",
]))


def tokenize(source):
    """Return a list of Tokens ending with an EOF token."""
    tokens = []
    line, line_start = 1, 0
    for m in _MASTER.finditer(source):
        kind, text = m.lastgroup, m.group()
        col = m.start() - line_start + 1
        if kind == "NL":
            line += 1
            line_start = m.end()
            continue
        if kind == "WS":
            continue
        if kind == "MISMATCH":
            if text == '"':
                raise LexError("unterminated string literal", line, col)
            if text == "'":
                raise LexError("unterminated or empty character literal", line, col)
            raise LexError("unexpected character %r" % text, line, col)
        if kind == "IDENT" and text in KEYWORDS:
            kind = "KEYWORD"
        tokens.append(Token(kind, text, line, col))
    tokens.append(Token("EOF", "", line, len(source) - line_start + 1))
    return tokens
