"""Stage 1 - Preprocessing.

* removes // and /* */ comments (string/char literals are respected)
* removes preprocessor directives (#include, #define, ...)
* keeps line numbers stable so later error messages point at the right line
* produces a whitespace-normalised view for display
"""
from .errors import LexError


class PreprocessResult:
    def __init__(self, clean, directives, comments_removed, normalized):
        self.clean = clean                      # line-number preserving, comment-free source
        self.directives = directives            # list of removed directive lines
        self.comments_removed = comments_removed
        self.normalized = normalized            # compact text (display only)


def strip_comments(source):
    out = []
    i, n = 0, len(source)
    removed = 0
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if c == '"' or c == "'":
            j = i + 1
            while j < n and source[j] != c and source[j] != "\n":
                j += 2 if (source[j] == "\\" and j + 1 < n) else 1
            if j < n and source[j] == c:
                j += 1
            out.append(source[i:j])             # unterminated literals are reported by the lexer
            i = j
        elif c == "/" and nxt == "/":
            j = source.find("\n", i)
            j = n if j == -1 else j
            out.append(" ")
            removed += 1
            i = j
        elif c == "/" and nxt == "*":
            j = source.find("*/", i + 2)
            if j == -1:
                raise LexError("unterminated /* comment", source.count("\n", 0, i) + 1)
            out.append(" " + "\n" * source.count("\n", i, j + 2))
            removed += 1
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out), removed


def strip_directives(text):
    lines = text.split("\n")
    directives = []
    k = 0
    while k < len(lines):
        if lines[k].lstrip().startswith("#"):
            directives.append(lines[k].strip())
            while lines[k].rstrip().endswith("\\") and k + 1 < len(lines):   # multi-line macro
                lines[k] = ""
                k += 1
            lines[k] = ""
        k += 1
    return "\n".join(lines), directives


def preprocess(source):
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    no_comments, removed = strip_comments(source)
    clean, directives = strip_directives(no_comments)
    clean = "\n".join(line.rstrip() for line in clean.split("\n"))
    normalized = "\n".join(l.strip() for l in clean.split("\n") if l.strip())
    return PreprocessResult(clean, directives, removed, normalized)
