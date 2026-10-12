"""Compiler error types shared by the preprocessor, lexer and parser."""


class CompilerError(Exception):
    phase = "Error"

    def __init__(self, message, line=None, col=None):
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col

    def __str__(self):
        where = ""
        if self.line is not None:
            where = " at line %d" % self.line
            if self.col is not None:
                where += ", column %d" % self.col
        return "%s%s: %s" % (self.phase, where, self.message)


class LexError(CompilerError):
    phase = "Lexical error"


class ParseError(CompilerError):
    phase = "Syntax error"
