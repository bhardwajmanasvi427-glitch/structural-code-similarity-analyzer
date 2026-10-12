"""Stage 3 - Syntax analysis: recursive-descent parser (C subset) that builds an AST.

Grammar (EBNF, simplified):
    program      = { external } ;
    external     = type declarator ( "(" params ")" ( block | ";" ) | declarators ";" ) ;
    declarators  = declarator { "," declarator } ;
    declarator   = { "*" } IDENT { "[" [expr] "]" } [ "=" initializer ] ;
    statement    = block | if | for | while | do | switch | return | break | continue
                 | declaration ";" | [expr] ";" ;
    expr         = assign { "," assign } ;
    assign       = ternary [ assign_op assign ] ;
    ternary      = binary [ "?" expr ":" ternary ] ;
    binary       = unary { binop unary }            (precedence climbing)
    unary        = ( "!" | "-" | "+" | "~" | "*" | "&" | "++" | "--" ) unary
                 | "sizeof" ... | "(" type ")" unary | postfix ;
    postfix      = primary { "(" args ")" | "[" expr "]" | "." IDENT | "->" IDENT | "++" | "--" } ;
    primary      = IDENT | NUMBER | STRING | CHAR | "(" expr ")" ;

Identifier names are stored in Node.name; they are NOT part of the structural
signature, so renaming variables never changes the AST shape.
"""
from .errors import ParseError

TYPE_WORDS = {"int", "char", "float", "double", "long", "short", "unsigned", "signed",
              "void", "const", "static", "extern", "register", "volatile"}
QUALIFIERS = {"const", "static", "extern", "register", "volatile"}
STD_FUNCS = {"printf", "scanf", "puts", "putchar", "getchar", "strlen", "strcmp", "strcpy",
             "strcat", "malloc", "free", "abs", "sqrt", "pow", "fabs", "memset", "memcpy",
             "atoi", "rand", "srand", "exit", "fgets", "sprintf", "fprintf"}
ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
BINARY_PREC = {"||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6,
               "<": 7, ">": 7, "<=": 7, ">=": 7, "<<": 8, ">>": 8,
               "+": 9, "-": 9, "*": 10, "/": 10, "%": 10}
UNARY_OPS = {"!", "-", "+", "~", "*", "&", "++", "--"}


class Node:
    __slots__ = ("kind", "value", "children", "name", "line")

    def __init__(self, kind, value=None, children=None, name=None, line=0):
        self.kind = kind
        self.value = value
        self.children = children if children is not None else []
        self.name = name
        self.line = line

    def walk(self):
        yield self
        for c in self.children:
            for n in c.walk():
                yield n

    def pretty(self, depth=0):
        label = self.kind
        if self.value is not None:
            label += " [%s]" % self.value
        if self.name:
            label += " '%s'" % self.name
        return "\n".join(["  " * depth + label] + [c.pretty(depth + 1) for c in self.children])

    def __repr__(self):
        return "Node(%s)" % self.kind


class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.p = 0

    # ---------------------------------------------------------------- helpers
    def peek(self, k=0):
        return self.t[min(self.p + k, len(self.t) - 1)]

    def advance(self):
        tok = self.t[self.p]
        if tok.type != "EOF":
            self.p += 1
        return tok

    def at(self, value, k=0):
        tok = self.peek(k)
        return tok.type in ("KEYWORD", "OP", "DELIM") and tok.value == value

    def accept(self, value):
        if self.at(value):
            return self.advance()
        return None

    def error(self, message, tok=None):
        tok = tok or self.peek()
        raise ParseError(message, tok.line, tok.col)

    def found(self, tok):
        return "end of file" if tok.type == "EOF" else "'%s'" % tok.value

    def expect(self, value):
        if self.at(value):
            return self.advance()
        tok = self.peek()
        self.error("expected '%s' but found %s" % (value, self.found(tok)), tok)

    def expect_ident(self):
        tok = self.peek()
        if tok.type != "IDENT":
            self.error("expected an identifier but found %s" % self.found(tok), tok)
        return self.advance()

    def starts_type(self, k=0):
        tok = self.peek(k)
        return tok.type == "KEYWORD" and tok.value in TYPE_WORDS

    # ------------------------------------------------------------ declarations
    def parse_program(self):
        prog = Node("Program", line=1)
        while self.peek().type != "EOF":
            prog.children.append(self.parse_external())
        return prog

    def parse_base_type(self):
        words = []
        while self.starts_type():
            words.append(self.advance().value)
        if not words:
            self.error("expected a type name but found %s" % self.found(self.peek()))
        core = [w for w in words if w not in QUALIFIERS]
        return " ".join(core) if core else "int"

    def parse_external(self):
        if not self.starts_type():
            self.error("expected a declaration or function definition but found %s"
                       % self.found(self.peek()))
        base = self.parse_base_type()
        save = self.p
        stars = 0
        while self.accept("*"):
            stars += 1
        name_tok = self.expect_ident()
        if self.at("("):
            params = self.parse_params()
            vtype = base + "*" * stars
            if self.at("{"):
                body = self.parse_block()
                return Node("FuncDef", vtype, [params, body], name_tok.value, name_tok.line)
            self.expect(";")
            return Node("FuncDecl", vtype, [params], name_tok.value, name_tok.line)
        self.p = save
        decl = Node("Decl", line=name_tok.line, children=self.parse_declarators(base))
        self.expect(";")
        return decl

    def parse_declarators(self, base):
        decls = []
        while True:
            stars = 0
            while self.accept("*"):
                stars += 1
            name_tok = self.expect_ident()
            node = Node("VarDecl", base + "*" * stars, name=name_tok.value, line=name_tok.line)
            while self.accept("["):
                dim = Node("ArrayDim")
                if not self.at("]"):
                    dim.children.append(self.parse_ternary())
                self.expect("]")
                node.children.append(dim)
            if self.accept("="):
                node.children.append(Node("Init", children=[self.parse_initializer()]))
            decls.append(node)
            if not self.accept(","):
                break
        return decls

    def parse_initializer(self):
        if self.accept("{"):
            node = Node("InitList")
            while not self.at("}"):
                node.children.append(self.parse_initializer())
                if not self.accept(","):
                    break
            self.expect("}")
            return node
        return self.parse_assign()

    def parse_params(self):
        self.expect("(")
        params = Node("Params")
        if self.accept(")"):
            return params
        if self.at("void") and self.at(")", 1):
            self.advance()
            self.advance()
            return params
        while True:
            base = self.parse_base_type()
            stars = 0
            while self.accept("*"):
                stars += 1
            name = None
            line = self.peek().line
            if self.peek().type == "IDENT":
                tok = self.advance()
                name, line = tok.value, tok.line
            node = Node("Param", base + "*" * stars, name=name, line=line)
            while self.accept("["):
                dim = Node("ArrayDim")
                if not self.at("]"):
                    dim.children.append(self.parse_ternary())
                self.expect("]")
                node.children.append(dim)
            params.children.append(node)
            if not self.accept(","):
                break
        self.expect(")")
        return params

    # -------------------------------------------------------------- statements
    def parse_block(self):
        open_tok = self.expect("{")
        block = Node("Block", line=open_tok.line)
        while not self.at("}"):
            if self.peek().type == "EOF":
                self.error("missing '}' to close the block opened at line %d" % open_tok.line)
            block.children.append(self.parse_statement())
        self.expect("}")
        return block

    def as_block(self, stmt):
        return stmt if stmt.kind == "Block" else Node("Block", children=[stmt], line=stmt.line)

    def parse_declaration(self):
        base = self.parse_base_type()
        return Node("Decl", line=self.peek().line, children=self.parse_declarators(base))

    def parse_statement(self):
        tok = self.peek()
        if self.at("{"):
            return self.parse_block()
        if self.at(";"):
            self.advance()
            return Node("Empty", line=tok.line)
        if tok.type == "KEYWORD":
            v = tok.value
            if v == "if":
                return self.parse_if()
            if v == "for":
                return self.parse_for()
            if v == "while":
                self.advance()
                self.expect("(")
                cond = self.parse_expr()
                self.expect(")")
                return Node("While", children=[cond, self.as_block(self.parse_statement())], line=tok.line)
            if v == "do":
                self.advance()
                body = self.as_block(self.parse_statement())
                self.expect("while")
                self.expect("(")
                cond = self.parse_expr()
                self.expect(")")
                self.expect(";")
                return Node("DoWhile", children=[body, cond], line=tok.line)
            if v == "switch":
                return self.parse_switch()
            if v == "return":
                self.advance()
                node = Node("Return", line=tok.line)
                if not self.at(";"):
                    node.children.append(self.parse_expr())
                self.expect(";")
                return node
            if v in ("break", "continue"):
                self.advance()
                self.expect(";")
                return Node(v.capitalize(), line=tok.line)
            if v in TYPE_WORDS:
                node = self.parse_declaration()
                self.expect(";")
                return node
            self.error("unsupported or misplaced keyword '%s'" % v, tok)
        expr = self.parse_expr()
        self.expect(";")
        return Node("ExprStmt", children=[expr], line=tok.line)

    def parse_if(self):
        tok = self.advance()
        self.expect("(")
        cond = self.parse_expr()
        self.expect(")")
        node = Node("If", children=[cond, self.as_block(self.parse_statement())], line=tok.line)
        if self.accept("else"):
            if self.at("if"):
                node.children.append(self.parse_if())
            else:
                node.children.append(self.as_block(self.parse_statement()))
        return node

    def parse_for(self):
        tok = self.advance()
        self.expect("(")
        if self.at(";"):
            init = Node("Empty")
        elif self.starts_type():
            init = self.parse_declaration()
        else:
            init = self.parse_expr()
        self.expect(";")
        cond = Node("Empty") if self.at(";") else self.parse_expr()
        self.expect(";")
        update = Node("Empty") if self.at(")") else self.parse_expr()
        self.expect(")")
        body = self.as_block(self.parse_statement())
        return Node("For", children=[init, cond, update, body], line=tok.line)

    def parse_switch(self):
        tok = self.advance()
        self.expect("(")
        subject = self.parse_expr()
        self.expect(")")
        self.expect("{")
        cases = Node("Block", line=tok.line)
        while not self.at("}"):
            if self.accept("case"):
                node = Node("Case", children=[self.parse_ternary()])
            elif self.accept("default"):
                node = Node("Default")
            else:
                self.error("expected 'case' or 'default' inside switch but found %s"
                           % self.found(self.peek()))
            self.expect(":")
            while not (self.at("}") or self.at("case") or self.at("default")):
                node.children.append(self.parse_statement())
            cases.children.append(node)
        self.expect("}")
        return Node("Switch", children=[subject, cases], line=tok.line)

    # ------------------------------------------------------------- expressions
    def parse_expr(self):
        first = self.parse_assign()
        if not self.at(","):
            return first
        node = Node("Comma", children=[first], line=first.line)
        while self.accept(","):
            node.children.append(self.parse_assign())
        return node

    def parse_assign(self):
        left = self.parse_ternary()
        tok = self.peek()
        if tok.type == "OP" and tok.value in ASSIGN_OPS:
            self.advance()
            right = self.parse_assign()
            return Node("Assign", tok.value, [left, right], line=tok.line)
        return left

    def parse_ternary(self):
        cond = self.parse_binary(1)
        if self.at("?"):
            tok = self.advance()
            a = self.parse_expr()
            self.expect(":")
            b = self.parse_ternary()
            return Node("Ternary", children=[cond, a, b], line=tok.line)
        return cond

    def parse_binary(self, min_prec):
        left = self.parse_unary()
        while True:
            tok = self.peek()
            prec = BINARY_PREC.get(tok.value) if tok.type == "OP" else None
            if prec is None or prec < min_prec:
                return left
            self.advance()
            right = self.parse_binary(prec + 1)
            left = Node("Binary", tok.value, [left, right], line=tok.line)

    def parse_unary(self):
        tok = self.peek()
        if tok.type == "OP" and tok.value in UNARY_OPS:
            self.advance()
            return Node("Unary", tok.value, [self.parse_unary()], line=tok.line)
        if self.at("sizeof"):
            self.advance()
            if self.at("(") and self.starts_type(1):
                self.advance()
                base = self.parse_base_type()
                stars = 0
                while self.accept("*"):
                    stars += 1
                self.expect(")")
                return Node("Sizeof", base + "*" * stars, line=tok.line)
            return Node("Sizeof", children=[self.parse_unary()], line=tok.line)
        if self.at("(") and self.starts_type(1):
            self.advance()
            base = self.parse_base_type()
            stars = 0
            while self.accept("*"):
                stars += 1
            self.expect(")")
            return Node("Cast", base + "*" * stars, [self.parse_unary()], line=tok.line)
        return self.parse_postfix()

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            tok = self.peek()
            if self.at("("):
                self.advance()
                call = Node("Call", children=[node], line=tok.line)
                if node.kind == "Ident" and node.name in STD_FUNCS:
                    call.value = node.name
                if not self.at(")"):
                    while True:
                        call.children.append(self.parse_assign())
                        if not self.accept(","):
                            break
                self.expect(")")
                node = call
            elif self.at("["):
                self.advance()
                idx = self.parse_expr()
                self.expect("]")
                node = Node("Index", children=[node, idx], line=tok.line)
            elif self.at(".") or self.at("->"):
                self.advance()
                field = self.expect_ident()
                node = Node("Member", tok.value, [node], field.value, tok.line)
            elif self.at("++") or self.at("--"):
                self.advance()
                node = Node("Postfix", tok.value, [node], line=tok.line)
            else:
                return node

    def parse_primary(self):
        tok = self.peek()
        if tok.type == "IDENT":
            self.advance()
            return Node("Ident", name=tok.value, line=tok.line)
        if tok.type == "NUMBER":
            self.advance()
            text = tok.value
            if text in ("0", "1", "2"):
                value = "int:" + text
            elif "." in text or (("e" in text.lower()) and not text.lower().startswith("0x")):
                value = "float"
            else:
                value = "int"
            return Node("Number", value, name=text, line=tok.line)
        if tok.type == "STRING":
            self.advance()
            return Node("String", name=tok.value, line=tok.line)
        if tok.type == "CHAR":
            self.advance()
            return Node("Char", name=tok.value, line=tok.line)
        if self.at("("):
            self.advance()
            inner = self.parse_expr()
            self.expect(")")
            return inner
        self.error("unexpected %s in expression" % self.found(tok), tok)


def parse(tokens):
    return Parser(tokens).parse_program()
