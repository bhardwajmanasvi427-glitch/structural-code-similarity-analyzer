"""Stage 4 - Feature extraction and pair-feature construction.

analyze(source)      -> ProgramAnalysis (tokens, AST, counts, structural metrics)
pair_features(a, b)  -> ordered dict of numeric features describing how A and B compare
                        (all features are symmetric: pair_features(a,b) == pair_features(b,a))
"""
import difflib
import math
from collections import Counter, OrderedDict

from .lexer import tokenize
from .parser import parse
from .preprocess import preprocess

KEYWORD_LIST = ["if", "else", "for", "while", "do", "switch", "case", "break", "continue",
                "return", "int", "char", "float", "double", "long", "short", "unsigned",
                "void", "const", "static"]
OPERATOR_LIST = ["+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "&&", "||", "!",
                 "&", "|", "^", "~", "<<", ">>", "=", "+=", "-=", "*=", "/=", "%=", "++",
                 "--", "?"]
AST_KIND_LIST = ["FuncDef", "FuncDecl", "Params", "Param", "Block", "Decl", "VarDecl",
                 "ArrayDim", "Init", "InitList", "If", "For", "While", "DoWhile", "Switch",
                 "Case", "Default", "Return", "Break", "Continue", "ExprStmt", "Empty",
                 "Assign", "Binary", "Unary", "Postfix", "Ternary", "Call", "Index",
                 "Member", "Cast", "Sizeof", "Comma", "Ident", "Number", "String", "Char"]

LOOPS = {"For", "While", "DoWhile"}
CONTROL = {"For", "While", "DoWhile", "If", "Switch"}
ARITH = {"+", "-", "*", "/", "%"}
RELATIONAL = {"<", ">", "<=", ">=", "==", "!="}
LOGICAL = {"&&", "||", "!"}
BITWISE = {"&", "|", "^", "~", "<<", ">>"}

METRIC_KEYS = ["functions", "params", "loops", "conditionals", "returns", "calls", "decls",
               "assigns", "arith_ops", "rel_ops", "logic_ops", "bit_ops", "array_ops",
               "recursive_calls", "loop_depth", "ctrl_depth", "ast_depth", "ast_nodes",
               "tokens"]

METRIC_LABELS = OrderedDict([
    ("functions", "Functions defined"), ("params", "Parameters"), ("loops", "Loops"),
    ("conditionals", "Conditionals (if/?:/switch)"), ("returns", "Return statements"),
    ("calls", "Function calls"), ("decls", "Variable declarations"),
    ("assigns", "Assignments"), ("arith_ops", "Arithmetic operators"),
    ("rel_ops", "Relational operators"), ("logic_ops", "Logical operators"),
    ("bit_ops", "Bitwise operators"), ("array_ops", "Array indexing"),
    ("recursive_calls", "Self-recursive calls"), ("loop_depth", "Max loop nesting"),
    ("ctrl_depth", "Max control nesting"), ("ast_depth", "AST depth"),
    ("ast_nodes", "AST nodes"), ("tokens", "Tokens"),
])


class ProgramAnalysis:
    pass


def _nest(node, loop_d, ctrl_d):
    ld = loop_d + (node.kind in LOOPS)
    cd = ctrl_d + (node.kind in CONTROL)
    best_l, best_c = ld, cd
    for idx, child in enumerate(node.children):
        # an "else if" does not add extra nesting
        pass_ctrl = ctrl_d if (node.kind == "If" and idx == 2 and child.kind == "If") else cd
        l, c = _nest(child, ld, pass_ctrl)
        best_l, best_c = max(best_l, l), max(best_c, c)
    return best_l, best_c


def _depth(node):
    return 1 + max((_depth(c) for c in node.children), default=0)


def _sig(node, depth):
    """Structural signature of a subtree (names ignored), cut off at `depth` levels."""
    label = node.kind + (":" + node.value if node.value else "")
    if not node.children:
        return label
    if depth <= 1:
        return label + "(..)"
    return label + "(" + ",".join(_sig(c, depth - 1) for c in node.children) + ")"


def _vector(counter, vocab):
    return [counter.get(v, 0) for v in vocab]


def analyze(source, name=""):
    a = ProgramAnalysis()
    a.name = name
    a.source = source
    pre = preprocess(source)
    a.pre = pre
    all_tokens = tokenize(pre.clean)
    a.tokens = all_tokens[:-1]                        # drop EOF
    a.ast = parse(all_tokens)
    nodes = list(a.ast.walk())
    a.nodes = nodes

    # ---- token level
    a.norm_tokens = []
    for t in a.tokens:
        if t.type == "IDENT":
            a.norm_tokens.append("ID")
        elif t.type in ("NUMBER", "STRING", "CHAR"):
            a.norm_tokens.append({"NUMBER": "NUM", "STRING": "STR", "CHAR": "CHR"}[t.type])
        else:
            a.norm_tokens.append(t.value)
    a.token_type_counts = Counter(t.type for t in a.tokens)
    a.kw_counts = Counter(t.value for t in a.tokens if t.type == "KEYWORD")
    a.op_counts = Counter(t.value for t in a.tokens if t.type == "OP")
    a.kw_vec = _vector(a.kw_counts, KEYWORD_LIST)
    a.op_vec = _vector(a.op_counts, OPERATOR_LIST)
    a.identifiers = sorted({t.value for t in a.tokens if t.type == "IDENT"})

    # ---- AST level
    a.kind_counts = Counter(n.kind for n in nodes)
    a.kind_vec = _vector(a.kind_counts, AST_KIND_LIST)
    a.kind_seq = [n.kind + (":" + n.value if n.value else "") for n in nodes]
    a.sigs = Counter(_sig(n, 3) for n in nodes if n.children)

    # ---- structural metrics
    kc = a.kind_counts
    bin_ops = Counter(n.value for n in nodes if n.kind == "Binary")
    un_ops = Counter(n.value for n in nodes if n.kind == "Unary")
    all_ops = bin_ops + un_ops
    recursive = 0
    for fn in (n for n in nodes if n.kind == "FuncDef"):
        for c in fn.walk():
            if c.kind == "Call" and c.children[0].kind == "Ident" and c.children[0].name == fn.name:
                recursive += 1
    loop_d, ctrl_d = _nest(a.ast, 0, 0)
    a.metrics = {
        "functions": kc["FuncDef"],
        "params": kc["Param"],
        "loops": sum(kc[k] for k in LOOPS),
        "conditionals": kc["If"] + kc["Ternary"] + kc["Switch"],
        "returns": kc["Return"],
        "calls": kc["Call"],
        "decls": kc["VarDecl"],
        "assigns": kc["Assign"] + kc["Postfix"],
        "arith_ops": sum(all_ops[o] for o in ARITH),
        "rel_ops": sum(all_ops[o] for o in RELATIONAL),
        "logic_ops": sum(all_ops[o] for o in LOGICAL),
        "bit_ops": sum(all_ops[o] for o in BITWISE),
        "array_ops": kc["Index"],
        "recursive_calls": recursive,
        "loop_depth": loop_d,
        "ctrl_depth": ctrl_d,
        "ast_depth": _depth(a.ast),
        "ast_nodes": len(nodes),
        "tokens": len(a.tokens),
    }
    a.loop_kinds = Counter({k: kc[k] for k in LOOPS if kc[k]})
    a.arith_set = {o for o in ARITH if all_ops[o]}
    return a


# --------------------------------------------------------------------------- similarity measures
def cosine(u, v):
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 and nv == 0:
        return 1.0
    if nu == 0 or nv == 0:
        return 0.0
    return sum(x * y for x, y in zip(u, v)) / (nu * nv)


def jaccard(s, t):
    if not s and not t:
        return 1.0
    return len(s & t) / float(len(s | t))


def multiset_jaccard(c1, c2):
    keys = set(c1) | set(c2)
    if not keys:
        return 1.0
    inter = sum(min(c1[k], c2[k]) for k in keys)
    union = sum(max(c1[k], c2[k]) for k in keys)
    return inter / float(union)


def ngrams(seq, n):
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


def seq_ratio(x, y):
    """Symmetric SequenceMatcher ratio (inputs are put in a canonical order first)."""
    if not x and not y:
        return 1.0
    if (len(x), x) > (len(y), y):
        x, y = y, x
    return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()


def rel_diff(x, y):
    return abs(x - y) / float(max(x, y, 1))


SIM_FEATURES = ["cos_keywords", "cos_operators", "cos_ast_kinds", "op_set_jaccard",
                "tok_bigram_jaccard", "tok_trigram_jaccard", "token_seq_ratio",
                "ast_seq_ratio", "subtree_jaccard"]
FEATURE_NAMES = SIM_FEATURES + ["d_" + k for k in METRIC_KEYS]

FEATURE_DESCRIPTIONS = {
    "cos_keywords": "Keyword-usage similarity (cosine)",
    "cos_operators": "Operator-usage similarity (cosine)",
    "cos_ast_kinds": "AST node-type distribution similarity (cosine)",
    "op_set_jaccard": "Overlap of operator sets (Jaccard)",
    "tok_bigram_jaccard": "Token bigram overlap, identifiers normalised (Jaccard)",
    "tok_trigram_jaccard": "Token trigram overlap, identifiers normalised (Jaccard)",
    "token_seq_ratio": "Normalised token-sequence similarity",
    "ast_seq_ratio": "AST pre-order sequence similarity",
    "subtree_jaccard": "Shared AST sub-structures (multiset Jaccard)",
}
for _k in METRIC_KEYS:
    FEATURE_DESCRIPTIONS["d_" + _k] = "Relative difference in: " + METRIC_LABELS[_k]


def pair_features(a, b):
    f = OrderedDict()
    f["cos_keywords"] = cosine(a.kw_vec, b.kw_vec)
    f["cos_operators"] = cosine(a.op_vec, b.op_vec)
    f["cos_ast_kinds"] = cosine(a.kind_vec, b.kind_vec)
    f["op_set_jaccard"] = jaccard({o for o in a.op_counts}, {o for o in b.op_counts})
    f["tok_bigram_jaccard"] = jaccard(ngrams(a.norm_tokens, 2), ngrams(b.norm_tokens, 2))
    f["tok_trigram_jaccard"] = jaccard(ngrams(a.norm_tokens, 3), ngrams(b.norm_tokens, 3))
    f["token_seq_ratio"] = seq_ratio(a.norm_tokens, b.norm_tokens)
    f["ast_seq_ratio"] = seq_ratio(a.kind_seq, b.kind_seq)
    f["subtree_jaccard"] = multiset_jaccard(a.sigs, b.sigs)
    for k in METRIC_KEYS:
        f["d_" + k] = rel_diff(a.metrics[k], b.metrics[k])
    return f


def feature_vector(a, b):
    f = pair_features(a, b)
    return [f[n] for n in FEATURE_NAMES]
