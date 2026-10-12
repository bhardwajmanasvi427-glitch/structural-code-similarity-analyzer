"""Synthetic training-data generator.

Each *family* is one algorithm written in C with one or more implementation *variants*
(e.g. for-loop / while-loop / recursion). Families that share a skeleton are put in the same
*group* (e.g. sorting algorithms).

Labelling rule for a pair of programs (this is the ground truth the ML model learns):
    High      (2)  same algorithm, same implementation; only names / comments / layout /
                   harmless extra statements differ
    Moderate  (1)  same algorithm with a different implementation (for vs while vs recursion)
                   OR different algorithms from the same group (similar skeleton, different logic)
    Low       (0)  algorithms from different groups
"""
import random
import re

LOW, MODERATE, HIGH = 0, 1, 2

HEADER = "#include <stdio.h>\n\n"

POOLS = {
    "fn": ["compute", "calc", "process", "solve", "helper", "run", "evaluate", "func",
           "doIt", "calculate", "getResult", "task", "work", "apply", "check", "do_work"],
    "arr": ["arr", "numbers", "data", "values", "nums", "list", "xs", "items", "v"],
    "n": ["n", "size", "len", "count", "total", "num", "limit", "length"],
    "i": ["i", "idx", "p", "pos", "ii", "k1"],
    "j": ["j", "q", "jj", "inner", "m2"],
    "k": ["k", "t", "c", "step", "z"],
    "acc": ["total", "result", "acc", "s", "res", "answer", "value", "out"],
    "tmp": ["tmp", "temp", "t", "swap", "hold", "aux"],
    "data": ["data", "input", "nums", "sample", "arr", "values", "testData"],
    "res": ["res", "r", "output", "answer", "ret", "final"],
    "a": ["a", "x", "first", "p1", "u"],
    "b": ["b", "y", "second", "p2", "w"],
    "s": ["s", "text", "str", "word", "buf"],
    "best": ["best", "maxVal", "largest", "m", "top", "biggest"],
    "count": ["count", "cnt", "found", "matches", "c"],
    "base": ["base", "x", "b", "number"],
    "e": ["e", "exp", "power", "pw"],
    "key": ["key", "current", "val", "item"],
    "low": ["low", "lo", "left", "start", "l"],
    "high": ["high", "hi", "right", "end", "r"],
    "mid": ["mid", "middle", "m", "center"],
    "target": ["target", "x", "needle", "wanted", "goal"],
    "minIdx": ["minIdx", "smallest", "mi", "minPos"],
    "swapped": ["swapped", "changed", "flag", "again"],
    "rev": ["rev", "reversed", "r", "out"],
    "orig": ["orig", "original", "copy", "saved"],
    "len": ["len", "length", "size", "cnt", "n"],
    "year": ["year", "y", "yr"],
    "ch": ["ch", "c", "letter", "target"],
    "m": ["m", "mat", "grid", "matrix", "table"],
    "rows": ["rows", "n", "height", "lines"],
    "text": ["text", "msg", "line", "word", "str"],
    "unused": ["unused", "spare", "extra", "dummy"],
}


def _main_value(call, decl="", fmt="%d", ctype="int"):
    return ("int main() {\n%s    %s @res@ = %s;\n    printf(\"%s\\n\", @res@);\n    return 0;\n}\n"
            % (decl, ctype, call, fmt))


def _main_arr(call, values, fmt="%d", ctype="int"):
    n = len(values.split(","))
    decl = "    int @data@[%d] = {%s};\n" % (n, values)
    return _main_value(call.replace("N", str(n)), decl, fmt, ctype)


MAIN_SORT = ("int main() {\n    int @data@[6] = {5, 2, 9, 1, 7, 3};\n    @fn@(@data@, 6);\n"
             "    for (int @k@ = 0; @k@ < 6; @k@++) {\n        printf(\"%d \", @data@[@k@]);\n    }\n"
             "    printf(\"\\n\");\n    return 0;\n}\n")

SIG_ARR = "int @fn@(int @arr@[], int @n@)"

FAMILIES = {}


def fam(name, group, variants, main):
    FAMILIES[name] = {"group": group, "variants": variants, "main": main}


# ------------------------------------------------------------------ array_scan
fam("sum_array", "array_scan", [
    SIG_ARR + """ {
    int @acc@ = 0;
    for (int @i@ = 0; @i@ < @n@; @i@++) {
        @acc@ += @arr@[@i@];
    }
    return @acc@;
}
""", SIG_ARR + """ {
    int @acc@ = 0;
    int @i@ = 0;
    while (@i@ < @n@) {
        @acc@ = @acc@ + @arr@[@i@];
        @i@++;
    }
    return @acc@;
}
""", SIG_ARR + """ {
    if (@n@ <= 0) {
        return 0;
    }
    return @arr@[@n@ - 1] + @fn@(@arr@, @n@ - 1);
}
"""], _main_arr("@fn@(@data@, N)", "4, 8, 15, 16, 23"))

fam("product_array", "array_scan", [
    SIG_ARR + """ {
    int @acc@ = 1;
    for (int @i@ = 0; @i@ < @n@; @i@++) {
        @acc@ *= @arr@[@i@];
    }
    return @acc@;
}
""", SIG_ARR + """ {
    int @acc@ = 1;
    int @i@ = 0;
    while (@i@ < @n@) {
        @acc@ = @acc@ * @arr@[@i@];
        @i@++;
    }
    return @acc@;
}
"""], _main_arr("@fn@(@data@, N)", "1, 2, 3, 4, 5"))

fam("max_array", "array_scan", [
    SIG_ARR + """ {
    int @best@ = @arr@[0];
    for (int @i@ = 1; @i@ < @n@; @i@++) {
        if (@arr@[@i@] > @best@) {
            @best@ = @arr@[@i@];
        }
    }
    return @best@;
}
""", SIG_ARR + """ {
    int @best@ = @arr@[0];
    int @i@ = 1;
    while (@i@ < @n@) {
        if (@arr@[@i@] > @best@) {
            @best@ = @arr@[@i@];
        }
        @i@++;
    }
    return @best@;
}
"""], _main_arr("@fn@(@data@, N)", "3, 17, 4, 9, 11, 2"))

fam("count_even", "array_scan", [
    SIG_ARR + """ {
    int @count@ = 0;
    for (int @i@ = 0; @i@ < @n@; @i@++) {
        if (@arr@[@i@] % 2 == 0) {
            @count@++;
        }
    }
    return @count@;
}
""", SIG_ARR + """ {
    int @count@ = 0;
    int @i@ = 0;
    while (@i@ < @n@) {
        if (@arr@[@i@] % 2 == 0) {
            @count@ = @count@ + 1;
        }
        @i@++;
    }
    return @count@;
}
"""], _main_arr("@fn@(@data@, N)", "2, 7, 8, 5, 10, 3, 4"))

# ------------------------------------------------------------------ number_loop
fam("factorial", "number_loop", [
    """long @fn@(int @n@) {
    long @acc@ = 1;
    for (int @i@ = 2; @i@ <= @n@; @i@++) {
        @acc@ *= @i@;
    }
    return @acc@;
}
""", """long @fn@(int @n@) {
    long @acc@ = 1;
    while (@n@ > 1) {
        @acc@ = @acc@ * @n@;
        @n@--;
    }
    return @acc@;
}
""", """long @fn@(int @n@) {
    if (@n@ <= 1) {
        return 1;
    }
    return @n@ * @fn@(@n@ - 1);
}
"""], _main_value("@fn@(10)", fmt="%ld", ctype="long"))

fam("fibonacci", "number_loop", [
    """int @fn@(int @n@) {
    int @a@ = 0;
    int @b@ = 1;
    for (int @i@ = 0; @i@ < @n@; @i@++) {
        int @tmp@ = @a@ + @b@;
        @a@ = @b@;
        @b@ = @tmp@;
    }
    return @a@;
}
""", """int @fn@(int @n@) {
    if (@n@ < 2) {
        return @n@;
    }
    return @fn@(@n@ - 1) + @fn@(@n@ - 2);
}
"""], _main_value("@fn@(10)"))

fam("power", "number_loop", [
    """long @fn@(int @base@, int @e@) {
    long @acc@ = 1;
    for (int @i@ = 0; @i@ < @e@; @i@++) {
        @acc@ *= @base@;
    }
    return @acc@;
}
""", """long @fn@(int @base@, int @e@) {
    if (@e@ == 0) {
        return 1;
    }
    return @base@ * @fn@(@base@, @e@ - 1);
}
"""], _main_value("@fn@(2, 10)", fmt="%ld", ctype="long"))

# ------------------------------------------------------------------ number_theory
fam("gcd", "number_theory", [
    """int @fn@(int @a@, int @b@) {
    while (@b@ != 0) {
        int @tmp@ = @b@;
        @b@ = @a@ % @b@;
        @a@ = @tmp@;
    }
    return @a@;
}
""", """int @fn@(int @a@, int @b@) {
    if (@b@ == 0) {
        return @a@;
    }
    return @fn@(@b@, @a@ % @b@);
}
"""], _main_value("@fn@(48, 18)"))

fam("is_prime", "number_theory", [
    """int @fn@(int @n@) {
    if (@n@ < 2) {
        return 0;
    }
    for (int @i@ = 2; @i@ * @i@ <= @n@; @i@++) {
        if (@n@ % @i@ == 0) {
            return 0;
        }
    }
    return 1;
}
""", """int @fn@(int @n@) {
    if (@n@ < 2) {
        return 0;
    }
    int @i@ = 2;
    while (@i@ * @i@ <= @n@) {
        if (@n@ % @i@ == 0) {
            return 0;
        }
        @i@++;
    }
    return 1;
}
"""], _main_value("@fn@(29)"))

# ------------------------------------------------------------------ digits
fam("digit_sum", "digits", [
    """int @fn@(int @n@) {
    int @acc@ = 0;
    while (@n@ > 0) {
        @acc@ += @n@ % 10;
        @n@ /= 10;
    }
    return @acc@;
}
""", """int @fn@(int @n@) {
    if (@n@ == 0) {
        return 0;
    }
    return @n@ % 10 + @fn@(@n@ / 10);
}
"""], _main_value("@fn@(12345)"))

fam("reverse_number", "digits", [
    """int @fn@(int @n@) {
    int @rev@ = 0;
    while (@n@ > 0) {
        @rev@ = @rev@ * 10 + @n@ % 10;
        @n@ = @n@ / 10;
    }
    return @rev@;
}
""", """int @fn@(int @n@) {
    int @rev@ = 0;
    for (; @n@ > 0; @n@ /= 10) {
        @rev@ = @rev@ * 10 + @n@ % 10;
    }
    return @rev@;
}
"""], _main_value("@fn@(12345)"))

fam("palindrome_number", "digits", [
    """int @fn@(int @n@) {
    int @orig@ = @n@;
    int @rev@ = 0;
    while (@n@ > 0) {
        @rev@ = @rev@ * 10 + @n@ % 10;
        @n@ /= 10;
    }
    if (@orig@ == @rev@) {
        return 1;
    }
    return 0;
}
"""], _main_value("@fn@(12321)"))

# ------------------------------------------------------------------ sorting
fam("bubble_sort", "sorting", [
    """void @fn@(int @arr@[], int @n@) {
    for (int @i@ = 0; @i@ < @n@ - 1; @i@++) {
        for (int @j@ = 0; @j@ < @n@ - @i@ - 1; @j@++) {
            if (@arr@[@j@] > @arr@[@j@ + 1]) {
                int @tmp@ = @arr@[@j@];
                @arr@[@j@] = @arr@[@j@ + 1];
                @arr@[@j@ + 1] = @tmp@;
            }
        }
    }
}
""", """void @fn@(int @arr@[], int @n@) {
    int @swapped@;
    do {
        @swapped@ = 0;
        for (int @i@ = 0; @i@ < @n@ - 1; @i@++) {
            if (@arr@[@i@] > @arr@[@i@ + 1]) {
                int @tmp@ = @arr@[@i@];
                @arr@[@i@] = @arr@[@i@ + 1];
                @arr@[@i@ + 1] = @tmp@;
                @swapped@ = 1;
            }
        }
    } while (@swapped@);
}
"""], MAIN_SORT)

fam("selection_sort", "sorting", [
    """void @fn@(int @arr@[], int @n@) {
    for (int @i@ = 0; @i@ < @n@ - 1; @i@++) {
        int @minIdx@ = @i@;
        for (int @j@ = @i@ + 1; @j@ < @n@; @j@++) {
            if (@arr@[@j@] < @arr@[@minIdx@]) {
                @minIdx@ = @j@;
            }
        }
        int @tmp@ = @arr@[@i@];
        @arr@[@i@] = @arr@[@minIdx@];
        @arr@[@minIdx@] = @tmp@;
    }
}
"""], MAIN_SORT)

fam("insertion_sort", "sorting", [
    """void @fn@(int @arr@[], int @n@) {
    for (int @i@ = 1; @i@ < @n@; @i@++) {
        int @key@ = @arr@[@i@];
        int @j@ = @i@ - 1;
        while (@j@ >= 0 && @arr@[@j@] > @key@) {
            @arr@[@j@ + 1] = @arr@[@j@];
            @j@--;
        }
        @arr@[@j@ + 1] = @key@;
    }
}
"""], MAIN_SORT)

# ------------------------------------------------------------------ searching
SIG_SEARCH = "int @fn@(int @arr@[], int @n@, int @target@)"
_SEARCH_MAIN = _main_arr("@fn@(@data@, N, 9)", "1, 4, 6, 9, 12, 15")

fam("linear_search", "searching", [
    SIG_SEARCH + """ {
    for (int @i@ = 0; @i@ < @n@; @i@++) {
        if (@arr@[@i@] == @target@) {
            return @i@;
        }
    }
    return -1;
}
""", SIG_SEARCH + """ {
    int @i@ = 0;
    while (@i@ < @n@) {
        if (@arr@[@i@] == @target@) {
            return @i@;
        }
        @i@++;
    }
    return -1;
}
"""], _SEARCH_MAIN)

fam("binary_search", "searching", [
    SIG_SEARCH + """ {
    int @low@ = 0;
    int @high@ = @n@ - 1;
    while (@low@ <= @high@) {
        int @mid@ = @low@ + (@high@ - @low@) / 2;
        if (@arr@[@mid@] == @target@) {
            return @mid@;
        } else if (@arr@[@mid@] < @target@) {
            @low@ = @mid@ + 1;
        } else {
            @high@ = @mid@ - 1;
        }
    }
    return -1;
}
""", """int @fn@Rec(int @arr@[], int @low@, int @high@, int @target@) {
    if (@low@ > @high@) {
        return -1;
    }
    int @mid@ = @low@ + (@high@ - @low@) / 2;
    if (@arr@[@mid@] == @target@) {
        return @mid@;
    }
    if (@arr@[@mid@] < @target@) {
        return @fn@Rec(@arr@, @mid@ + 1, @high@, @target@);
    }
    return @fn@Rec(@arr@, @low@, @mid@ - 1, @target@);
}

""" + SIG_SEARCH + """ {
    return @fn@Rec(@arr@, 0, @n@ - 1, @target@);
}
"""], _SEARCH_MAIN)

# ------------------------------------------------------------------ arith2
fam("add2", "arith2", [
    "int @fn@(int @a@, int @b@) {\n    return @a@ + @b@;\n}\n",
    "int @fn@(int @a@, int @b@) {\n    int @acc@ = @a@ + @b@;\n    return @acc@;\n}\n",
], _main_value("@fn@(3, 4)"))

fam("mul2", "arith2", [
    "int @fn@(int @a@, int @b@) {\n    return @a@ * @b@;\n}\n",
    "int @fn@(int @a@, int @b@) {\n    int @acc@ = @a@ * @b@;\n    return @acc@;\n}\n",
], _main_value("@fn@(6, 7)"))

# ------------------------------------------------------------------ compare2
fam("max_of_two", "compare2", [
    """int @fn@(int @a@, int @b@) {
    if (@a@ > @b@) {
        return @a@;
    } else {
        return @b@;
    }
}
""", "int @fn@(int @a@, int @b@) {\n    return @a@ > @b@ ? @a@ : @b@;\n}\n"],
    _main_value("@fn@(12, 30)"))

fam("min_of_two", "compare2", [
    """int @fn@(int @a@, int @b@) {
    if (@a@ < @b@) {
        return @a@;
    } else {
        return @b@;
    }
}
""", "int @fn@(int @a@, int @b@) {\n    return @a@ < @b@ ? @a@ : @b@;\n}\n"],
    _main_value("@fn@(12, 30)"))

# ------------------------------------------------------------------ branching
fam("leap_year", "branching", [
    """int @fn@(int @year@) {
    if ((@year@ % 4 == 0 && @year@ % 100 != 0) || @year@ % 400 == 0) {
        return 1;
    }
    return 0;
}
""", "int @fn@(int @year@) {\n    return (@year@ % 4 == 0 && @year@ % 100 != 0) || @year@ % 400 == 0;\n}\n"],
    _main_value("@fn@(2024)"))

fam("fizzbuzz", "branching", [
    """void @fn@(int @n@) {
    for (int @i@ = 1; @i@ <= @n@; @i@++) {
        if (@i@ % 15 == 0) {
            printf("FizzBuzz\\n");
        } else if (@i@ % 3 == 0) {
            printf("Fizz\\n");
        } else if (@i@ % 5 == 0) {
            printf("Buzz\\n");
        } else {
            printf("%d\\n", @i@);
        }
    }
}
""", """void @fn@(int @n@) {
    int @i@ = 1;
    while (@i@ <= @n@) {
        if (@i@ % 15 == 0) {
            printf("FizzBuzz\\n");
        } else if (@i@ % 3 == 0) {
            printf("Fizz\\n");
        } else if (@i@ % 5 == 0) {
            printf("Buzz\\n");
        } else {
            printf("%d\\n", @i@);
        }
        @i@++;
    }
}
"""], "int main() {\n    @fn@(15);\n    return 0;\n}\n")

# ------------------------------------------------------------------ strings
fam("str_length", "strings", [
    """int @fn@(char @s@[]) {
    int @len@ = 0;
    while (@s@[@len@] != '\\0') {
        @len@++;
    }
    return @len@;
}
""", """int @fn@(char @s@[]) {
    int @len@ = 0;
    for (int @i@ = 0; @s@[@i@] != '\\0'; @i@++) {
        @len@++;
    }
    return @len@;
}
"""], _main_value("@fn@(@text@)", decl='    char @text@[] = "hello world";\n'))

fam("count_char", "strings", [
    """int @fn@(char @s@[], char @ch@) {
    int @count@ = 0;
    for (int @i@ = 0; @s@[@i@] != '\\0'; @i@++) {
        if (@s@[@i@] == @ch@) {
            @count@++;
        }
    }
    return @count@;
}
"""], _main_value("@fn@(@text@, 'o')", decl='    char @text@[] = "hello world";\n'))

fam("reverse_string", "strings", [
    """void @fn@(char @s@[]) {
    int @len@ = 0;
    while (@s@[@len@] != '\\0') {
        @len@++;
    }
    for (int @i@ = 0; @i@ < @len@ / 2; @i@++) {
        char @tmp@ = @s@[@i@];
        @s@[@i@] = @s@[@len@ - 1 - @i@];
        @s@[@len@ - 1 - @i@] = @tmp@;
    }
}
"""], "int main() {\n    char @text@[] = \"hello world\";\n    @fn@(@text@);\n"
      "    printf(\"%s\\n\", @text@);\n    return 0;\n}\n")

# ------------------------------------------------------------------ matrix
_MAT_MAIN = ("int main() {\n    int @data@[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};\n"
             "    %s\n    return 0;\n}\n")
fam("matrix_sum", "matrix", [
    """int @fn@(int @m@[3][3]) {
    int @acc@ = 0;
    for (int @i@ = 0; @i@ < 3; @i@++) {
        for (int @j@ = 0; @j@ < 3; @j@++) {
            @acc@ += @m@[@i@][@j@];
        }
    }
    return @acc@;
}
"""], _MAT_MAIN % 'printf("%d\\n", @fn@(@data@));')

fam("matrix_transpose", "matrix", [
    """void @fn@(int @m@[3][3]) {
    for (int @i@ = 0; @i@ < 3; @i@++) {
        for (int @j@ = @i@ + 1; @j@ < 3; @j@++) {
            int @tmp@ = @m@[@i@][@j@];
            @m@[@i@][@j@] = @m@[@j@][@i@];
            @m@[@j@][@i@] = @tmp@;
        }
    }
}
"""], _MAT_MAIN % '@fn@(@data@);\n    printf("%d\\n", @data@[0][1]);')

# ------------------------------------------------------------------ pattern
fam("triangle", "pattern", [
    """void @fn@(int @rows@) {
    for (int @i@ = 1; @i@ <= @rows@; @i@++) {
        for (int @j@ = 1; @j@ <= @i@; @j@++) {
            printf("*");
        }
        printf("\\n");
    }
}
""", """void @fn@(int @rows@) {
    int @i@ = 1;
    while (@i@ <= @rows@) {
        int @j@ = 1;
        while (@j@ <= @i@) {
            printf("*");
            @j@++;
        }
        printf("\\n");
        @i@++;
    }
}
"""], "int main() {\n    @fn@(5);\n    return 0;\n}\n")

HOLDOUT = ["selection_sort", "gcd", "count_char", "fizzbuzz", "min_of_two"]
TRAIN_FAMILIES = [f for f in FAMILIES if f not in HOLDOUT]
GROUPS = {}
for _name, _f in FAMILIES.items():
    GROUPS.setdefault(_f["group"], []).append(_name)


# ------------------------------------------------------------------ rendering
def _fill(template, rng):
    mapping, used = {}, set()

    def sub(m):
        key = m.group(1)
        if key not in mapping:
            pool = POOLS[key]
            choices = [c for c in pool if c not in used] or [key + str(len(used))]
            mapping[key] = rng.choice(choices)
            used.add(mapping[key])
        return mapping[key]

    return re.sub(r"@(\w+)@", sub, template)


def _restyle(src, rng):
    lines = src.split("\n")
    out = []
    allman = rng.random() < 0.3
    unit = rng.choice(["  ", "    ", "\t", "        "])
    for line in lines:
        indent_len = len(line) - len(line.lstrip(" "))
        body = line.strip(" ")
        indent = unit * (indent_len // 4)
        if allman and body.endswith("{") and body != "{":
            out.append(indent + body[:-1].rstrip())
            out.append(indent + "{")
        else:
            out.append(indent + body)
        if body and not body.startswith("#") and rng.random() < 0.12:
            out[-1] += rng.choice(["  // step", " /* note */", "  // TODO: check", " // helper"])
        if rng.random() < 0.08:
            out.append("")
    if rng.random() < 0.5:
        out.insert(0, rng.choice(["// Program: %s" % rng.choice(["demo", "assignment 3", "lab"]),
                                  "/* auto-generated sample\n   multi-line comment */"]))
    return "\n".join(out)


def render(family, variant, rng, with_main=None, light_edit=False, restyle=True):
    """Return C source for family/variant with random identifiers, layout and comments."""
    f = FAMILIES[family]
    if with_main is None:
        with_main = rng.random() < 0.75
    body = f["variants"][variant]
    code = HEADER + body
    if with_main:
        main = f["main"]
        if light_edit:
            main = main.replace("int main() {\n", "int main() {\n    int @unused@ = 0;\n", 1)
        code += "\n" + main
    code = _fill(code, rng)
    return _restyle(code, rng) if restyle else code


# ------------------------------------------------------------------ pair generation
def _pick_family(rng, pool):
    return rng.choice(pool)


def make_pair(rng, label, pool, must_include=None):
    """Return (source_a, source_b, fam_a, fam_b, var_a, var_b)."""
    first_pool = must_include or pool
    for _ in range(100):
        fa = _pick_family(rng, first_pool)
        ga = FAMILIES[fa]["group"]
        if label == HIGH:
            fb = fa
            va = rng.randrange(len(FAMILIES[fa]["variants"]))
            vb = va
        elif label == MODERATE:
            want_variant = rng.random() < 0.5
            nv = len(FAMILIES[fa]["variants"])
            mates = [x for x in GROUPS[ga] if x != fa]
            if (want_variant and nv > 1) or not mates:
                if nv < 2:
                    continue
                fb = fa
                va = rng.randrange(nv)
                vb = rng.choice([v for v in range(nv) if v != va])
            else:
                fb = rng.choice(mates)
                va = rng.randrange(nv)
                vb = rng.randrange(len(FAMILIES[fb]["variants"]))
        else:
            others = [x for x in pool + (must_include or []) if FAMILIES[x]["group"] != ga]
            if not others:
                continue
            fb = rng.choice(others)
            va = rng.randrange(len(FAMILIES[fa]["variants"]))
            vb = rng.randrange(len(FAMILIES[fb]["variants"]))
        break
    else:
        raise RuntimeError("could not build a pair for label %s" % label)
    if rng.random() < 0.5:
        fa, fb, va, vb = fb, fa, vb, va
    sa = render(fa, va, rng)
    sb = render(fb, vb, rng, light_edit=(label == HIGH and rng.random() < 0.4))
    return sa, sb, fa, fb, va, vb


def generate_pairs(n, seed, pool=None, must_include=None):
    """Balanced list of (src_a, src_b, label, fam_a, fam_b)."""
    rng = random.Random(seed)
    pool = list(pool or TRAIN_FAMILIES)
    out = []
    labels = [LOW, MODERATE, HIGH]
    for k in range(n):
        label = labels[k % 3]
        sa, sb, fa, fb, _, _ = make_pair(rng, label, pool, must_include)
        out.append((sa, sb, label, fa, fb))
    rng.shuffle(out)
    return out
