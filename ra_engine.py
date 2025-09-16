from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Set, Iterable


# ChatGPT for debugging and handling Error or Exception Case
class RelAlgError(Exception):
    """Base error for relational algebra engine."""

class SchemaError(RelAlgError):
    pass

class ParseError(RelAlgError):
    pass


# Create table and schema. allow user to type in there own table 
@dataclass(frozen=True)
class Relation:
    """A simple set-based relation (duplicates removed)."""
    name: str
    schema: List[str]
    rows: List[Tuple[Any, ...]] = field(default_factory=list)

    def __post_init__(self):
        for r in self.rows:
            if len(r) != len(self.schema):
                raise SchemaError(f"Row length {len(r)} does not match schema length {len(self.schema)} for relation {self.name}")
        deduped = list(dict.fromkeys(self.rows))
        object.__setattr__(self, "rows", deduped)
        object.__setattr__(self, "schema", list(self.schema))

    @classmethod
    def from_rows(cls, name: str, schema: Iterable[str], rows: Iterable[Iterable[Any]]) -> "Relation":
        return cls(name, list(schema), [tuple(r) for r in rows])

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "schema": list(self.schema), "rows": [list(r) for r in self.rows]}

    def to_csv(self, delimiter: str = ",") -> str:
        out = [delimiter.join(self.schema)]
        for r in self.rows:
            out.append(delimiter.join(map(_to_str, r)))
        return "\n".join(out)

    def pretty(self, max_width: int = 24) -> str:
        cols = self.schema
        data = [cols] + [list(map(_to_str, r)) for r in self.rows]
        widths = [0] * len(cols)
        for row in data:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        widths = [min(w, max_width) for w in widths]

        def fmt(row):
            cells = []
            for i, cell in enumerate(row):
                s = str(cell)
                if len(s) > widths[i]:
                    s = s[: max(0, widths[i] - 1)] + "…"
                cells.append(s.ljust(widths[i]))
            return " | ".join(cells)

        lines = [fmt(cols), "-+-".join("-" * w for w in widths)]
        for r in self.rows:
            lines.append(fmt(list(map(_to_str, r))))
        return "\n".join(lines)

def _to_str(x: Any) -> str:
    return x if isinstance(x, str) else str(x)

def _assert(cond: bool, msg: str, err=RelAlgError):
    if not cond:
        raise err(msg)

########################
# Core Operations
########################

# pi relation - 1. check for the schema in the table, 2. the postion of index in rel.schema will be insert into idxs
# Schema = ["EID","Name","Age"], attrs = ["Name","Age"] →
# idxs = [1, 2].
def project(rel: Relation, attrs: List[str], name: Optional[str] = None) -> Relation:
    idxs = []
    for a in attrs:
        _assert(a in rel.schema, f"Project: attribute {a!r} not in schema {rel.schema}", SchemaError)
        idxs.append(rel.schema.index(a))
    new_rows = [tuple(row[i] for i in idxs) for row in rel.rows]
    return Relation(name or rel.name, attrs, new_rows)

# sigma relation - 1. create a condition checking
#  Schema: ["EID","Name","Age"]
#  Condition: Age > 30 AND Name != "Alice"
#  Take Age compare with 30, Name compare with “Alice”, then  AND.
def select(rel: Relation, condition: str, name: Optional[str] = None) -> Relation:
    evaluator = ConditionEvaluator(rel.schema, condition)
    new_rows = [row for row in rel.rows if evaluator.eval_row(row)]
    return Relation(name or rel.name, rel.schema, new_rows)


# Join relation
# 1.  Determine Join Condition
#     If on is provided, it parses the condition (e.g. "EID = EmpID AND Dept = DeptID") into pairs of columns to match.
#     If on is not provided, it performs a natural join on all columns with the same name.
#     If there are no common columns, it performs a Cartesian product (every row from left combined with every row from right).
# 2.  Build Output Schema
#     Starts with all columns from the left relation.
#     Adds columns from the right relation, skipping duplicates for join columns and renaming if there are naming conflicts.
# 3.  Perform Join Efficiently (Hash Join)
#     Picks the smaller relation as the “build side” and creates a hash table using the join keys.
#     Loops through the “probe side” and finds matching rows in the hash table.
#     Combines matching rows into new result rows.
# 4.  Return New Relation
#     Creates and returns a new Relation with the combined schema and rows.
#     Duplicate rows (if any) are automatically removed because Relation enforces set semantics.
def join(left: Relation, right: Relation, on: Optional[str] = None, name: Optional[str] = None) -> Relation:
    if on is None or not on.strip():
        common = [c for c in left.schema if c in right.schema]
        on_pairs = [(c, c) for c in common]
    else:
        on_pairs = _parse_on_pairs(on)

    out_schema, right_index_map = _join_output_schema(left, right, on_pairs)

    if not on_pairs:
        out_rows = []
        for lr in left.rows:
            for rr in right.rows:
                out_rows.append(_concat_rows(lr, rr, right_index_map))
        return Relation(name or f"{left.name}⋈{right.name}", out_schema, out_rows)

    left_key_idx = [left.schema.index(l) for (l, _) in on_pairs]
    right_key_idx = [right.schema.index(r) for (_, r) in on_pairs]

    build_is_left = len(left.rows) <= len(right.rows)
    build_rel, probe_rel = (left, right) if build_is_left else (right, left)
    build_idx = left_key_idx if build_is_left else right_key_idx
    probe_idx = right_key_idx if build_is_left else left_key_idx

    hash_table: Dict[Tuple[Any, ...], List[Tuple[Any, ...]]] = {}
    for row in build_rel.rows:
        key = tuple(row[i] for i in build_idx)
        hash_table.setdefault(key, []).append(row)

    out_rows = []
    if build_is_left:
        for prow in probe_rel.rows:
            key = tuple(prow[i] for i in probe_idx)
            for lrow in hash_table.get(key, []):
                out_rows.append(_concat_rows(lrow, prow, right_index_map))
    else:
        for prow in probe_rel.rows:
            key = tuple(prow[i] for i in probe_idx)
            for rrow in hash_table.get(key, []):
                out_rows.append(_concat_rows(prow, rrow, right_index_map))

    return Relation(name or f"{left.name}⋈{right.name}", out_schema, out_rows)

# It starts with all columns from the left relation.
# For each column in the right relation:
# If it is a join column:
# Skip it if it has the same name as the left column (avoid duplicates).
# Keep it if its name is different (both columns are needed).
# If it is not a join column:
# Add it to the schema, but rename it as RightTable.Column if the name already exists in the left schema (to avoid ambiguity).
def _join_output_schema(left: Relation, right: Relation, on_pairs: List[Tuple[str, str]]):
    out_schema = list(left.schema)
    paired_right = {r for (_, r) in on_pairs}
    left_partner_for_right = {r: l for (l, r) in on_pairs}

    right_index_map: List[Optional[int]] = []
    for j, col in enumerate(right.schema):
        if col in paired_right:
            left_name = left_partner_for_right[col]
            if left_name != col:
                out_schema.append(col)
                right_index_map.append(j)
            else:
                right_index_map.append(None)
        else:
            if col in out_schema:
                out_schema.append(f"{right.name}.{col}")
            else:
                out_schema.append(col)
            right_index_map.append(j)
    return out_schema, right_index_map

# helper to merge 2 rows toghether to create new row
def _concat_rows(left_row: Tuple[Any, ...], right_row: Tuple[Any, ...], right_index_map: List[Optional[int]]):
    vals = list(left_row)
    for idx in right_index_map:
        if idx is not None:
            vals.append(right_row[idx])
    return tuple(vals)

def union(r1: Relation, r2: Relation, name: Optional[str] = None) -> Relation:
    _assert(r1.schema == r2.schema, f"Union: schema mismatch {r1.schema} vs {r2.schema}", SchemaError)
    return Relation(name or f"{r1.name}⋃{r2.name}", r1.schema, list(r1.rows) + list(r2.rows))

def intersect(r1: Relation, r2: Relation, name: Optional[str] = None) -> Relation:
    _assert(r1.schema == r2.schema, f"Intersect: schema mismatch {r1.schema} vs {r2.schema}", SchemaError)
    set2 = set(r2.rows)
    return Relation(name or f"{r1.name}∩{r2.name}", r1.schema, [r for r in r1.rows if r in set2])

def minus(r1: Relation, r2: Relation, name: Optional[str] = None) -> Relation:
    _assert(r1.schema == r2.schema, f"Minus: schema mismatch {r1.schema} vs {r2.schema}", SchemaError)
    set2 = set(r2.rows)
    return Relation(name or f"{r1.name}−{r2.name}", r1.schema, [r for r in r1.rows if r not in set2])

#############################
# Condition Evaluator (σ)
#############################

class ConditionEvaluator:
    """Parse and evaluate boolean expressions over a row.
    Supports: =, !=, >, >=, <, <= with AND/OR/NOT and parentheses.
    Identifiers must be attribute names in the relation schema.
    """
    def __init__(self, schema: List[str], expr: str):
        self.schema = schema
        self.expr = expr.strip()
        self.rpn = self._to_rpn(self._tokenize(self.expr))

    def eval_row(self, row: Tuple[Any, ...]) -> bool:
        stack: List[Any] = []
        for ttype, tval in self.rpn:
            if ttype in ("NUMBER", "STRING"):
                stack.append(tval)
            elif ttype == "IDENT":
                _assert(tval in self.schema, f"Unknown attribute {tval!r} in condition", SchemaError)
                stack.append(row[self.schema.index(tval)])
            elif ttype == "OP":
                if tval in ("=", "!=", ">", ">=", "<", "<="):
                    b = stack.pop(); a = stack.pop()
                    stack.append(_compare(a, b, tval))
                elif tval in ("AND", "OR"):
                    b = bool(stack.pop()); a = bool(stack.pop())
                    stack.append((a and b) if tval == "AND" else (a or b))
                elif tval == "NOT":
                    a = bool(stack.pop()); stack.append(not a)
                else:
                    raise ParseError(f"Unknown operator {tval}")
            else:
                raise ParseError(f"Bad token {ttype}:{tval}")
        _assert(len(stack) == 1, "Condition evaluation error", ParseError)
        return bool(stack[0])

    def _tokenize(self, s: str):
        tokens = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isspace():
                i += 1; continue
            if c in "(),":
                tokens.append(("PUNC", c)); i += 1; continue
            if c in "<>=":
                if i+1 < len(s) and s[i:i+2] in ("<=", ">=", "!="):
                    tokens.append(("OP", s[i:i+2])); i += 2; continue
                tokens.append(("OP", c)); i += 1; continue
            if c == "!":
                if i+1 < len(s) and s[i+1] == "=":
                    tokens.append(("OP", "!=")); i += 2; continue
                raise ParseError("Unexpected '!' without '='")
            if c in "'\"":
                quote = c; i += 1; start = i
                buf = []
                while i < len(s) and s[i] != quote:
                    if s[i] == "\\" and i+1 < len(s):
                        buf.append(s[i+1]); i += 2
                    else:
                        buf.append(s[i]); i += 1
                if i >= len(s):
                    raise ParseError("Unclosed string literal")
                tokens.append(("STRING", "".join(buf))); i += 1; continue
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
            if m:
                ident = m.group(0); up = ident.upper()
                tokens.append(("OP", up) if up in ("AND", "OR", "NOT") else ("IDENT", ident))
                i += len(ident); continue
            m = re.match(r"-?(?:\d+\.\d*|\d*\.\d+|\d+)", s[i:])
            if m:
                num = m.group(0); tokens.append(("NUMBER", float(num) if "." in num else int(num))); i += len(num); continue
            raise ParseError(f"Unexpected character {s[i]!r} in condition")
        return tokens

    def _to_rpn(self, tokens):
        prec = {"NOT": 3, "=": 2, "!=": 2, ">": 2, ">=": 2, "<": 2, "<=": 2, "AND": 1, "OR": 0}
        out = []; ops = []
        for ttype, tval in tokens:
            if ttype in ("NUMBER", "STRING", "IDENT"):
                out.append((ttype, tval))
            elif ttype == "OP":
                if tval == "NOT":
                    while ops and ops[-1][0] == "OP" and prec.get(ops[-1][1], -1) > prec["NOT"]:
                        out.append(ops.pop())
                    ops.append((ttype, tval))
                else:
                    while ops and ops[-1][0] == "OP" and prec.get(ops[-1][1], -1) >= prec.get(tval, -1):
                        out.append(ops.pop())
                    ops.append((ttype, tval))
            elif ttype == "PUNC" and tval == "(":
                ops.append((ttype, tval))
            elif ttype == "PUNC" and tval == ")":
                while ops and not (ops[-1][0] == "PUNC" and ops[-1][1] == "("):
                    out.append(ops.pop())
                if not ops:
                    raise ParseError("Mismatched parentheses")
                ops.pop()
            else:
                raise ParseError(f"Unexpected token {ttype}:{tval}")
        while ops:
            if ops[-1][0] == "PUNC":
                raise ParseError("Mismatched parentheses")
            out.append(ops.pop())
        return out

def _compare(a, b, op: str) -> bool:
    if op == "=": return a == b
    if op == "!=": return a != b
    if op == ">": return a > b
    if op == ">=": return a >= b
    if op == "<": return a < b
    if op == "<=": return a <= b
    raise ParseError(f"Unknown comparison {op}")

#############################
# Parsing Relations
#############################

RELATION_DEF_RE = re.compile(
    r"""([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*=\s*\{\s*(.*?)\s*\}""",
    re.DOTALL,
)

def parse_relations(input_text: str) -> Dict[str, Relation]:
    """Parse one or more relation blocks into Relation objects."""
    relations: Dict[str, Relation] = {}
    for m in RELATION_DEF_RE.finditer(input_text):
        name = m.group(1).strip()
        schema_raw = m.group(2).strip()
        rows_raw = m.group(3).strip()

        schema = [c.strip() for c in schema_raw.split(",") if c.strip()]
        rows: List[Tuple[Any, ...]] = []

        if rows_raw:
            for line in rows_raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in split_csv_like(line)]
                if len(parts) != len(schema):
                    raise ParseError(f"Row {line!r} has {len(parts)} values but schema has {len(schema)} columns for relation {name}")
                rows.append(tuple(_coerce_literal(p) for p in parts))

        relations[name] = Relation(name, schema, rows)

    if not relations:
        raise ParseError("No relation definitions found. Check your input format.")
    return relations

def split_csv_like(line: str) -> List[str]:
    """Split a single line by commas, respecting quotes and escapes."""
    out: List[str] = []
    cur: List[str] = []
    i = 0; quote: Optional[str] = None
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\" and i + 1 < len(line):
                cur.append(line[i:i+2]); i += 2; continue
            if ch == quote:
                cur.append(ch); i += 1; quote = None; continue
            cur.append(ch); i += 1; continue
        else:
            if ch in ("'", '"'):
                quote = ch; cur.append(ch); i += 1; continue
            if ch == ",":
                out.append("".join(cur).strip()); cur = []; i += 1; continue
            cur.append(ch); i += 1; continue
    if cur: out.append("".join(cur).strip())
    return out

def _coerce_literal(s: str):
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    if re.fullmatch(r"-?\d+", s): return int(s)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", s): return float(s)
    return s

#############################
# Parsing RA Queries
#############################

class ASTNode:
    def evaluate(self, env: Dict[str, Relation]) -> Relation:
        raise NotImplementedError()

class NameNode(ASTNode):
    def __init__(self, name: str):
        self.name = name
    def evaluate(self, env):
        _assert(self.name in env, f"Unknown relation {self.name}", ParseError)
        return Relation(self.name, env[self.name].schema, env[self.name].rows)
    def __repr__(self): return f"Name({self.name})"

class SelectNode(ASTNode):
    def __init__(self, condition: str, child: ASTNode):
        self.condition = condition; self.child = child
    def evaluate(self, env): return select(self.child.evaluate(env), self.condition)
    def __repr__(self): return f"σ[{self.condition}]({self.child})"

class ProjectNode(ASTNode):
    def __init__(self, attrs: List[str], child: ASTNode):
        self.attrs = attrs; self.child = child
    def evaluate(self, env): return project(self.child.evaluate(env), self.attrs)
    def __repr__(self): return f"π{self.attrs}({self.child})"

class JoinNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode, on: Optional[str] = None):
        self.left = left; self.right = right; self.on = on
    def evaluate(self, env): return join(self.left.evaluate(env), self.right.evaluate(env), self.on)
    def __repr__(self): return f"({self.left})⋈[{self.on}]({self.right})"

class UnionNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left; self.right = right
    def evaluate(self, env): return union(self.left.evaluate(env), self.right.evaluate(env))
    def __repr__(self): return f"({self.left})⋃({self.right})"

class IntersectNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left; self.right = right
    def evaluate(self, env): return intersect(self.left.evaluate(env), self.right.evaluate(env))
    def __repr__(self): return f"({self.left})∩({self.right})"

class MinusNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left; self.right = right
    def evaluate(self, env): return minus(self.left.evaluate(env), self.right.evaluate(env))
    def __repr__(self): return f"({self.left})−({self.right})"

def parse_query(q: str) -> ASTNode:
    return RAParser(q.strip()).parse_expression()

class RAParser:
    def __init__(self, s: str):
        self.s = s
        self.tokens = self._tokenize(s)
        self.pos = 0

    def _peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", "", "")
    def _eat(self, kind=None, value=None):
        tok = self._peek()
        if kind is not None and tok[0] != kind: raise ParseError(f"Expected {kind} but found {tok}")
        if value is not None and tok[1] != value: raise ParseError(f"Expected {value!r} but found {tok}")
        self.pos += 1; return tok

    def parse_expression(self) -> ASTNode:
        node = self.parse_term()
        while True:
            tok = self._peek()
            if tok[0] == "SYMBOL" and tok[1] in ("⋃", "∩", "−"):
                op = tok[1]; self._eat("SYMBOL", op)
                right = self.parse_term()
                if op == "⋃": node = UnionNode(node, right)
                elif op == "∩": node = IntersectNode(node, right)
                elif op == "−": node = MinusNode(node, right)
            else:
                break
        return node

    def parse_term(self) -> ASTNode:
        tok = self._peek()
        # select
        if tok[0] == "SYMBOL" and tok[1] == "σ":
            self._eat("SYMBOL", "σ")
            cond = self._parse_until_paren_text()
            child = self._parse_paren_expr()
            return SelectNode(cond, child)
        if tok[0] == "IDENT" and tok[1].lower() == "select":
            self._eat("IDENT")
            cond = self._parse_until_paren_text()
            child = self._parse_paren_expr()
            return SelectNode(cond, child)

        # project
        if tok[0] == "SYMBOL" and tok[1] == "π":
            self._eat("SYMBOL", "π")
            attrs = self._parse_until_paren_text().split(",")
            attrs = [a.strip() for a in attrs if a.strip()]
            _assert(attrs, "Empty projection list", ParseError)
            child = self._parse_paren_expr()
            return ProjectNode(attrs, child)
        if tok[0] == "IDENT" and tok[1].lower() == "project":
            self._eat("IDENT")
            attrs = self._parse_until_paren_text().split(",")
            attrs = [a.strip() for a in attrs if a.strip()]
            _assert(attrs, "Empty projection list", ParseError)
            child = self._parse_paren_expr()
            return ProjectNode(attrs, child)

        # join keyword
        if tok[0] == "IDENT" and tok[1].lower() == "join":
            self._eat("IDENT"); self._eat("PUNC", "(")
            left = self.parse_expression(); self._eat("PUNC", ",")
            right = self.parse_expression()
            on = None
            nxt = self._peek()
            if nxt[0] == "PUNC" and nxt[1] == ",":
                self._eat("PUNC", ",")
                key = self._eat("IDENT")[1].lower()
                if key != "on": raise ParseError('Expected on="..." in join(...)')
                self._eat("OP", "=")
                on = self._eat("STRING")[1]
            self._eat("PUNC", ")")
            return JoinNode(left, right, on)

        # name / paren / symbolic join
        if tok[0] == "IDENT":
            name = self._eat("IDENT")[1]
            node: ASTNode = NameNode(name)
            nxt = self._peek()
            if nxt[0] == "SYMBOL" and nxt[1] == "⋈":
                self._eat("SYMBOL", "⋈")
                on = None
                nxt = self._peek()
                if nxt[0] == "SUB":
                    on = nxt[1]; self._eat("SUB")
                right = self.parse_term()
                node = JoinNode(node, right, on)
            return node

        if tok[0] == "PUNC" and tok[1] == "(":
            return self._parse_paren_expr()

        raise ParseError(f"Unexpected token {tok} in term")

    def _parse_paren_expr(self) -> ASTNode:
        self._eat("PUNC", "(")
        inner = self.parse_expression()
        self._eat("PUNC", ")")
        return inner

    def _parse_until_paren_text(self) -> str:
        parts = []
        while True:
            tok = self._peek()
            if tok[0] == "PUNC" and tok[1] == "(":
                break
            if tok[0] == "EOF":
                raise ParseError("Expected '('")
            parts.append(tok[2] if len(tok) > 2 else tok[1])
            self.pos += 1
        text = "".join(parts).trim() if hasattr(str, "trim") else "".join(parts).strip()
        if not text:
            raise ParseError("Missing text before '('")
        return text

    def _tokenize(self, s: str):
        tokens = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch.isspace():
                i += 1; continue
            if ch in "(),":
                tokens.append(("PUNC", ch, ch)); i += 1; continue
            if ch == "_" and i + 1 < len(s) and s[i+1] == "{":
                j = i + 2; depth = 1; buf = []
                while j < len(s) and depth > 0:
                    if s[j] == "{": depth += 1
                    elif s[j] == "}": depth -= 1
                    if depth > 0: buf.append(s[j])
                    j += 1
                if depth != 0: raise ParseError("Unclosed _{ ... }")
                tokens.append(("SUB", "".join(buf).strip(), s[i:j])); i = j; continue
            if ch in "σπ⋈⋃∩−":
                tokens.append(("SYMBOL", ch, ch)); i += 1; continue
            if ch in "<>=":
                if i+1 < len(s) and s[i:i+2] in ("<=", ">=", "!="):
                    tokens.append(("OP", s[i:i+2], s[i:i+2])); i += 2; continue
                tokens.append(("OP", ch, ch)); i += 1; continue
            if ch == "!":
                if i+1 < len(s) and s[i+1] == "=":
                    tokens.append(("OP","!=", "!=")); i += 2; continue
                raise ParseError("Unexpected '!'")
            if ch in "'\"":
                quote = ch; j = i + 1; buf = []
                while j < len(s) and s[j] != quote:
                    if s[j] == "\\" and j+1 < len(s): buf.append(s[j+1]); j += 2
                    else: buf.append(s[j]); j += 1
                if j >= len(s): raise ParseError("Unclosed string literal")
                tokens.append(("STRING", "".join(buf), s[i:j+1])); i = j + 1; continue
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
            if m:
                ident = m.group(0); tokens.append(("IDENT", ident, ident)); i += len(ident); continue
            m = re.match(r"-?(?:\d+\.\d*|\d*\.\d+|\d+)", s[i:])
            if m:
                num = m.group(0); tokens.append(("NUMBER", num, num)); i += len(num); continue
            raise ParseError(f"Bad character {ch!r} in query")
        tokens.append(("EOF", "", ""))
        return tokens

#############################
# Join ON parser
#############################

def _parse_on_pairs(on: str) -> List[Tuple[str, str]]:
    parts = [p.strip() for p in on.replace("and", "AND").split("AND")]
    pairs: List[Tuple[str, str]] = []
    for p in parts:
        if not p: continue
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*=\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*$", p)
        if not m:
            raise ParseError(f"Invalid join ON segment: {p!r}")
        left = m.group(1).split(".")[-1]
        right = m.group(2).split(".")[-1]
        pairs.append((left, right))
    return pairs

#############################
# Runner utilities (public)
#############################

def run(relations_text: str, query_text: str) -> Relation:
    env = parse_relations(relations_text)
    ast = parse_query(query_text)
    return ast.evaluate(env)
