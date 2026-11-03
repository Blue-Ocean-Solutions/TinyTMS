"""The voice parser rebuilt from its source with one word deleted from one word list (ORDER Nube-9, item 8).

own_words(name) lists the words a list's own source literal holds (a derived list such as _NOT_FILLER holds only the
words written in it, not those it takes from other lists). without(name, word) execs tinytms/adapters.py again with
that word deleted from that literal and returns the new module's voice_from_transcript: everything computed from the
list afterwards (unions, regexes) is computed without the word, as if it had been deleted by hand.
"""

from __future__ import annotations

import ast
import copy
import functools
from pathlib import Path

from tinytms import adapters

SRC = Path(adapters.__file__).read_text(encoding="utf-8")
TREE = ast.parse(SRC)
_VOICE = (SRC.index("# --- voice"), SRC.index("# --- fixtures"))
# Module-level names in the voice section that are not word lists: compiled regexes (_PHONE and _INTERSTATE match
# digit shapes, not words), and _DIGIT_OF (built only from the number tables, which have their own cases).
NOT_LISTS = frozenset({"_UNIT", "_TOKEN", "_THOUSANDS", "_GROUPED", "_DIGIT_OF", "_PHONE", "_INTERSTATE"})


def _assign(name: str) -> ast.Assign:
    for node in TREE.body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == [name]:
            return node
    raise KeyError(name)


def list_names() -> list[str]:
    """Every module-level constant of the voice section: its word lists, and NOT_LISTS."""
    out = []
    for node in TREE.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, off = node.targets[0].id, node.col_offset + _line_offset(node.lineno)
            if _VOICE[0] < off < _VOICE[1] and name.startswith("_") and name[1:].replace("_", "").isupper():
                out.append(name)
    return out


def _line_offset(lineno: int) -> int:
    return sum(len(x) for x in SRC.splitlines(keepends=True)[:lineno - 1])


def _split_receivers(node: ast.AST) -> set[int]:
    """ids of the string constants that are split into words ('kilos kilo kg'.split())."""
    return {id(n.func.value) for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "split" and isinstance(n.func.value, ast.Constant)}


def own_words(name: str) -> list[str]:
    """The words written in the list's own source literal, as its cases name them ('mejor dicho', 'doble ve')."""
    node = _assign(name).value
    if name == "_UNITS":
        return node.value.split("|")
    if name == "_ES_HUNDREDS":
        return list(adapters._ES_HUNDREDS)
    words: list[str] = []
    split = _split_receivers(node)
    for n in ast.walk(node):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant):
                    words.append(k.value)
                elif isinstance(k, ast.Tuple):
                    words.append(" ".join(e.value for e in k.elts))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) in split:
            words += n.value.split()
        elif isinstance(n, (ast.Set, ast.Tuple, ast.List)) and not _is_dict_key(node, n):
            words += [e.value for e in n.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return words


def _is_dict_key(root: ast.AST, n: ast.AST) -> bool:
    return any(isinstance(d, ast.Dict) and any(k is n for k in d.keys) for d in ast.walk(root))


class _Drop(ast.NodeTransformer):
    def __init__(self, word: str, split: set[int]):
        self.word, self.split, self.dropped = word, split, 0

    def visit_Constant(self, n: ast.Constant) -> ast.AST:
        if isinstance(n.value, str) and id(n) in self.split and self.word in n.value.split():
            self.dropped += 1
            return ast.copy_location(ast.Constant(" ".join(w for w in n.value.split() if w != self.word)), n)
        return n

    def _elts(self, n):
        keep = [e for e in n.elts if not (isinstance(e, ast.Constant) and e.value == self.word)]
        self.dropped += len(n.elts) - len(keep)
        n.elts = keep
        return self.generic_visit(n)

    visit_Set = visit_Tuple = visit_List = _elts

    def visit_Dict(self, n: ast.Dict) -> ast.AST:
        pairs = [(k, v) for k, v in zip(n.keys, n.values) if not (
            (isinstance(k, ast.Constant) and k.value == self.word)
            or (isinstance(k, ast.Tuple) and " ".join(e.value for e in k.elts) == self.word))]
        self.dropped += len(n.keys) - len(pairs)
        n.keys, n.values = [k for k, _ in pairs], [v for _, v in pairs]
        return self.generic_visit(n)


_FLAGS = __import__("__future__").annotations.compiler_flag
_CODES = [compile(ast.Module([stmt], []), str(Path(adapters.__file__)), "exec", flags=_FLAGS, dont_inherit=True)
          for stmt in TREE.body]


@functools.lru_cache(maxsize=None)
def without(name: str, word: str):
    """voice_from_transcript of the parser with word deleted from list name. Every statement of the module but the
    list's own is the one compiled once from the source."""
    index = TREE.body.index(_assign(name))
    target = copy.deepcopy(TREE.body[index])
    extra = []
    if name == "_UNITS":
        alts = target.value.value.split("|")
        assert word in alts, word
        target.value = ast.Constant("|".join(a for a in alts if a != word))
    elif name == "_ES_HUNDREDS":  # its keys are built by a comprehension: drop the key right after the assignment
        extra = ast.parse(f"_ES_HUNDREDS.pop({word!r})").body
    else:
        drop = _Drop(word, _split_receivers(target.value))
        drop.visit(target.value)
        assert drop.dropped, (name, word)
    mutated = ast.fix_missing_locations(ast.Module([target, *extra], []))
    code = compile(mutated, str(Path(adapters.__file__)), "exec", flags=_FLAGS, dont_inherit=True)
    ns = {"__name__": "tinytms._adapters_without", "__package__": "tinytms"}
    for k, c in enumerate(_CODES):
        exec(code if k == index else c, ns)
    return ns["voice_from_transcript"]
