"""Print the exact set of Python files a training run loads, by static import closure.

This is what produced metafiles/PYTHON_CLOSURE.txt. Regenerate any time:

    python scripts/list_training_closure.py .            # the 50-file closure
    python scripts/list_training_closure.py . --unused   # .py present but never imported

Method: AST walk of `import` / `from ... import` starting at main.py, resolving only
in-repo modules and including every parent-package __init__.py (which Python must
execute for the package chain to import). No code is executed, so it is safe to run
anywhere and needs no GPU or environment.

Caveat worth knowing: this reports what Python *loads*, which is a superset of what a
given run *exercises*. Eager registries (algorithms/wan/__init__.py,
experiments/exp_video.py, datasets/__init__.py, datasets/mixture.py) import their whole
package regardless of the config. Those were pruned on this branch, which is why the
closure here is 50 files rather than the 108 it was in the working repo. See
TRAINING_CODE_TRACE.md for the per-model breakdown of which files each run actually uses.
"""
import ast
import sys
from pathlib import Path


def build_closure(repo: Path):
    pkgs = {p.name for p in repo.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    tops = {p.stem for p in repo.glob("*.py")}

    def mod_to_paths(mod: str):
        parts = mod.split(".")
        if parts[0] not in pkgs and parts[0] not in tops:
            return []
        out = []
        base = repo.joinpath(*parts)
        if base.with_suffix(".py").exists():
            out.append(base.with_suffix(".py"))
        if (base / "__init__.py").exists():
            out.append(base / "__init__.py")
        # every parent package's __init__.py is executed on import
        for i in range(1, len(parts)):
            ini = repo.joinpath(*parts[:i], "__init__.py")
            if ini.exists():
                out.append(ini)
        return out

    def imports_of(path: Path):
        try:
            tree = ast.parse(path.read_text(errors="ignore"), filename=str(path))
        except SyntaxError:
            return []
        pkg = path.relative_to(repo).parent.as_posix().replace("/", ".")
        pkg = "" if pkg == "." else pkg
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import
                    base = pkg.split(".") if pkg else []
                    if node.level > 1:
                        base = base[: len(base) - (node.level - 1)]
                    mod = ".".join(base + ([node.module] if node.module else []))
                    found.append(mod)
                    found += [f"{mod}.{a.name}" for a in node.names]
                elif node.module:
                    found.append(node.module)
                    found += [f"{node.module}.{a.name}" for a in node.names]
        return found

    seen, stack = set(), [repo / "main.py"]
    while stack:
        f = stack.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        for mod in imports_of(f):
            for p in mod_to_paths(mod):
                if p not in seen:
                    stack.append(p)
    return {p.relative_to(repo).as_posix() for p in seen}


if __name__ == "__main__":
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    keep = build_closure(repo)
    if "--unused" in sys.argv:
        allpy = {
            p.relative_to(repo).as_posix()
            for p in repo.rglob("*.py")
            if ".git" not in p.parts
        }
        rows = sorted(allpy - keep)
        print(f"# {len(rows)} .py files present but NOT imported by a training run")
    else:
        rows = sorted(keep)
        print(f"# {len(rows)} Python files loaded by a training run (closure from main.py)")
    for r in rows:
        print(r)
