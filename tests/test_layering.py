"""The layering rules, enforced. The structure was cleaned up deliberately (production code no longer imports research, the
domain layer stands alone, the `src/` root holds only the application core); these tests stop it decaying again.
Imports under `if TYPE_CHECKING:` are ignored: they cost nothing at run time and create no real dependency."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
ROOT_ALLOWED = {"config", "control", "logging_setup", "pipeline", "results", "runner", "versions", "workflow"}
PACKAGES = {"agents", "app", "data", "database", "engine", "intraday", "learning", "llm", "monitoring", "ops", "research"}


def module_name(path: Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def package_of(name: str) -> str:
    parts = name.split(".")
    return parts[1] if len(parts) > 1 and parts[1] in PACKAGES else "(root)"


def runtime_imports(path: Path):
    """(imported module, level) for every import that runs, i.e. not under TYPE_CHECKING."""
    tree = ast.parse(path.read_text())
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            skip.update(id(n) for n in ast.walk(node))
    out = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            out.append((node.module, [a.name for a in node.names], node.col_offset == 0))
        elif isinstance(node, ast.Import):
            out.extend((a.name, [], node.col_offset == 0) for a in node.names if a.name.startswith("src"))
    return out


MODULES = {module_name(p): p for p in sorted(SRC.rglob("*.py"))}
IMPORTS = {m: runtime_imports(p) for m, p in MODULES.items()}


def offenders(sources, forbidden):
    return sorted(f"{m} imports {t}" for m in MODULES if package_of(m) in sources
                  for t, _, _ in IMPORTS[m] if package_of(t) in forbidden)


def test_production_code_never_depends_on_research_or_ops():
    production = {"agents", "data", "database", "engine", "intraday", "llm", "monitoring"}
    assert offenders(production, {"research", "ops", "learning"}) == []


def test_the_engine_is_a_leaf_of_the_domain_it_does_not_reach_up_into_agents_llm_or_data():
    assert offenders({"engine"}, {"agents", "llm", "data", "app", "research", "ops", "monitoring", "intraday"}) == []


def test_the_data_layer_does_not_depend_on_agents_llm_or_the_orchestration():
    assert offenders({"data"}, {"agents", "llm", "app", "research", "ops"}) == []
    assert not [m for m in MODULES if package_of(m) == "data" and any(t in ("src.pipeline", "src.workflow") for t, _, _ in IMPORTS[m])]


def test_the_learning_layer_reads_the_database_and_data_only_and_nothing_trades_through_it():
    assert offenders({"learning"}, {"app", "research", "ops", "monitoring", "intraday", "llm"}) == []


def test_the_llm_client_knows_nothing_about_agents_or_trading():
    assert offenders({"llm"}, {"agents", "data", "app", "research", "ops", "monitoring", "intraday", "database"}) == []


def test_the_src_root_holds_only_the_application_core():
    root = {p.stem for p in SRC.glob("*.py") if p.stem != "__init__"}
    assert root <= ROOT_ALLOWED, f"new module(s) in the src/ root: {sorted(root - ROOT_ALLOWED)}: put them in a package"


def test_no_module_imports_a_private_name_from_another_module():
    found = [f"{m} imports {t}.{n}" for m, deps in IMPORTS.items() for t, names, _ in deps for n in names
             if n.startswith("_") and not n.startswith("__")]
    assert found == []


def test_there_are_no_import_cycles_between_modules():
    graph = {m: sorted({t for t, _, top in deps if top and t in MODULES and t != m}) for m, deps in IMPORTS.items()}
    cycles, state = [], {}

    def visit(node, path):
        state[node] = "open"
        for nxt in graph[node]:
            if state.get(nxt) == "open":
                cycles.append(path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt])
            elif nxt not in state:
                visit(nxt, path + [nxt])
        state[node] = "done"

    for module in graph:
        if module not in state:
            visit(module, [module])
    assert cycles == []


@pytest.mark.parametrize("module", ["src.workflow"])
def test_the_graphs_do_not_import_the_pipeline_that_builds_them(module):
    assert not [t for t, _, _ in IMPORTS[module] if t == "src.pipeline"]
