from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import json

from .common import DOCS_DEP_MAP_PATH, REPO_ROOT, SRC_ROOT, repo_relative


HOTSPOT_MODULES = (
    "dagzoo.core.execution_semantics",
    "dagzoo.core.fixed_layout_batched",
    "dagzoo.core.fixed_layout_runtime",
    "dagzoo.core.layout",
    "dagzoo.core.node_pipeline",
    "dagzoo.core.dataset",
    "dagzoo.core.config_resolution",
    "dagzoo.core.generation_runtime",
    "dagzoo.config",
    "dagzoo.io.lineage_schema",
)


@dataclass(frozen=True)
class ModuleSummary:
    imports: tuple[str, ...]
    direct_importers: tuple[str, ...]
    transitive_importers: tuple[str, ...]
    impacted_packages: tuple[str, ...]


@dataclass(frozen=True)
class ImportGraph:
    imports: dict[str, tuple[str, ...]]
    reverse_imports: dict[str, tuple[str, ...]]
    module_paths: dict[str, str]

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(sorted(self.imports))

    def package_graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}
        for module, imports in self.imports.items():
            package = module_to_package(module)
            graph.setdefault(package, set())
            for dependency in imports:
                dependency_package = module_to_package(dependency)
                if dependency_package != package:
                    graph[package].add(dependency_package)
                    graph.setdefault(dependency_package, set())
        return graph

    def transitive_importers(self, module: str) -> tuple[str, ...]:
        visited: set[str] = set()
        stack = list(self.reverse_imports.get(module, ()))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.reverse_imports.get(current, ()))
        return tuple(sorted(visited))

    def module_summary(self, module: str) -> ModuleSummary:
        direct_importers = self.reverse_imports.get(module, ())
        transitive_importers = self.transitive_importers(module)
        impacted_packages = sorted(
            {
                module_to_package(importer)
                for importer in transitive_importers
                if module_to_package(importer)
                not in {module_to_package(module), "dagzoo", "dagzoo.__main__"}
            }
        )
        return ModuleSummary(
            imports=self.imports.get(module, ()),
            direct_importers=direct_importers,
            transitive_importers=transitive_importers,
            impacted_packages=tuple(impacted_packages),
        )

    def module_to_path(self, module: str) -> str | None:
        return self.module_paths.get(module)


def module_to_package(module: str) -> str:
    parts = module.split(".")
    if len(parts) <= 2:
        return module
    return ".".join(parts[:2])


def module_name_for_path(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT)
    module = "dagzoo." + ".".join(relative.parts)
    module = module[:-3] if module.endswith(".py") else module
    if module.endswith(".__init__"):
        module = module[:-9]
    return module


def path_to_module(path_str: str) -> str | None:
    path = REPO_ROOT / path_str
    if not path.is_relative_to(SRC_ROOT) or path.suffix != ".py":
        return None
    return module_name_for_path(path)


def build_import_graph() -> ImportGraph:
    module_paths: dict[str, str] = {}
    known_modules: set[str] = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = module_name_for_path(path)
        module_paths[module] = repo_relative(path)
        known_modules.add(module)

    imports: dict[str, tuple[str, ...]] = {}
    reverse_imports: dict[str, set[str]] = {module: set() for module in known_modules}
    for module, path_str in module_paths.items():
        path = REPO_ROOT / path_str
        module_imports = tuple(sorted(_collect_imports(path, module, known_modules)))
        imports[module] = module_imports
        for dependency in module_imports:
            reverse_imports.setdefault(dependency, set()).add(module)

    return ImportGraph(
        imports=imports,
        reverse_imports={key: tuple(sorted(value)) for key, value in reverse_imports.items()},
        module_paths=module_paths,
    )


def _collect_imports(path: Path, module: str, known_modules: set[str]) -> set[str]:
    tree = ast.parse(path.read_text())
    current_package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    discovered: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                normalized = _normalize_import(alias.name, known_modules)
                if normalized is not None and normalized != module:
                    discovered.add(normalized)
        elif isinstance(node, ast.ImportFrom):
            for dependency in _normalize_import_from(node, current_package, known_modules):
                if dependency != module:
                    discovered.add(dependency)
    return discovered


def _normalize_import(candidate: str, known_modules: set[str]) -> str | None:
    matches = [
        module
        for module in known_modules
        if candidate == module or candidate.startswith(module + ".")
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _normalize_import_from(
    node: ast.ImportFrom, current_package: str, known_modules: set[str]
) -> set[str]:
    if node.level == 0:
        base_parts = ()
    else:
        current_parts = current_package.split(".")
        if node.level == 1:
            base_parts = tuple(current_parts)
        else:
            base_parts = tuple(current_parts[: -(node.level - 1)])
    module_parts = tuple(node.module.split(".")) if node.module else ()
    base_module = ".".join(base_parts + module_parts)
    discovered: set[str] = set()
    if base_module.startswith("dagzoo"):
        normalized = _normalize_import(base_module, known_modules)
        if normalized is not None:
            discovered.add(normalized)
    for alias in node.names:
        if alias.name == "*":
            continue
        if base_module:
            child_candidate = f"{base_module}.{alias.name}"
        else:
            child_candidate = alias.name
        normalized_child = _normalize_import(child_candidate, known_modules)
        if normalized_child is not None:
            discovered.add(normalized_child)
    return discovered


def strongly_connected_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for dependency in sorted(graph.get(node, ())):
            if dependency not in indices:
                visit(dependency)
                lowlinks[node] = min(lowlinks[node], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[dependency])

        if lowlinks[node] != indices[node]:
            return

        component: list[str] = []
        while stack:
            current = stack.pop()
            on_stack.remove(current)
            component.append(current)
            if current == node:
                break
        components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(components, key=lambda component: (len(component), component))


def condensation_graph(
    graph: dict[str, set[str]],
) -> tuple[list[tuple[str, ...]], dict[int, set[int]]]:
    components = strongly_connected_components(graph)
    index_by_node = {
        node: component_index
        for component_index, component in enumerate(components)
        for node in component
    }
    condensed: dict[int, set[int]] = {index: set() for index in range(len(components))}
    for node, dependencies in graph.items():
        source_index = index_by_node[node]
        for dependency in dependencies:
            target_index = index_by_node[dependency]
            if source_index != target_index:
                condensed[source_index].add(target_index)
    return components, condensed


def topological_order(graph: dict[int, set[int]]) -> list[int]:
    indegree = {node: 0 for node in graph}
    for dependencies in graph.values():
        for dependency in dependencies:
            indegree[dependency] = indegree.get(dependency, 0) + 1
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for dependency in sorted(graph.get(node, ())):
            indegree[dependency] -= 1
            if indegree[dependency] == 0:
                ready.append(dependency)
                ready.sort()
    return order


def package_dag_lines(graph: ImportGraph) -> list[str]:
    package_graph = graph.package_graph()
    components, condensed = condensation_graph(package_graph)
    lines: list[str] = []
    for component_index in topological_order(condensed):
        component = components[component_index]
        label = ", ".join(f"`{name}`" for name in component)
        dependencies = [
            ", ".join(f"`{name}`" for name in components[target_index])
            for target_index in sorted(condensed[component_index])
        ]
        if dependencies:
            lines.append(f"- {label} depends on {', '.join(dependencies)}")
        else:
            lines.append(f"- {label} has no internal package dependencies")
    return lines


def hotspot_summaries(
    graph: ImportGraph, modules: tuple[str, ...] = HOTSPOT_MODULES
) -> dict[str, ModuleSummary]:
    return {module: graph.module_summary(module) for module in modules if module in graph.imports}


def render_dependency_map_markdown(graph: ImportGraph) -> str:
    lines = [
        "# Module Dependency Map",
        "",
        "This file is generated from imports under `src/dagzoo`.",
        "Run `./scripts/dev deps --write-docs` after changing internal module edges.",
        "",
        "## Package Dependency DAG",
        "",
        "The package graph below collapses strongly connected components so the result stays acyclic.",
        "",
        *package_dag_lines(graph),
        "",
        "## Change-Impact Hotspots",
        "",
        "The sections below list direct importers and full transitive downstream modules.",
        "Use them to predict which runtime paths are likely to move when a hot module changes.",
        "",
    ]
    for module, summary in hotspot_summaries(graph).items():
        lines.extend(
            [
                f"### `{module}`",
                "",
                f"- Path: `{graph.module_to_path(module)}`",
                f"- Imports: {_inline_list(summary.imports)}",
                f"- Direct downstream modules: {_inline_list(summary.direct_importers)}",
                f"- Transitive downstream modules: {_inline_list(summary.transitive_importers)}",
                f"- Downstream package areas: {_inline_list(summary.impacted_packages)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _inline_list(values: tuple[str, ...]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def render_scope_text(graph: ImportGraph, scope: str) -> str:
    if scope == "package":
        return "\n".join(package_dag_lines(graph)) + "\n"
    if scope == "full":
        payload = {
            module: {
                "imports": list(graph.imports[module]),
                "imported_by": list(graph.reverse_imports.get(module, ())),
            }
            for module in graph.modules
        }
        return json.dumps(payload, indent=2) + "\n"
    return render_dependency_map_markdown(graph)


def render_scope_json(graph: ImportGraph, scope: str) -> str:
    if scope == "package":
        payload = {
            package: sorted(dependencies) for package, dependencies in graph.package_graph().items()
        }
        return json.dumps(payload, indent=2) + "\n"
    if scope == "full":
        payload = {
            module: {
                "imports": list(graph.imports[module]),
                "imported_by": list(graph.reverse_imports.get(module, ())),
            }
            for module in graph.modules
        }
        return json.dumps(payload, indent=2) + "\n"
    payload = {
        "package_dag": package_dag_lines(graph),
        "hotspots": {
            module: {
                "imports": list(summary.imports),
                "direct_downstream": list(summary.direct_importers),
                "transitive_downstream": list(summary.transitive_importers),
                "downstream_packages": list(summary.impacted_packages),
            }
            for module, summary in hotspot_summaries(graph).items()
        },
    }
    return json.dumps(payload, indent=2) + "\n"


def write_dependency_docs(graph: ImportGraph) -> str:
    content = render_dependency_map_markdown(graph)
    DOCS_DEP_MAP_PATH.write_text(content)
    return content


def dependency_docs_are_current(graph: ImportGraph) -> bool:
    if not DOCS_DEP_MAP_PATH.exists():
        return False
    return DOCS_DEP_MAP_PATH.read_text() == render_dependency_map_markdown(graph)
