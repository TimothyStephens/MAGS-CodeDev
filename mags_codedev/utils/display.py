"""Terminal display utilities: status tree, flat table, token formatting."""

from typing import Optional

from rich.console import Group
from rich.table import Table
from rich.tree import Tree


def _status_color(status: str) -> str:
    """Return ANSI color name for a status string."""
    if "Completed" in status or "Success" in status:
        return "green"
    if "Failed" in status or "Error" in status or "Fail" in status:
        return "red"
    if "Blocked" in status:
        return "yellow"
    if "Waiting" in status:
        return "dim"
    return "blue"


def _status_icon(status: str) -> str:
    """Return a short icon for a status string."""
    if "Completed" in status or "Success" in status:
        return "✓"
    if "Failed" in status or "Error" in status or "Fail" in status:
        return "✗"
    if "Blocked" in status:
        return "⚠"
    if "Waiting" in status:
        return "◌"
    return "●"


def _fmt_tok(n: int) -> str:
    """Compact token number."""
    if n == 0:
        return "0"
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _token_str(info: dict) -> str:
    """Format compact token breakdown: 'in/out=total'."""
    tok_in = info.get("tokens_in", 0)
    tok_out = info.get("tokens_out", 0)
    total = tok_in + tok_out
    if total == 0:
        return "—"
    in_s = _fmt_tok(tok_in)
    out_s = _fmt_tok(tok_out)
    total_s = _fmt_tok(total)
    return f"{in_s}/{out_s}={total_s}t"


def _short_status(status: str) -> str:
    """Strip status prefixes for cleaner display."""
    for prefix in ("Waiting: ", "Blocked: ", "Success: ", "Failed: ", "Error: "):
        if status.startswith(prefix):
            return status[len(prefix):]
    return status


def generate_status_table(status_dict: dict, module_map: Optional[dict] = None):
    """Generates a dependency tree with totals footer: file | hash | status | tokens."""
    if not module_map:
        return _flat_table(status_dict)

    # Find roots
    all_deps: set = set()
    for spec in module_map.values():
        for dep in spec.get("dependencies", []):
            all_deps.add(dep)
    roots = sorted(loc for loc in module_map if loc not in all_deps)
    if not roots:
        roots = sorted(module_map.keys())

    def _node_label(loc: str) -> str:
        """Build node label: file  hash  STATUS  [artifact?]  tok."""
        info = status_dict.get(loc, {})
        status = info.get("status", "Pending")
        hash_val = info.get("hash", "")[:8] or "—"
        color = _status_color(status)
        icon = _status_icon(status)
        short = _short_status(status)
        tok = _token_str(info)

        # Artifact indicator
        artifact = ""
        if info.get("has_artifact"):
            a_hash = info.get("artifact_hash", "")[:6] or "?"
            artifact = f" [dim](artifact: {a_hash})[/]"

        return (
            f"[bold]{loc}[/bold]  "
            f"[dim]{hash_val}[/dim]  "
            f"[{color}]{icon} {short}[/]"
            f"{artifact}  "
            f"[dim]{tok}[/dim]"
        )

    tree = Tree("[bold cyan]Build Dependency Tree[/bold cyan]")

    def add_children(node, loc: str, path: set):
        if loc in path:
            node.add(f"[red]{loc} (cycle)[/red]")
            return
        branch = node.add(_node_label(loc))
        spec = module_map.get(loc, {})
        deps = sorted(spec.get("dependencies", []))
        new_path = path | {loc}
        for dep in deps:
            if dep in module_map:
                add_children(branch, dep, new_path)
            else:
                branch.add(f"[red]{dep} (missing/external)[/red]")

    for root in roots:
        add_children(tree, root, set())

    # Compute totals from status_dict
    total_in = sum(info.get("tokens_in", 0) for info in status_dict.values())
    total_out = sum(info.get("tokens_out", 0) for info in status_dict.values())
    completed = sum(1 for info in status_dict.values() if "Completed" in info.get("status", ""))
    failed = sum(
        1 for info in status_dict.values()
        if "Failed" in info.get("status", "") or "Error" in info.get("status", "")
    )
    blocked = sum(1 for info in status_dict.values() if "Blocked" in info.get("status", ""))
    waiting = len(status_dict) - completed - failed - blocked

    tok_str = _token_str({"tokens_in": total_in, "tokens_out": total_out})
    footer = (
        "[dim]─[/dim]" * 40 + "\n"
        f"{blocked} ⚠ blocked, {waiting} ◌ waiting  "
        f"[dim]{tok_str}[/dim]"
    )

    return Group(tree, footer)


def _flat_table(status_dict: dict):
    """Fallback flat table when no dependency map is available."""
    table = Table(title="Parallel Function Builder", show_header=True, header_style="bold cyan")
    table.add_column("Module", style="bold")
    table.add_column("Hash", style="dim", width=9)
    table.add_column("Status")
    table.add_column("Artifact", style="dim", width=8)
    table.add_column("Tokens")

    for loc in sorted(status_dict):
        info = status_dict[loc]
        status = info.get("status", "Pending")
        color = _status_color(status)
        icon = _status_icon(status)
        hash_val = info.get("hash", "")[:8] or "—"
        tok = _token_str(info)

        # Artifact indicator
        if info.get("has_artifact"):
            a_hash = info.get("artifact_hash", "")[:6] or "?"
            artifact = a_hash
        else:
            artifact = "—"

        table.add_row(loc, hash_val, f"[{color}]{icon} {status}[/{color}]", artifact, tok)

    # Totals
    total_in = sum(info.get("tokens_in", 0) for info in status_dict.values())
    total_out = sum(info.get("tokens_out", 0) for info in status_dict.values())
    completed = sum(1 for info in status_dict.values() if "Completed" in info.get("status", ""))
    failed = sum(
        1 for info in status_dict.values()
        if "Failed" in info.get("status", "") or "Error" in info.get("status", "")
    )
    blocked = sum(1 for info in status_dict.values() if "Blocked" in info.get("status", ""))
    waiting = len(status_dict) - completed - failed - blocked
    tok_str = _token_str({"tokens_in": total_in, "tokens_out": total_out})
    footer = (
        "[dim]────────────────────────────────────────[/dim]\n"
        "  [bold]Total:[/bold] {completed} ✓ completed, {failed} ✗ failed, "
        "{blocked} ⚠ blocked, {waiting} ◌ waiting  "
        "[dim]{tok_str}[/dim]"
    ).format(completed=completed, failed=failed, blocked=blocked, waiting=waiting, tok_str=tok_str)

    return Group(table, footer)
