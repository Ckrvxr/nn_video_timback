"""Rich-based console wrapper replacing loguru."""

from rich.console import Console as _RichConsole

_console = _RichConsole(stderr=True, highlight=False, markup=True)


def _fmt(msg, *args):
    return msg.format(*args) if args else msg


class _ConsoleProxy:
    """Mimics loguru's console.info / warning / error / success / debug API on top of rich."""

    @staticmethod
    def info(msg, *args):
        _console.print(_fmt(msg, *args))

    @staticmethod
    def warning(msg):
        _console.print(f"[yellow]{msg}[/]")

    @staticmethod
    def error(msg):
        _console.print(f"[red]{msg}[/]")

    @staticmethod
    def success(msg):
        _console.print(f"[green]{msg}[/]")

    @staticmethod
    def debug(msg):
        _console.print(f"[dim]{msg}[/]")

    @staticmethod
    def opt(colors=True):
        """Dummy for loguru compat — rich always processes markup."""
        return _ConsoleProxy


console = _ConsoleProxy()


def section(title: str):
    _console.print(f"[bold cyan]═══ {title} ═══[/]")


def sub_section(title: str):
    _console.print(f"[bold]▸ {title}[/]")


def metric(key: str, value: str):
    _console.print(f"  [white]{key}:[/] [green]{value}[/]")


def detail(key: str, value: str):
    _console.print(f"    [dim]{key}:[/] [green]{value}[/]")


def divider():
    _console.print("[dim]──────────────────────────────────────[/]")
