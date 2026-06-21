import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

console = logger


def section(title: str):
    console.opt(colors=True).info("<bold><cyan>═══ {} ═══</cyan></bold>", title)


def sub_section(title: str):
    console.opt(colors=True).info("<bold><level>▸ {}</level></bold>", title)


def metric(key: str, value: str):
    console.opt(colors=True).info("  <white>{}</white> <green>{}</green>", f"{key}:", value)


def divider():
    console.opt(colors=True).info("<dim>──────────────────────────────────────</dim>")
