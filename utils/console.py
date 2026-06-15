import sys

from loguru import logger

# Remove default handler and add a clean console handler.
logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)

console = logger
