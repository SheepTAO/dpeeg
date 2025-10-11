# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

import sys
import logging
from pathlib import Path


class Logger:
    """Logging hooks for terminals and file streams.

    Can handle terminal printing and file writing in three situations:
    1.Terminal print;
    2.Terminal and file streams write simultaneously;
    3.Ordinary file write.

    Parameters
    ----------
    logger : str
        Specify the logger name, creating it if necessary. If no name is
        specified, return the root logger.
    path : str, optional
        The path of log file.
    mode : str
        The write mode of the log file.
    clevel : int, str
        The log level of console.
    flevel : int, str, optional
        The log level of filehandler. If it is None, the log file will not be
        written and parameter path will be ignored.
    """

    def __init__(
        self,
        logger: str = "dpeeg_root",
        path: str | Path | None = None,
        mode: str = "a",
        clevel: int | str = logging.INFO,
        flevel: int | str | None = None,
    ) -> None:
        self._logger = logging.getLogger(logger)
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)

        self._sh = logging.StreamHandler()
        shfmt = logging.Formatter("%(message)s")
        self._sh.setFormatter(shfmt)
        self._sh.setLevel(clevel)
        self._logger.addHandler(self._sh)
        if flevel is not None:
            assert path is not None, "path cannot be empty when use filer."
            path = Path(path).absolute()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = logging.FileHandler(str(path), mode)
            fhfmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)8s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self._fh.setFormatter(fhfmt)
            self._fh.setLevel(flevel)
            self._logger.addHandler(self._fh)

        sys.excepthook = self.handle_exception

    def update_sh_level(self, level: int | str):
        self._sh.setLevel(level)

    def get_sh_level(self) -> str:
        return logging.getLevelName(self._sh.level)

    def debug(self, message: str):
        self._logger.debug(f"[DEBUG]: {message}")

    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(f"[WARNING]: {message}")

    def error(self, message: str):
        self._logger.error(f"[ERROR]: {message}")

    def critical(self, message: str):
        self._logger.critical(f"[CRITICAL]: {message}")

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        self._logger.critical(
            "Uncaught exception>>>", exc_info=(exc_type, exc_value, exc_traceback)
        )
