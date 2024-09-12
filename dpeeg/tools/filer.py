# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from pathlib import Path
import csv


class Filer:
    """A simple file handler to write in realtime.

    Parameters
    ----------
    path : str | Path
        File path. Support hierarchical folder structure.
    mode : str
        Mode of write.
    exist_ok : bool
        Whether to allow files that already exist.
    """

    def __init__(
        self, path: str | Path, mode: str = "a", exist_ok: bool = False
    ) -> None:
        self.path = Path(path)
        self.mode = mode
        if self.path.exists() and not exist_ok:
            raise FileExistsError(f"File '{path}' already exists.")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: str) -> None:
        with open(self.path, self.mode) as f:
            f.write(data)


class CSVer(Filer):
    """Heavy development."""

    def __init__(self, path: str, mode: str = "a") -> None:
        super().__init__(path, mode)

    def write(self, data: str) -> None:
        with open(self.path, self.mode) as f:
            w = csv.writer(f)
            w.writerows(data)
