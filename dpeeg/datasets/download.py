# Authors: SheepTAO <sheeptao@outlook.com>

# License: MIT
# Copyright the dpeeg contributors.

from pathlib import Path

from pooch import file_hash, retrieve
from pooch.downloaders import choose_downloader


def data_dl(
    url: str,
    path: str,
    force_update: bool = False,
):
    """Download file from url to specified path.

    Parameters
    ----------
    url : str
        Path to remote location of data.
    path : str
        Location of where to look for the data storing location. Default is
        `~/dpeeg/datasets`. If the dataset is not found under the given path,
        the data will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.

    Returns
    -------
    path : str
        Local path to the given data file.
    """
    destination = Path(path)

    downloader = choose_downloader(url, progressbar=True)
    if type(downloader).__name__ in ["HTTPDownloader", "DOIDownloader"]:
        downloader.kwargs.setdefault("verify", False)

    # Fetch the file
    if not destination.is_file() or force_update:
        if destination.is_file():
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        known_hash = None
    else:
        known_hash = file_hash(str(destination))
    dlpath = retrieve(
        url,
        known_hash,
        fname=Path(url).name,
        path=str(destination),
        progressbar=True,
        downloader=downloader,
    )
    return dlpath
