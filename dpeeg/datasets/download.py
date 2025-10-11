from pathlib import Path

import pooch
from pooch import file_hash, retrieve
from pooch.downloaders import choose_downloader


def data_dl(
    url: str | Path,
    path: str | Path,
    force_update: bool = False,
    processor=None,
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
    processor : None, "unzip", "untar", instance of pooch.Unzip, instance of pooch.Untar
        What to do after downloading the file. ``"unzip"`` and ``"untar"`` will
        decompress the downloaded file in place; for custom extraction (e.g.,
        only extracting certain files from the archive) pass an instance of
        ``pooch.Unzip`` or ``pooch.Untar``. If ``None`` (the
        default), the files are left as-is.

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

    if processor == "untar":
        processor = pooch.Untar(extract_dir=path)
    elif processor == "unzip":
        processor = pooch.Unzip(extract_dir=path)

    dlpath = retrieve(
        url,
        known_hash,
        fname=Path(url).name,
        path=str(destination),
        processor=processor,
        downloader=downloader,
        progressbar=True,
    )
    return dlpath
