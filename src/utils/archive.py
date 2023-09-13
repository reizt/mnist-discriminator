import os.path
import os
import gzip
import shutil


def unarchive(file_from: str, file_to: str, *, delete_archive: bool = False) -> None:
    if not os.path.exists(file_from) or not os.path.isfile(file_from):
        raise ValueError(f"{file_from} doesn't exist as file")

    with gzip.open(file_from, "rb") as f_in:
        with open(file_to, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if delete_archive:
        os.remove(file_from)
