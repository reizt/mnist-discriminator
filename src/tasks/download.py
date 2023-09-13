import os.path
from src.utils.download_url import download_url
from src.utils.archive import unarchive
import asyncio


async def download_mnist() -> None:
    download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
    download_files: list[tuple[str, str, str]] = [
        ("train-images-idx3-ubyte.gz", "train-images.gz", "train-images"),
        ("train-labels-idx1-ubyte.gz", "train-labels.gz", "train-labels"),
        ("t10k-images-idx3-ubyte.gz", "test-images.gz", "test-images"),
        ("t10k-labels-idx1-ubyte.gz", "test-labels.gz", "test-labels"),
    ]

    async def download(file: tuple[str, str, str]):
        download_from, save_as, unarchive_as = file
        url = f"http://yann.lecun.com/exdb/mnist/{download_from}"
        download_path = f"{download_dir}/{save_as}"
        download_url(url, download_path)
        unarchive_path = f"{download_dir}/{unarchive_as}"
        unarchive(download_path, unarchive_path, delete_archive=True)

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    await asyncio.gather(*[download(f) for f in download_files])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(download_mnist())
    loop.close()
