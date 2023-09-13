import urllib.request


default_user_agent = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"


def download_url(url: str, download_to: str, *, user_agent: str = default_user_agent) -> None:
    print(f"Downloading from {url}...")

    request = urllib.request.Request(url=url, headers={"User-Agent": user_agent})
    response = urllib.request.urlopen(request).read()

    with open(download_to, mode="wb") as f:
        f.write(response)

    print(f"Downloaded from {url}")
