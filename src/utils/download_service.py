import os

from utils.get_file_path import get_file_path
import gdown


class GoogleDriverDownloader:

    def download(self, file_id: str, destination: str) -> None:
        file_url = f'https://drive.google.com/uc?id={file_id}'
        output = get_file_path(destination)
        gdown.download(file_url, output, quiet=False)
