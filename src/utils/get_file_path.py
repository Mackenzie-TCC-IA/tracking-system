from os import path, getcwd


def get_file_path(file_path: str) -> str:
    return path.join(getcwd(), file_path)
