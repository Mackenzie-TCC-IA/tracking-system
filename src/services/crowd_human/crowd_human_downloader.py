from enum import Enum
import json
import os
import shutil
from typing import Callable, TypedDict

import cv2
from tqdm import tqdm

from utils.download_service import GoogleDriverDownloader
from utils.get_file_path import get_file_path


class CrowdHumanFiles(Enum):
    train01 = 'CrowdHuman_train01'
    train02 = 'CrowdHuman_train02'
    train03 = 'CrowdHuman_train03'
    train = 'CrowdHuman_train'
    val = 'CrowdHuman_val'
    train_labels = 'CrowdHuman_annotation_train'
    val_labels = 'CrowdHuman_annotation_val'


class CrowdHumanDownloader:

    google_drive_downloader = GoogleDriverDownloader()

    crowd_human_dataset_dir = get_file_path('datasets/crowd_human')

    files_to_download = {
        'CrowdHuman_train01': {'id': '134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y', 'ext': 'zip'},
        'CrowdHuman_train02': {'id': '17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla', 'ext': 'zip'},
        'CrowdHuman_train03': {'id': '1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW', 'ext': 'zip'},
        'CrowdHuman_val': {'id': '18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO', 'ext': 'zip'},
        'CrowdHuman_annotation_train': {'id': '1UUTea5mYqvlUObsC1Z8CFldHJAtLtMX3', 'ext': 'odgt'},
        'CrowdHuman_annotation_val': {'id': '10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL', 'ext': 'odgt'}
    }

    def init(self, download: bool, extract: bool) -> None:
        self.create_crowd_human_dir()

        self.download_files(download)
        self.unzip_files(extract)

        self.merge_folders(CrowdHumanFiles.train.value, [
                           CrowdHumanFiles.train01.value, CrowdHumanFiles.train02.value, CrowdHumanFiles.train03.value])

        self.normalize_folder(CrowdHumanFiles.train.value, 'train')
        self.normalize_folder(CrowdHumanFiles.val.value, 'valid')

        self.normalize_annotations(CrowdHumanFiles.train_labels.value, 'train')
        self.normalize_annotations(CrowdHumanFiles.val_labels.value, 'valid')

    def create_crowd_human_dir(self) -> None:

        has_already_crowd_dir = self.exists(self.crowd_human_dataset_dir)

        if not has_already_crowd_dir:
            os.mkdir(self.crowd_human_dataset_dir)

    def download_files(self, download: bool) -> None:
        if not download:
            return

        def downloadFile(filename: str, file_data: dict[str, str]) -> None:
            self.google_drive_downloader.download(
                file_data['id'], f'{self.crowd_human_dataset_dir}/{filename}.{file_data["ext"]}')

        self.iterate_over_files(downloadFile)

    def unzip_files(self, extract: bool) -> None:
        if not extract:
            return

        def unzip_file(filename: str, file_data: dict[str, str]) -> None:
            if file_data['ext'] != 'zip':
                return

            extract_to = f'{self.crowd_human_dataset_dir}/{filename}'
            exists = self.exists(extract_to)

            if exists:
                os.rmdir(extract_to)

            os.mkdir(extract_to)

            shutil.unpack_archive(
                f'{self.crowd_human_dataset_dir}/{filename}.zip', extract_dir=extract_to)

        print('Extracting files...')
        self.iterate_over_files(unzip_file)
        print('Done!')

    def iterate_over_files(self, callback: Callable[[str, dict[str, str]], None]) -> None:
        for filename, file_data in self.files_to_download.items():
            callback(filename, file_data)

    def merge_folders(self, folder_to_merge: str, folders: list[str]) -> None:
        folder_to_merge_path = f'{self.crowd_human_dataset_dir}/{folder_to_merge}'

        folder_to_merge_already_exists = self.exists(folder_to_merge_path)

        if folder_to_merge_already_exists:
            return

        os.mkdir(folder_to_merge_path)
        new_images_folder = os.path.join(folder_to_merge_path, 'Images')
        os.mkdir(new_images_folder)

        for folder in folders:
            folder_path = os.path.join(
                self.crowd_human_dataset_dir, folder)

            folder_path_exists = self.exists(folder_path)

            if not folder_path_exists:
                return

            folder_images_path = os.path.join(folder_path, 'Images')

            folder_image_files = os.listdir(folder_images_path)

            for folder_image_file in folder_image_files:
                folder_image_file_path = os.path.join(
                    folder_images_path, folder_image_file)

                shutil.move(folder_image_file_path, new_images_folder)

            shutil.rmtree(folder_path)

    def normalize_folder(self, folder_name: str, normalized_folder_name: str) -> None:
        normalized_folder_path = os.path.join(
            self.crowd_human_dataset_dir, normalized_folder_name)

        normalized_folder_path_exists = self.exists(normalized_folder_path)

        if normalized_folder_path_exists:
            return

        folder_to_rename = os.path.join(
            self.crowd_human_dataset_dir, folder_name)

        os.rename(folder_to_rename, normalized_folder_path)

        os.rename(os.path.join(normalized_folder_path, 'Images'),
                  os.path.join(normalized_folder_path, 'images'))

        os.mkdir(os.path.join(normalized_folder_path, 'labels'))

    def normalize_annotations(self, annotation_file: str, to_folder: str) -> None:
        annotation_file_path = os.path.join(
            self.crowd_human_dataset_dir, f'{annotation_file}.odgt')

        to_folder_path = os.path.join(
            self.crowd_human_dataset_dir, to_folder)

        to_folder_images_path = os.path.join(to_folder_path, 'images')
        to_folder_labels_path = os.path.join(to_folder_path, 'labels')

        annotation_file_exists = self.exists(annotation_file_path)

        if not annotation_file_exists:
            raise Exception(
                f'Annotation file {annotation_file} does not exists.')

        annotation_file_data = open(annotation_file_path, 'r')

        lines = annotation_file_data.readlines()

        print(f'Processing {to_folder} labels.')

        for line in tqdm(lines):
            current_annotation_data: AnnotationFileData = json.loads(line)

            image_id = current_annotation_data["ID"]

            file_to_write_path = os.path.join(
                to_folder_labels_path, f'{image_id}.txt')

            file_to_write = open(file_to_write_path, 'w')

            image = cv2.imread(os.path.join(
                to_folder_images_path, f'{image_id}.jpg'))

            image_heigth, image_width, _ = image.shape

            for gtbox in current_annotation_data['gtboxes']:
                if gtbox['tag'] == 'mask':
                    continue

                cls = 0
                visible_box = gtbox['vbox']
                x, y, width, heigth = visible_box

                x = max(int(x), 0)
                y = max(int(y), 0)
                width = min(int(width), image_width - x)
                heigth = min(int(heigth), image_heigth - y)

                center_x = (x + width / 2.)
                center_y = (y + heigth / 2.)

                normalized_width = float(width) / image_width
                normalized_heigth = float(heigth) / image_heigth

                normalized_x = center_x / image_width
                normalized_y = center_y / image_heigth

                file_to_write.write(
                    f'{cls} {normalized_x} {normalized_y} {normalized_width} {normalized_heigth}\n')

            file_to_write.close()

    def exists(self, file_path: str) -> bool:
        return os.path.exists(file_path)


class AnnotationFileData(TypedDict):
    ID: str
    gtboxes: list['GtBox']


class GtBox(TypedDict):
    tag: str
    vbox: list[int]
