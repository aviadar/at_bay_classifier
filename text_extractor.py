import os
from bs4 import BeautifulSoup
from typing import Optional


class TextExtractor:
    @staticmethod
    def extract_from_dir(dir_path: str, obj_depth: dict, max_depth_to_consider=1) -> Optional[str]:
        try:
            file_list = os.listdir(dir_path)
        except FileNotFoundError:
            return None

        appended_processed_text = ''
        for file in file_list:
            if obj_depth[os.path.join(dir_path, file)] <= max_depth_to_consider:
                appended_processed_text += TextExtractor.extract_from_file(os.path.join(dir_path, file))
        return appended_processed_text

    @staticmethod
    def extract_from_file(file_path: str) -> Optional[str]:
        try:
            with open(file_path) as fid:
                soup = BeautifulSoup(fid, 'lxml')
                processed_text = ' ' + soup.text.replace('\n', '').replace('\xa0', '').replace('\t', '')
                return processed_text
        except FileNotFoundError:
            return None
