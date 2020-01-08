# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import re
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import pandas as pd
from smart_open import open
from random import shuffle
import logging


class data_preparation(object):

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def __load_csv(self, raw_data_path, csv_file_name):
        # load csv file for processing.
        file_path = raw_data_path+"/"+csv_file_name
        print(file_path)
        pea_all_record_df = pd.read_csv(file_path)
        # logger.info("CSV uploaded")
        return pea_all_record_df

    def _preprocess_data(self, raw_text):
        text = ''.join(BeautifulSoup(raw_text,
                                     "html.parser").findAll(text=True))
        text = "<br />".join(text.split("\n"))
        norm_text = text.lower()
        # Replace breaks with spaces
        norm_text = norm_text.replace('<br />', ' ')
        # Pad punctuation with spaces on both sides
        norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
        tokens = norm_text.split()
        return tokens

    def main(self, raw_data_path, csv_file_name):
        all_data_text = []
        all_records = self.__load_csv(raw_data_path, csv_file_name)
        self.logger.info("Data Loading started:")
        with tqdm(total=len(all_records)) as pbar:
            for index, row in all_records.iterrows():
                pbar.update(1)
                text = row['text']
                tag = row['l3']
                all_data_text.append(TaggedDocument(words=text,
                                                    tags=tag))
        shuffle(all_data_text)
        self.logger.info("Data loading Completed")
        self.logger.info("total documents: {}".format(len(all_data_text)))
        return all_data_text


if __name__ == "__main__":
    # this is for the debuggind this file only.
    data_prepration_obj = data_preparation()
    data_prepration_obj.main()
