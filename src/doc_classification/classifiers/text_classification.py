# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from src.para2vec.predict_para2vec import modeL_prediction
from tqdm import tqdm
import re
from smart_open import open
from datetime import datetime
import logging
from ..utils import load_csv


class text_classification():
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def preprocess_data(self, raw_text):
        text = ''.join(BeautifulSoup(raw_text,
                                     "html.parser").findAll(string=True))
        text = "<br />".join(text.split("\n"))
        norm_text = text.lower()
        # Replace breaks with spaces
        norm_text = norm_text.replace('<br />', ' ')
        # Pad punctuation with spaces on both sides
        norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
        tokens = norm_text.split()
        return tokens

    def get_detault_detail(self):
        classifier_details = {}
        classifier_details["model_name"] = None
        classifier_details["model_uuid"] = None
        classifier_details["model_path"] = None
        classifier_details["model_object"] = None
        classifier_details["is_default"] = True
        classifier_details["is_root"] = False
        classifier_details["version"] = 0.00
        classifier_details["description"] = None
        classifier_details["creation_date"] = datetime.now().strftime("%H:%M:%S.%f - %b %d %Y")
        classifier_details["update_date"] = datetime.now().strftime("%H:%M:%S.%f - %b %d %Y")
        classifier_details["evaluation"] = None
        classifier_details["data_path"] = "data"
        classifier_details["data_file_name"] = "DBPEDIA_train.csv"
        other_dict = {"para2Vec_path": "src/para2vec/models",
                      "para2vec_file_name": "dbpedia_para2vec_model.pickle"}
        classifier_details["other_details"] = other_dict
        classifier_details["child"] = {"sub_classifiers":[],
                                       "lables": []}
        return classifier_details

    def set_classes(self, approved_lables):
        # approved lables is list of tupple each tupple  contain the sub type 
        # of dictionary approved_lables = [('A', 'B'),('c'), ("D", "E")]
        self.logger.info(approved_lables)
        self.approved_lables = self._lable_convertor(approved_lables)
        self.masked_lables = self.mask_lables(approved_lables)
        self.logger.info(self.masked_lables)

    def define_model(self, model_name, model_path, version):
        self.model_name = model_name
        self.model_path = model_path
        self.version = version

    def _lable_convertor(self, lables):
        # convert the list of tupple into list
        return_list = []
        for model_name, lable_tuple in lables.items():
            self.logger.info(len(lable_tuple))
            for lable in lable_tuple:
                return_list.append(lable)
        self.logger.info(return_list)
        return return_list

    def mask_lables(self, lables):
        # create a Mask opetatoin for approved lables.
        # approved_lables = [('A', 'B'),('C'), ("D", "E")]
        # masked will be {"1": ['A', 'B'], "2": ['c'], "3": ["D", "E"]}
        self.logger.info(lables)
        return_mask = {}
        if len(lables) > 1:
            for key, all_lables in lables.items():
                if len(key.split("--lables")) > 1:
                    for lable in all_lables:
                        return_mask[lable] = lable
                else:
                    return_mask[key] = all_lables
        if len(lables) == 1:
            for key, all_lables in lables.items():
                for lable in all_lables:
                    return_mask[lable] = lable
        if len(return_mask) == 0:
            return lables
        return return_mask

    def unmask_lable(self, masked_value):
        # get masked input and convert it into its original form.
        # masked will be {"1": ['A', 'B'], "2": ['c'], "3": ["D", "E"]}
        # return unmasked value as masked value is 1 then return ["A", "B"]
        return [lables for mask, lables in self.masked_lables.items()
                if masked_value == mask]

    def train_test_spliting(self, features, lables, test_size=0.2):
        train_x, test_x, train_y, test_y = train_test_split(features, lables,
                                                            test_size=test_size,
                                                            random_state=42)
        return train_x, test_x, train_y, test_y

    def embedding(self, para2vec_model_name, para2vec_model_path, raw_text):
        modeL_prediction_obj = modeL_prediction(self.logger)
        model_path = "{}/{}".format(para2vec_model_path, para2vec_model_name)
        modeL_prediction_obj.load_model(model_path)
        return modeL_prediction_obj.get_para_2_vector(raw_text)

    def embedding_all(self, para2vec_model_name, para2vec_model_path, csv_df,
                      raw_data_path):
        all_para_2_vector = []
        modeL_prediction_obj = modeL_prediction(self.logger)
        model_path = "{}/{}".format(para2vec_model_path, para2vec_model_name)
        modeL_prediction_obj.load_model(model_path)
        with tqdm(total=len(csv_df)) as pbar:
            for index, row in csv_df.iterrows():
                pbar.update(1)
                text = row['text']
                file_vector = modeL_prediction_obj.get_para_2_vector(text)
                all_para_2_vector.append(file_vector)
        csv_df['para_2_vec'] = all_para_2_vector
        # save_dataframe(csv_df, raw_data_path, "featured.csv")
        return csv_df

    def _filter_data(self, csv_dataframe):
        self.logger.info("Total Length of Data: {}".format(len(csv_dataframe)))
        csv_df = csv_dataframe[csv_dataframe.l3.isin(
                self.approved_lables)]
        for mask, lables in self.masked_lables.items():
            csv_df = csv_df.replace(to_replace=lables, value=mask)
        self.logger.info("Data After Filter: {}".format(len(csv_df)))
        return csv_df

    def get_data(self, data_path, file_name):
        csv_df = load_csv(self.logger, data_path, file_name)
        return self._filter_data(csv_df)

    def analysis(self):
        # do the data analysis on this classification.
        pass
