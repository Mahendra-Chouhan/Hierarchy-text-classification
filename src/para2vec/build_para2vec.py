# -*- coding: utf-8 -*-
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
import logging
import dill
from smart_open import open
from .data_preparation import data_preparation


class build_model(object):
    def __init__(self, logger=None):
        self.cores = multiprocessing.cpu_count()
        assert gensim.models.doc2vec.FAST_VERSION > -1, """This
        will be painfully slow otherwise"""
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = None

    def define_model(self, model_name):
        self.model = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0,
                             min_count=2, sample=0, epochs=2,
                             workers=self.cores)
        self.model_name = model_name

    def build_vocab(self, all_data):
        self.model.build_vocab(all_data)
        self.logger.info("Vocab build for model {}".format(self.model_name))

    def _save_model(self, model_name, model_path):
        model_path_complete = model_path+"/"+str(model_name)+".pickle"
        with open(model_path_complete, 'wb') as f:
            dill.dump(self.model, f)
        self.logger.info("""Model Saved at {}""".format(model_path))

    def train_model(self, all_data):
        self.logger.info("Traning started")
        self.model.train(all_data, total_examples=len(all_data),
                         epochs=self.model.epochs)
        self.logger.info("""Traning Completed""")

    def evaluation(self, test_data):
        pass

    def main(self, model_name, raw_data_path, csv_file_name,
             model_path="models"):
        data_prepration_obj = data_preparation(self.logger)
        all_data = data_prepration_obj.main(raw_data_path=raw_data_path,
                                            csv_file_name=csv_file_name)
        self.define_model(model_name)
        self.build_vocab(all_data)
        self.train_model(all_data)
        self._save_model(self.model_name, model_path)


if __name__ == "__main__":
    # this is for the debuggind this file only.
    model_name = "first_model"
    build_model_obj = build_model()
    build_model_obj.main(model_name)
