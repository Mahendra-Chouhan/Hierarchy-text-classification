# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec
import logging
from .data_preparation import data_preparation


class modeL_prediction(object):
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.data_preparation_obj = data_preparation(self.logger)

    def load_model(self, model_path):
        # load the para2Vec model
        self.model = Doc2Vec.load(model_path)

    def reset_model(self):
        # once you done with model please reset it.
        self.model = None

    def get_para_2_vector(self, text, is_preparation=True):
        if is_preparation:
            process_text = self.data_preparation_obj._preprocess_data(text)
            return self.model.infer_vector(process_text, steps=20)
        return self.model.infer_vector([text], steps=20)


if __name__ == "__main__":
    # this is for the debuggind this file only.
    model_path = "models/first_model.pickle"
    predicted_text = """
    545sadsa ads sadsad sad adsadsad4s4s44s sd sdd 4ds d sd4sd sd4 sd 4sd4
    """
    modeL_prediction_obj = modeL_prediction()
    modeL_prediction_obj.load_model(model_path)
    print(modeL_prediction_obj.get_para_2_vector(predicted_text))