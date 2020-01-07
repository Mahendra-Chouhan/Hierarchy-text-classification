#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 04:26:39 2019

@author: mahi-1234
"""
import fire
import json
import logging
import pandas
from tqdm import tqdm
# all hierarchy classification import
from src.doc_classification.classification_hierarchy import classification_hierarchy
# all text classification import
from src.doc_classification.classifiers.L1_Classification import l1_classification
from src.doc_classification.classifiers.L2_Classification import l2_classification
from src.doc_classification.classifiers.L3_Classification import l3_classification
# para2vector model
from src.para2vec.build_para2vec import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class hirarchy_classifiers(object):
    def __init__(self, datapath, logger=None):
        self.logger = logger
        self.create_hirarchy_structure(datapath)
    
    def get_network_detail(self):
        return self.classification_hierarchy_obj.network_details()
    
    def train_para2vec(self):
        build_model_obj = build_model()
        build_model_obj.main(model_name="Third_model", 
                         raw_data_path="src/raw_data",
                         model_path = "src/para2vec/models",
                         csv_file_name = "PEA_renamed_file_report.csv")
    def __read__train_data(self, datapath):
        # return the data 
        pass
    
    def create_hirarchy_structure(self, datapath):
        # all classification objects
        l1_obj = l1_classification(self.logger)
        l2_obj = l2_classification(self.logger)
        l3_obj = l3_classification(self.logger)
        # all classification details
        train_data = self.__read__train_data(datapath)
        # group by L1 
        
        # group by L2
        
        # group by l3
        
        basic_details = pea_basic_obj.get_classifier_default_details()
        other_details = pea_other_obj.get_classifier_default_details()
        main_details = main_clf_obj.get_classifier_default_details()
        support_doc_details = support_doc_obj.get_classifier_default_details()
        # creating Hirarchy structure.
        l1_obj["is_root"] = True
        # add model object into structure
        main_details["model_object"] = main_clf_obj
        support_doc_details["model_object"] = support_doc_obj
        basic_details["model_object"] = pea_basic_obj
        other_details["model_object"] = pea_other_obj
        # main sub classifiers
        main_sub_classifiers_support = {"model_name": support_doc_details["model_name"],
         "version": support_doc_details["version"]}
        main_sub_classifiers_basic = {"model_name": basic_details["model_name"],
         "version": basic_details["version"]}
        main_details["child"]["sub_classifiers"].append(main_sub_classifiers_support)
        main_details["child"]["sub_classifiers"].append(main_sub_classifiers_basic)
        # pea basic sub classifiers
        basic_sub_classifier = {"model_name": other_details["model_name"],
         "version": other_details["version"]}
        basic_details["child"]["sub_classifiers"].append(basic_sub_classifier)
        # creating Hirarchy structure.
        hierarchy_strct = []
        hierarchy_strct.append(main_details)
        hierarchy_strct.append(support_doc_details)
        hierarchy_strct.append(basic_details)
        hierarchy_strct.append(other_details)

        self.classification_hierarchy_obj = classification_hierarchy(self.logger)
        self.classification_hierarchy_obj.create_graph(hierarchy_strct)
        network_detail = self.classification_hierarchy_obj.network_details()
        print(self.logger)
        logger.info(json.dumps(network_detail, indent=2))
#        self.classification_hierarchy_obj.save_graph("hirarchy_structure.png")
        
    
    def train_model(self, model_name, model_version):
        self.classification_hierarchy_obj.train_model(model_name,
                                                      model_version)
        all_model_details = self.get_network_detail()
        logger.debug(json.dumps(all_model_details, indent=2))    
        
    def train_model_all(self, root_model=None):
        # train all the models
        self.classification_hierarchy_obj.train_all_model(root_model)
        all_model_details = self.get_network_detail()
        results = json.dumps(all_model_details, indent=2)
        with open('results/result.json', 'w+') as outfile:
            json.dump(all_model_details, outfile, indent=2)
        logger.debug(results)

    def prediction(self, doc_id):
        prediction = self.classification_hierarchy_obj.prediction(doc_id)
        logger.debug(prediction)
        return prediction
        
    
    def predict(self, model_name, model_version, doc_id):
        self.classification_hierarchy_obj.predict(model_name, model_version,
                                                  doc_id)
    def predict_all(self):
        # predict all the records in csv
        csv_path = "src/raw_data"
        csv_name = "PEA_renamed_file_report.csv"
        all_data_df = pandas.read_csv(csv_path+"/"+csv_name)
        total_file = 0
        total_correct = 0
#        size = 100#len(all_data_df)
#        x_vec = np.linspace(0,1,size+1)[0:-1]
#        y_vec = np.zeros(len(x_vec))
#        line1 = []
        approved_lables = []
        filtered_df = all_data_df[all_data_df.file_type.isin(approved_lables)]
        with tqdm(total=len(filtered_df)) as pbar:
            for index, row in filtered_df.iterrows():
                pbar.update(1)
                total_file += 1
                prediction = self.prediction(row['file_path'])
                logger.info("Predicted  is {} and real is {}".format(prediction,
                             row['file_type']))
                if prediction == row['file_type']:
                    total_correct += 1
                total_accuracy = total_correct / total_file
#                y_vec[-1] = total_accuracy
#                line1 = live_plotter(x_vec,y_vec,line1)
#                y_vec = np.append(y_vec[1:],0.0)
                pbar.set_description("Total Documents: {} Accuracy: {}".format(total_file,
                            total_accuracy))
        final_accuracy = total_correct / total_file
        print("Final Accuracy is{}".format(final_accuracy))



import matplotlib.pyplot as plt
import numpy as np

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.001):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    # return line so we can update it again in the next iteration
    return line1





if __name__ == "__main__":
    fire.Fire(hirarchy_classifiers)
