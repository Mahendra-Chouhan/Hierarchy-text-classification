#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 04:26:39 2019

@author: mahi-1234
"""
import fire
import json
import logging
import pandas as pd
from tqdm import tqdm
# all hierarchy classification import
from src.doc_classification.classification_hierarchy import classification_hierarchy
# all text classification import
from src.doc_classification.classifiers.L1_Classification import l1_classification
from src.doc_classification.classifiers.L2_Classification import l2_classification
from src.doc_classification.classifiers.L3_Classification import l3_classification
# para2vector model
from src.para2vec.build_para2vec import build_model

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class hirarchy_classifiers(object):
    def __init__(self, datapath="data/DBPEDIA_train.csv", logger=None):
        self.logger = logger
        self.datapath = datapath
        self.create_hirarchy_structure()
    
    def get_network_detail(self):
        return self.classification_hierarchy_obj.network_details()
    
    def train_para2vec(self):
        build_model_obj = build_model()
        build_model_obj.main(model_name="dbpedia_para2vec_model", 
                         raw_data_path="data",
                         model_path = "src/para2vec/models",
                         csv_file_name = "DBPEDIA_train.csv")
    def __read__train_data(self, datapath):
        # return the data 
        return pd.read_csv(datapath)
    
    def __get_label_structure(self, train_df):
        groupse = train_df.groupby(["l1", "l2", "l3"])
        all_groups = groupse.groups.keys()
        final_dict = {}
        for group in all_groups:
            l1, l2, l3 = group
            if l1 in final_dict:
                if l2 in final_dict[l1]:
                    final_dict[l1][l2].append(l3)
                if l2 not in final_dict[l1]:
                    final_dict[l1][l2] = [l3]
            if l1 not in final_dict:
                final_dict[l1] = {l2:[l3]}
        return final_dict

    def analysis_structure(self, tree_structure):
        # analysis the structure and removed ambuguity.
        removed_node_uuid = []
        for node_number in range(len(tree_structure)):
            node = tree_structure[node_number]
            child = node["child"]
            if len(child["sub_classifiers"]) >1 or len(child["lables"]) > 1:
                continue
            if len(child["sub_classifiers"]) <= 1:
                if len(child["sub_classifiers"]) == 1 and len(child["lables"]) == 1:
                    continue
                removed_node_uuid.append(node["model_uuid"])
                logger.info(node["model_uuid"])
                logger.info(node)
                #del tree_structure[node_number] 
        return_struct = list(filter(lambda i: i['model_uuid'] not in removed_node_uuid, tree_structure))        
        return return_struct
    
    def __analyis_L2(self, l2_node):
        child = l2_node["child"]
        if len(child["sub_classifiers"]) >1 or len(child["lables"]) > 1:
            return True
            if len(child["sub_classifiers"]) <= 1:
                if len(child["sub_classifiers"]) == 1 and len(child["lables"]) == 1:
                    return True
                return False
    
    def create_hirarchy_structure(self):
        # all classification objects
        datapath = self.datapath
        l1_obj = l1_classification(self.logger)
        l2_obj = l2_classification(self.logger)
        l3_obj = l3_classification(self.logger)
        # all classification details
        train_data_df = self.__read__train_data(datapath)
        classes_structure = self.__get_label_structure(train_data_df)
        model_path = "src/doc_classification/models"
        
        l1_details = l1_obj.get_classifier_default_details(model_name="L1", 
                                                           model_path=model_path,
                                                           version=0.01)
        l1_details["is_root"] = True
        l1_details["model_object"] = l1_obj
        hierarchy_strct = []
        self.l2_alias = {}
        self.l3_alias = {}
        single_lables_count = 0
        for l1, l2_structure in classes_structure.items():
            L2_model_name = "L2_{}".format(l1)
            l2_details = l2_obj.get_classifier_default_details(model_name=L2_model_name, 
                                                           model_path=model_path,
                                                           version=0.01)
            # L2 sub classifiers
            for l2, l3_lables in l2_structure.items():
                if len(l3_lables) == 1:
                    single_lables_count += 1
                    self.l3_alias[l2] = l3_lables[0]
                    l2_details["child"]["lables"].extend(l3_lables)
                if len(l3_lables) > 1:
                    L3_model_name = "L3_{}".format(l2)
                    l3_details = l3_obj.get_classifier_default_details(model_name=L3_model_name, 
                                                               model_path=model_path,
                                                               version=0.01)
                    l3_details["model_object"] = l3_obj
                    l3_detail_sub_cls = {"model_name": l3_details["model_name"],
                                     "version": l3_details["version"]}
                    l2_details["child"]["sub_classifiers"].append(l3_detail_sub_cls)
                    l3_details["child"]["lables"].extend(l3_lables)
                    hierarchy_strct.append(l3_details)
            l2_response = self.__analyis_L2(l2_details)
            if not l2_response:
                self.l2_alias[l1] = {l2:l2_details["child"]["lables"][0]}
                l1_details["child"]["lables"].append(l2_details["child"]["lables"][0])
            if l2_response:
                l2_details["model_object"] = l2_obj
                l2_detail_sub_cls = {"model_name": l2_details["model_name"],
                                     "version": l2_details["version"]}
                l1_details["child"]["sub_classifiers"].append(l2_detail_sub_cls)
                hierarchy_strct.append(l2_details)
        hierarchy_strct.append(l1_details)
        hierarchy_strct = self.analysis_structure(hierarchy_strct)
        self.classification_hierarchy_obj = classification_hierarchy(self.logger)
        self.classification_hierarchy_obj.create_graph(hierarchy_strct)
        network_detail = self.classification_hierarchy_obj.network_details()
        # logger.info(json.dumps(network_detail, indent=2))
        logger.info("total one lable classes {}".format(single_lables_count))
        logger.info(self.l2_alias)
        logger.info(self.l3_alias)
        with open("structure.json", 'w+') as f:
            json.dump(network_detail, f, indent=2)
        #self.classification_hierarchy_obj.save_graph("hirarchy_structure.png")

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
        csv_path = "data"
        csv_name = "DBPEDIA_test.csv"
        all_data_df = pd.read_csv(csv_path+"/"+csv_name)
        total_file = 0
        total_correct = 0
        l1_correct = 0
        l2_correct = 0
        l3_correct = 0
        all_data_df["l1_predicted"] = ""
        all_data_df["l2_predicted"] = ""
        all_data_df["l3_predicted"] = ""
#        size = 100#len(all_data_df)
#        x_vec = np.linspace(0,1,size+1)[0:-1]
#        y_vec = np.zeros(len(x_vec))
#        line1 = []
        approved_lables = []
        filtered_df = all_data_df#[all_data_df.l3.isin(approved_lables)]
        with tqdm(total=len(filtered_df)) as pbar:
            for index, row in filtered_df.iterrows():
                pbar.update(1)
                total_file += 1
                all_predictions = self.prediction(row['text'])
                prediction_len = len(all_predictions)
                predicted_l1 = ""
                predicted_l2 = ""
                predicted_l3 = ""
                if prediction_len == 3:
                    predicted_l1 = all_predictions[0].split("_")[1]
                    if predicted_l1 == row['l1']:
                        l1_correct += 1
                    predicted_l2 = all_predictions[1].split("_")[1]
                    if predicted_l2 == row['l2']:
                        l2_correct += 1
                    predicted_l3 = all_predictions[2]
                    if predicted_l3 == row['l3']:
                        l3_correct += 1                                  
                if prediction_len == 2:
                   predicted_l1 = all_predictions[0].split("_")[1]
                   if predicted_l1 == row['l1']:
                       l1_correct += 1
                   predicted_l2 = [key for key, value in self.l3_alias.items() if value == all_predictions[1]][0]
                   if predicted_l2 == row['l2']:
                       l2_correct += 1
                   predicted_l3 = all_predictions[1]
                   if predicted_l3 == row['l3']:
                       l3_correct += 1 
                if prediction_len == 1:
                    results = [(l1, l2) for l1, values in self.l2_alias.items() for l2, value in values.items() if value == all_predictions[0]][0]
                    predicted_l1 = results[0]
                    if predicted_l1 == row['l1']:
                        l1_correct += 1
                    predicted_l2 = results[1]
                    if predicted_l2 == row['l2']:
                        l2_correct += 1
                    predicted_l3 = all_predictions[0]
                    if predicted_l3 == row['l3']:
                        l3_correct += 1
                logger.info("Predicted are:{} , {},  {} and real are {}, {}, {}".format(predicted_l1,
                            predicted_l2, predicted_l3, row['l1'], row['l2'],
                            row['l3']))
                all_data_df.at[index, 'l1_predicted'] = predicted_l1
                all_data_df.at[index, 'l2_predicted'] = predicted_l2
                all_data_df.at[index, 'l3_predicted'] = predicted_l3
                total_accuracy = total_correct / total_file
#                y_vec[-1] = total_accuracy
#                line1 = live_plotter(x_vec,y_vec,line1)
#                y_vec = np.append(y_vec[1:],0.0)
                pbar.set_description("L1 Correct: {} L2 Correct: {} L3 Correct: {}".format(l1_correct,
                            l2_correct, l3_correct))
        final_accuracy = total_correct / total_file
        all_data_df.to_csv("results/{}".format(csv_name))
        print("Final Accuracy is{}".format(final_accuracy))


if __name__ == "__main__":
    fire.Fire(hirarchy_classifiers)
