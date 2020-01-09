#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 04:23:26 2019
@author: mahi-1234
This file is for defining the hierarchy flow of document classification
"""
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import copy
from smart_open import open
import logging


class classification_hierarchy():
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.classification_graph = nx.DiGraph()
        self.lable_dictoinary = {}
        self.lables = []

    def add_node(self, model_name, model_details):
        self.classification_graph.add_node(model_name)
        attrs = {model_name: model_details}
        nx.set_node_attributes(self.classification_graph, attrs)

    def add_edge(self, parent_node, child_node, edge_name=None):
        if edge_name is None:
            self.classification_graph.add_edge(parent_node, child_node)

    def __get_node_name(self, model_name, version):
        node_name = "{}_V_{}".format(model_name, str(version))
        return node_name

    def create_graph(self, hierarchy_strcture=None):
        # PEA Basic classification
        for node in hierarchy_strcture:
            node_name = self.__get_node_name(node["model_name"],
                                                node["version"])
            self.add_node(node_name, node)
        for node in hierarchy_strcture:
            sub_classifiers = node["child"]["sub_classifiers"]
            if len(sub_classifiers) > 0:
                for sub_clf in sub_classifiers:
                    parent_node = self.__get_node_name(node["model_name"],
                                                node["version"])
                    child_node = self.__get_node_name(sub_clf["model_name"],
                                                sub_clf["version"])
                    self.add_edge(parent_node, child_node)

    def get_node(self, searched_node_name):
        # get the node details in node name.
        all_node = self.__get_all_nodes()
        for node in all_node:
            node_name, node_details = node
            if node_name == searched_node_name:
                return node_details
        return None

    def __get_all_nodes(self):
        return self.classification_graph.nodes.data()

    def __find_root_node(self):
        all_node = self.__get_all_nodes()
        for node in all_node:
            node_name, node_details = node
            if node_details["is_root"]:
                return node_name

    def save_graph(self, image_path):
        nx.draw(self.classification_graph, with_labels=True)
        plt.savefig(image_path)

    def network_details(self):
        all_nodes = copy.deepcopy(self.__get_all_nodes())
        all_model_details = []
        for node in all_nodes:
            mode_name, model_details = node
            model_details["model_object"] = None
            all_model_details.append(model_details)
        return all_model_details


    def get_connected_node(self, parent_node):
        # get the all connected nodes to this parent node.
        return list(nx.dfs_preorder_nodes(self.classification_graph,
                                   parent_node))

    def get_lables(self, parent_model_name, sub_classes, lables):
        if len(sub_classes) != 0 and len(lables) > 0:
            self.lable_dictoinary[parent_model_name+"--lables"] = tuple(lables)
        if len(sub_classes) == 0 and len(lables) > 0:
            self.lable_dictoinary[parent_model_name+"--lables"] = tuple(lables)
#            for lable in lables:
#                self.lable_dictoinary[lable] = lable
        for sub_class in sub_classes:
            model_name = sub_class["model_name"]
            version = sub_class["version"]
            model_node_name = self.__get_node_name(model_name, version)
            node_detail = self.get_node(model_node_name)
            node_sub_class = node_detail["child"]["sub_classifiers"]
            node_lable = node_detail["child"]["lables"]
            self.lables = []
            self.get_all_lable(node_sub_class, node_lable)
            self.lable_dictoinary[model_node_name] = tuple(self.lables)

    def get_all_lable(self, sub_clfs, lables):
        self.lables.extend(lables)
        if len(sub_clfs) == 0:
            return
        for sub_class in sub_clfs:
            model_name = sub_class["model_name"]
            version = sub_class["version"]
            model_node_name = self.__get_node_name(model_name, version)
            node_detail = self.get_node(model_node_name)
            node_sub_class = node_detail["child"]["sub_classifiers"]
            node_lable = node_detail["child"]["lables"]
            self.get_all_lable(node_sub_class, node_lable)

    def train_all_model(self, root_node=None):
        # train all models.
        if root_node == None:
            root_node = self.__find_root_node()
        # all other connect node then root node.
        all_node = self.get_connected_node(root_node)
        for node in all_node:
            model_name, version = node.split("_V_")
            self.train_model(model_name, version)

    def train_model(self, model_name, version, model_path=None):
        # train one model in hierarchy. If path is not given old model is been replaced
        model_node_name = self.__get_node_name(model_name, version)
        node_details = self.get_node(model_node_name)
        model_obj = node_details["model_object"]
        node_lables = node_details["child"]["lables"]
        node_sub_clf = node_details["child"]["sub_classifiers"]
        self.lable_dictoinary = {}
        self.get_lables(model_node_name, node_sub_clf, node_lables)
        lable_dictoinary = self.lable_dictoinary
        model_obj.set_classes(lable_dictoinary)
        model_obj.define_model(node_details["model_name"],
                               node_details["model_path"],
                               node_details["version"])
        csv_df = model_obj.get_data(data_path=node_details["data_path"],
                                    file_name=node_details["data_file_name"])
        csv_df = model_obj.embedding_all(para2vec_model_name=node_details["other_details"]["para2vec_file_name"],
                                               para2vec_model_path=node_details["other_details"]["para2Vec_path"],
                                               csv_df = csv_df,
                                               raw_data_path=node_details["data_path"])
        is_success, evaluation = model_obj.train_and_evaluation(csv_df,
                                                              feature_name="para_2_vec",
                                                              lable_name="l3")
        if is_success:
            searched_node_name = self.__get_node_name(model_name, version)
            self.classification_graph.nodes[searched_node_name]["update_date"] = datetime.now().strftime("%H:%M:%S.%f - %b %d %Y")
            self.classification_graph.nodes[searched_node_name]["evaluation"] = evaluation

    def prediction(self, raw_text):
        root_node = self.__find_root_node()
        root_details = self.get_node(root_node)
        model_name = root_details["model_name"]
        model_version = root_details["version"]
        all_precitions = []
        pridicted_lable = self.prediction_by_model(model_name, model_version,
                                                   raw_text)
        self.logger.info(pridicted_lable)
        node_details = self.get_node(pridicted_lable[0])
        #self.logger.info(node_details)
        all_precitions.append(pridicted_lable[0])
        while node_details != None:
            self.logger.debug(node_details)
            model_name = node_details["model_name"]
            model_version = node_details["version"]
            pridicted_lable = self.prediction_by_model(model_name, 
                                                       model_version, raw_text)
            self.logger.info(pridicted_lable)
            all_precitions.append(pridicted_lable[0])
            node_details = self.get_node(pridicted_lable[0])
        return all_precitions    
    
    def prediction_by_model(self, model_name, version, raw_text,
                            raw_data_path=None):
        model_node_name = self.__get_node_name(model_name, version)
        node_details = self.get_node(model_node_name)
        
        
        model_obj = node_details["model_object"]
        para_2_vec = model_obj.embedding(para2vec_model_name=node_details["other_details"]["para2vec_file_name"],
                                               para2vec_model_path=node_details["other_details"]["para2Vec_path"],
                                               raw_text=raw_text)
        model_obj.define_model(node_details["model_name"],
                               node_details["model_path"],
                               node_details["version"])
        predicted_lable = model_obj.predict_label(para_2_vec.reshape(1, -1))
        return predicted_lable