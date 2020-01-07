#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 01:40:10 2019

@author: mahi-1234
"""
import pandas as pd
def load_csv(logger, raw_data_path, csv_file_name):
    # load csv file for processing.
    file_path = raw_data_path+"/"+csv_file_name
    print(file_path)
    pea_all_record_df = pd.read_csv(file_path)
    # logger.info("CSV uploaded")
    return pea_all_record_df

def save_dataframe(dataframe, path, file_name, mode="w", header=True,
                   index=False):
    dataframe.to_csv(path+"/"+file_name, mode=mode, header=header, 
                     index=index)
    