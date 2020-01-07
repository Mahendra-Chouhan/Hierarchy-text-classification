# -*- coding: utf-8 -*-

def create_para2Vec():
    from src.para2vec.build_para2vec import build_model
    build_model_obj = build_model()
    build_model_obj.main(model_name="Second_model", 
                         raw_data_path="src/raw_data",
                         model_path = "src/para2vec/models",
                         csv_file_name = "PEA_renamed_file_report.csv")

def predict_para2vec():
    from src.para2vec.predict_para2vec import modeL_prediction
    model_path = "src/para2vec/models/Second_model.pickle"
    predicted_text = """
    545sadsa ads sadsad sad adsadsad4s4s44s sd sdd 4ds d sd4sd sd4 sd 4sd4
    """
    modeL_prediction_obj = modeL_prediction()
    modeL_prediction_obj.load_model(model_path)
    print(modeL_prediction_obj.get_para_2_vector(predicted_text))


def text_classification():
    from src.doc_classification.classifiers.PEA_Auth_text_classification import pea_auth_classification
    model_name = "PEA_Basic"
    model_path = "src/doc_classification/models"
    PEA_Auth_text_classification_obj = pea_auth_classification()
    approved_lables = [("PEACho",
                        "PEABene",
                        "PEAChoAuth",
                        "PEAConAuth",
                        "PEAConForm",
                        "PEAElect",
                        "PEAOpt",
                        "PEAPChoAuthForm",
                        "PEARetdAuth",
                        "PEARetdElect"),
                        ("PEAuth",)]
    PEA_Auth_text_classification_obj.set_classes(approved_lables)
    PEA_Auth_text_classification_obj.define_model(model_name, model_path)
    csv_df = PEA_Auth_text_classification_obj.get_data(data_path = "src/raw_data", 
                                              file_name = "PEA_renamed_file_report.csv")
#    csv_df = load_csv(raw_data_path = "src/raw_data", csv_file_name = "featured.csv", logger=None)
    csv_df = PEA_Auth_text_classification_obj.embedding_all(para2vec_model_name="Second_model.pickle",
                                               para2vec_model_path = "src/para2vec/models",
                                               csv_df = csv_df,
                                               raw_data_path = "src/raw_data")
    
    PEA_Auth_text_classification_obj.train_and_evaluation(csv_df)
    
def create_classification_graph():
    # all hierarchy classification import
    from src.doc_classification.classification_hierarchy import classification_hierarchy
    # all text classification import
    from src.doc_classification.classifiers.Main_Classification import main_classification
    from src.doc_classification.classifiers.Support_Document_Classification import support_document_classification
    from src.doc_classification.classifiers.PEA_Basic_Classification import pea_basic_classification
    from src.doc_classification.classifiers.PEA_Other_Classification import pea_other_classification

    # all classification objects
    pea_basic_obj = pea_basic_classification()
    pea_other_obj = pea_other_classification()
    main_clf_obj = main_classification()
    support_doc_obj = support_document_classification()
    # all classification details
    basic_details = pea_basic_obj.get_classifier_default_details()
    other_details = pea_other_obj.get_classifier_default_details()
    main_details = main_clf_obj.get_classifier_default_details()
    support_doc_details = support_doc_obj.get_classifier_default_details()
    # creating Hirarchy structure.
    main_details["is_root"] = True
    # add model object into structure
#    main_details["model_object"] = main_clf_obj
#    support_doc_details["model_object"] = support_doc_obj
#    basic_details["model_object"] = pea_basic_obj
#    other_details["model_object"] = pea_other_obj
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
    import json 
    print(json.dumps(hierarchy_strct, indent=2))
    classification_hierarchy_obj = classification_hierarchy()
    classification_hierarchy_obj.create_graph(hierarchy_strct)
    classification_hierarchy_obj.network_details()
#    classification_hierarchy_obj.train_model(main_details["model_name"],
#                                             main_details["version"])
    classification_hierarchy_obj.save_graph("hirarchy_structure.png")

create_classification_graph()
