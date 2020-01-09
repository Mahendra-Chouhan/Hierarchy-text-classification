# Hierarchy Text Classification

## About / Synopsis

* Hierarchy Text Classification for DBPedia.
* Conveting Raw text into Vector using Para2Vec.
* Create End to End development for Text classification.
* Project status: Development

## About Data
DBpedia (from "DB" for "database") is a data aiming to extract structured content from the information created in Wikipedia. This is an extract of the data that provides taxonomic, hierarchical categories ("classes") for 342,782 wikipedia articles. There are 3 levels L1, L2 and L3 with 9, 70 and 219 classes respectively.

![sample image](https://github.com/Mahendra-Chouhan/hierarchy-text-classification/blob/master/images/data_sample.png)

## Folder Structure
folder structure of project.

## Flow Diagram


![data Structure image](https://github.com/Mahendra-Chouhan/hierarchy-text-classification/blob/master/images/hirarchy_text_classification_flow.png)



## Installation
Its coded in python3.5 for all dependent libraies please check requirement.txt

## Result
The Exepriment is performed on Test file.
Model we used is logistic regression for all levels.

| Model Used          | Total Test Documents | L1 Correct | L2 Correct | L3 Correct  | Total Accuracy |
|---------------------|----------------------|------------|------------|-------------|----------------|
| Logistic Regression | 60794                | 32731       | 10618       | 3879        | 6.38 %         |

