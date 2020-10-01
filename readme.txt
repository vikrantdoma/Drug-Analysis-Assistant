
3 datasets are from kaggle

[URLS]
Primary dataset 
1) https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018 [1]

Secondary datasets 
1) https://data.cdc.gov/NCHS/NCHS-Potentially-Excess-Deaths-from-the-Five-Leadi/vdpk-qzpr [2]
2) https://www.kaggle.com/plarmuseau/sdsort [3]
 

[WARNING] 
in Analysis_new.ipynb there is a graph which takes 40min to 1 HR for execution because of pipelined classifier training over 5-fold cross validation i.e CV = 5
(please read comment " #WARNING 40min to 1 HR required for execution " before executing that box)


[Execution files 5]
[FOLLOW THIS ORDER of steps]
Please inatall nltk library using nltk.download() and remaining libraries

1) Execute " Analysis_new.ipynb " requires dataset " [1] " both train.csv and test.csv and produces 'drug_analysis.csv' 
2) Execute " analysis3.ipynb " requires " [2] " add dataset according to the names used in code
3) Execute " Analysis_dataset3.ipynb " requires [3] 
4) Execute " BDProjML.ipynb " requires 'drug_analysis.csv' created by file "Analysis_new.ipynb" at the begining and towards the end requires " [1] " both train.csv and test.csv 
5) Execute " drug_recom " requires 'drug_analysis.csv' created by file "Analysis_new.ipynb" in 2 places add path accordingly

[Execution Notes]
Main analysis is done on the jupyter files which is to executed in the above order only
Analysis of ML techniques with test cases is in step 4
Final recomendation system is in [drug_recom.py] execute as a python file (average time till system ready 18~20 min)
