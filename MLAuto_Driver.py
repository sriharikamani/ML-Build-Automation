
##########################################################################################################################
#                                                    SMART AUTO ML TOOL
# 
# Validates the possible options based the problem type (Classification & Regression) and the final outcome will be 
# different metrics stats to help data scientists to decide on which model with combination of options to take it forward 
# without extensive programming knowledge            
##########################################################################################################################
                                                            

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tkinter import *
from pandastable import Table, TableModel
import tensorflow as tf

# Custom Functions
from MLAuto_Functions import *
from MLAuto_DataSelect import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
tf.keras.backend.clear_session()
###################################################
# Generic routine the show the output on the screen
###################################################
class TestApp(Frame):
    def __init__(self, parent=None):
        self.parent = parent
        Frame.__init__(self)
        self.main = self.master
        self.main.geometry('600x400+200+100')
        self.main.title('Table app')
        f = Frame(self.main)
        f.pack(fill=BOTH,expand=1)
        self.table = pt = Table(f, dataframe=df1,showtoolbar=True, showstatusbar=True)
        pt.show()
        return

###############
# Load the Data
###############

data        = Get_datasets()

if (data != {}):
    train_df    = pd.read_csv(data['train'])
    test_df     = pd.read_csv(data['test'])
    data        = {}
    dependCol   = train_df.columns[-1]

    ############################
    # Determine the problem type
    ############################
    train_df[dependCol],problem = chkTrgtType(train_df,dependCol)

    if problem == 'Classification':
        #ClsImbal = pltClsDist(train_df,train_df.columns[-1])
        ClsImbal = 'Y'
    else:
        ClsImbal = 'N'

    ###################################################################    
    # Check and impute Missing Values (if any) without user innervation
    ###################################################################

    missing  = missingValues(train_df)
    if (missing == 'Y'):
        impute_missing_data(train_df)
        missing = missingValues(train_df)

    # Check for outliers
    cat_cols = [c for c in train_df.iloc[:, :-1].columns if train_df.iloc[:, :-1][c].dtype=='O']
    num_cols = [n for n in train_df.iloc[:, :-1].columns if n not in cat_cols]
    detectOutliers(train_df[num_cols],'CHK')
    del cat_cols
    del num_cols
    
    ##################### 
    # Show the GUI Screen
    ##################### 
    
    Options        = Auto_ml(train_df,train_df.columns[-1],problem)
    
    #######################################
    # Process based on the selected options
    #######################################
    if ((Options['ComnFeat'] != 'N') | (Options['DefaultFeat'] != 'N') | (Options['ComnCorFeat'] != 'N')): # If No EXIT 
        
        ###########################################
        # SETP 1: Remove unwanted columns (if any)
        ###########################################

        if (Options['RmvCols'] != []):

            train_df       = train_df.drop(Options['RmvCols'],axis=1)
            test_df        = test_df.drop(Options['RmvCols'],axis=1)    

        ###########################################
        # SETP 2: Perform Data Binning/bucketing
        ###########################################

        if (Options['Binning'] != ''):
            for feature, bins in Options['Binning'].items():
                train_df[feature+'_bin'] = pd.qcut(train_df[feature], bins, labels=False, duplicates='drop')
                test_df[feature+'_bin']  = pd.qcut(test_df[feature], bins, labels=False, duplicates='drop')
                train_df                 = train_df.drop(feature,axis=1)
                test_df                  = test_df.drop(feature,axis=1)
                #train_df[feature+'_bin'] = train_df[feature+'_bin'].astype('object')


        #########################################################################################
        # STEP 3: Data preparation methods to handle Imbalanced data with Traditional split ONLY
        #########################################################################################

        if ( ((ClsImbal == 'Y') | (Options['ImbFlg'] == 'Y') | (problem == 'Regression')) and (Options['dfultSplit'] == 'Y')):

            X                                = train_df.drop(dependCol, axis=1)
            y                                = train_df[dependCol]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            X_train,y_train                  = procesImbal(Options,X_train,pd.DataFrame(y_train),dependCol)

        #############################    
        # In case of Cross Validation
        #############################

        if ( (Options['strtkFld'] == 'Y') | (Options['kFld'] == 'Y') | (Options['rrSplit'] == 'Y')):

            X_train                          = train_df.drop(dependCol, axis=1)
            y_train                          = train_df[dependCol]

            # Dummy X_test & y_test for smooth execution in routines
            X_test                           = pd.DataFrame(columns=test_df.columns)
            y_test                           = pd.DataFrame(columns=test_df.columns)


        #######################################################
        # STEP 4: Seperate Numerical and categorical attributes
        #######################################################

        cat_cols_names,num_cols_names = splitNumCat(X_train,Options)
        #################################################################
        # STEP 5: Change numerical to categorical  (from Menu Selection)
        #################################################################

        #######################################    
        # STEP 5: Detect and treat the Outliers 
        #######################################

        X_train,X_test,test_df = impOutlyr(X_train,X_test,test_df,num_cols_names,Options)

        ################################ 
        # STEP 6: Scaling the attributes
        ################################
        X_train,X_test,test_df,num_cols_names,cat_cols_names = procesScaling(Options,X_train,X_test,test_df,num_cols_names,cat_cols_names)
        all_features = num_cols_names + cat_cols_names

        ################################################################
        # STEP 7 : Feature Selection Techniques to select best features
        ################################################################

        if (Options['feaSelFlg'] == 'Y'):

            PearsonPredictors,AnovaPredictors,KBestChi2Predictors,KBestMuInfoClsfPredictors,ExttreePredictors,NonLowCorrPredictors,NonHiCorrPredictors,IgainPredictors = feaEng(X_train[cat_cols_names],X_train[num_cols_names],pd.DataFrame(y_train),Options,dependCol)
            consolidated         = getConsolidatedFeatures(PearsonPredictors,AnovaPredictors,KBestChi2Predictors,KBestMuInfoClsfPredictors,ExttreePredictors,NonLowCorrPredictors,NonHiCorrPredictors,IgainPredictors)
            #print(consolidated.iloc[:, :-1])

            ##################################################################################################################### 
            # Get common significant features from all the feature engineering methods and features from low and high correlation
            ##################################################################################################################### 

            common_features      = getCommonFeatures(consolidated)
            common_Corr_features = list(set(NonLowCorrPredictors).intersection(NonHiCorrPredictors))

        ############################################################
        # STEP 8 : Finialize the features based on Feature Selection 
        ############################################################

        if (Options['ComnFeat'] == 'Y'):
            X_train = X_train[common_features]
            X_test  = X_test[common_features]
            X_TEST  = test_df[common_features]
            FeaSel  = 'Common'


        if (Options['ComnCorFeat'] == 'Y'):
            X_train = X_train[common_Corr_features]
            X_test  = X_test[common_Corr_features]
            X_TEST  = test_df[common_Corr_features]
            FeaSel  = 'Common Correlation'

        if (Options['DefaultFeat'] == 'Y'):
            X_TEST  = test_df
            FeaSel  = 'All'

        ##################################################################################
        # STEP 9 : Build ML models based on default split or K fold Cross Validation data
        ##################################################################################

        buildML(X_train, y_train, X_test, y_test,Options,problem)

        ########################################### 
        # STEP 10 : Display the performance metrics
        ########################################### 

        if problem == 'Classification':
            df1 = pd.DataFrame({'Model': Classifier,
                                'Split':DataSplit,
                                'Features': FeaSel,
                                'Score': cvScore,
                                'Accuracy': Accuracy, 
                                'Recall': Recall,
                                'Precision': Precision,
                                'F1': F1,
                                'Logloss': Logloss})
            df1 = df1.sort_values(['Logloss'], ascending=[True])

        else:
            df1 = pd.DataFrame({'Model': Classifier,
                                'Split':DataSplit,
                                'Features': FeaSel,
                                'RMSE': Rmse,
                                'R-Square': R2sq})
            df1 = df1.sort_values(['R-Square'], ascending=[False])

        # Show the output
        app = TestApp()
        app.mainloop()

       
 ### END OF THE CODE ###





