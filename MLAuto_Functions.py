####################################
# Generic Functions for AUTO ML Tool
####################################

import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os
from tkinter import *
from tkinter import messagebox
import tkinter as tk
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score,log_loss,classification_report,recall_score,precision_score,accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

##################
# Global Variables
##################

Classifier = []
DataSplit  = [] 
cvScore    = []
Accuracy   = []
Recall     = []
Precision  = []
F1         = []
Logloss    = []
Rmse       = []
R2sq       = []

impute_missing = 'N'
errFlg         = 'N'
    

tf.keras.backend.clear_session()
#####################
# Early Stop Routine
#####################

class accuracyTresholdCallback(tf.keras.callbacks.Callback): 
    
    def on_epoch_end(self, epoch, logs={}): 
        
        ACCURACY_THRESHOLD = 0.94
        
        if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):   
            print("\nReached %2.2f%% accuracy, so stopping training" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True
        #else:
        #    print("\nUnable to reach desired Accuracy yet") 
            
#####################################################################################################################
# Routine to verify Dependent attribute type and determine problem type based on the length of the last column values
#####################################################################################################################

def chkTrgtType(df,dependent_Feau):

    #global targetCol_le
    global problemType
    
    measurer = np.vectorize(len)
    
    if (df[dependent_Feau].dtypes != 'int64'):

        from sklearn.preprocessing import LabelEncoder
        targetCol_le  = LabelEncoder()    
        df[dependent_Feau] = targetCol_le.fit_transform(df[dependent_Feau])  
        
        problemType = 'Regression' if measurer(df[dependent_Feau].astype(str)).max(axis=0) > 1 else 'Classification'

    return(df[dependent_Feau], 'Regression' if measurer(df[dependent_Feau].astype(str)).max(axis=0) > 1 else 'Classification')

#####################################################################################################################
# Routine to get the common significant features from all the feature engineering methods
####################################################################################################################

def getCommonFeatures(consolidated):
    
    common = ''
    inLoop = 'N'

    for i in range(len(consolidated)):
        
        if (consolidated['Name'][i] != 'Pearson'):
            
            if inLoop == 'N':
                common = common + '[value for value in ' + str(consolidated['FeaEng'][i])
                nxtStr = 'Y'
                inLoop = 'Y'
                
            if ((i >= 1) and (nxtStr == 'Y')):
                common = common + ' if value in '+ str(consolidated['FeaEng'][i])
                nxtStr = 'N'

            if ((i > 1) and (nxtStr == 'N')):
                common = common + ' and value in '+ str(consolidated['FeaEng'][i])
    common     = eval(common + ']')
    #hypothesis = eval('[value for value in PearsonPredictors if value in AnovaPredictors]')
    
    return(common)


############################
# START OF GUI SCREEN DESIGN 
############################

def Auto_ml(train_df,target_col,problem):
    
    global main_col_lst
    global cat_col_lst
    global num_col_lst
    global fin_df
    
    tf.keras.backend.clear_session()
     
    target_col = train_df.columns[-1]
    main_col_lst = train_df.columns.to_list()
    #main_col_lst.sort()  
    
    # categorical and Numerical feature lists
    cat_col_lst = [c for c in train_df.columns if train_df[c].dtype=='O']
    num_col_lst = [n for n in train_df.columns if train_df[n].dtype!='O']
    
    def imb_check(data,tarcol):
        tar_vcnt = data[tarcol].value_counts(normalize=True).values 
        for x in tar_vcnt:
            if x > 0.65:
                return 'Y'

    root = tk.Tk()   
    root.resizable(False, False)  
    window_height = 700 
    window_width  = 1300
    screen_width  = root.winfo_screenwidth() 
    screen_height = root.winfo_screenheight() 
    x_cordinate   = int((screen_width/2) - (window_width/2)) 
    y_cordinate   = int((screen_height/2) - (window_height/2)) 
    root.geometry("{}x{}+{}+{}".format(window_width,window_height, x_cordinate, y_cordinate)) 
    path          = os.getcwd()
    Psx_img       = PhotoImage(file = path + '\psx.png')  
    Psx_imbal     = PhotoImage(file = path + '\imbal.png')
    root.title('ML Model Automation')
    
    ################
    # Frame Creation
    ################
   
    header = Frame(root, width=1300, height=60, bg="white")
    header.grid(columnspan=3, rowspan=2, row=0)
    header_heading = Label(root,text = "ML MODEL BUILD  AUTOMATION",bg ='#FFFFFF',fg= '#004488', font = ('Poppins',30,"bold"))   
    header_heading.place(x = 380,y = 5)
    header_1 = Label(root,image = Psx_img)
    header_1.place(x=0,y=1)

    time = Label(root, text=f"{'{0:%d-%m-%Y %H:%M %p}'.format(datetime.datetime.now())}",bg ='#FFFFFF',fg= '#004488',font = ('Poppins',12,"bold"))
    time.place(x = 1130,y=30)

    main_content3 = Frame(root,highlightbackground="#f80", bg="#048", highlightcolor="#f80", highlightthickness=2,width=1300, height=580)   
    main_content3.grid()

    main_content3 = Frame(root,highlightbackground="#f80",bg="#048", highlightcolor="#f80", highlightthickness=2,width=705, height=580 )  
    main_content3.grid(columnspan=3, rowspan=1, row=2)  

    footer = Frame(root, width=1300, height=60, bg='#004488')
    footer.grid(columnspan=3, rowspan=2, row=3)

    #######################
    # Pre-Procesing Section
    #######################
    
    pre_processing_heading = Label(root, text='PRE-PROCESSING',bg = "#048", fg="#f80", font=('inherit',16,'bold','underline'), bd=15)  
    pre_processing_heading.place(x = 40,y = 65) 
    
    pre_processing_sub_heading = Label(root, text='BASIC',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15,anchor="e")  
    pre_processing_sub_heading.place(x = 80,y = 110)
        
    pp_submit    = {'OutlyrImp':'N','RmvCols':[]}    
    pp_sub_label = list(pp_submit.keys())
    pp_label     = ['Outliers Imputation', 'Remove Features', 'Numeric-2-String', 'Binning']  
    pp_var_sel   = []

    # Labels
    for x,y in zip(range(155,525,50),pp_label):  
        pp_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")
        pp_labels.place(x = 10,y = x) 
        pp_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        pp_align.place(x = 220,y = x)   
    
    # Checkbox selection
    def outlyr_selections():
            res  = outlyr_var.get()
            pp_submit['OutlyrImp'] = res

    outlyr_var = StringVar()
    outlyr_var.set('N')
    outlyr_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=outlyr_var,command =outlyr_selections)
    outlyr_cb.place(x = 235,y = 145)#220
    
    # Routine for listbox
    def select_all():
        global listbox
        listbox.select_set(0, END)
        
    def exit_btn():
        tl.destroy()

    #################
    # Remove Features 
    #################
    
    drop_feat_selections = []
    drop_var             = StringVar()
    drop_var.set('N')
    
    def selected_item_drop():
        global train_df
        global listbox
        global tl
        temp = []
        for i in listbox.curselection():
            drop_feat_selections.append(listbox.get(i))
        tl.destroy()

    def drop_feat():
        global main_col_lst
        global listbox
        global tl

        tl      = toplevel('Select Features',Psx_img)
        listbox = Listbox(tl, selectmode=MULTIPLE, highlightbackground="#004488", bg="#004488",fg='white',width =100,highlightcolor="#f80", highlightthickness=2)
        
        for val in main_col_lst:
            listbox.insert(END, val)
        listbox.place(x = 100,y = 80,width=300) 
        
        sub_button = Button(tl, text='SUBMIT',command=selected_item_drop, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sub_button.place(x = 230,y = 260,height=30)
        
        sel_button = Button(tl, text='SELECT All',command=select_all, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sel_button.place(x = 100,y = 260,height=30)

        exit_button = Button(tl, text='EXIT',command=exit_btn, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        exit_button.place(x = 330,y = 260,height=30)
        
    # Checkbox selection
    drop_feat_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=drop_var,command =drop_feat)
    drop_feat_cb.place(x = 235,y = 195)  
    
    #################### 
    # Numeric to Object 
    #################### 
    
    num_cat_selections = []
    num_cat_var = StringVar()
    num_cat_var.set('N')
    
    def selected_item_num_cat():
        global listbox
        global tl
        for i in listbox.curselection():
            num_cat_selections.append(listbox.get(i))
        tl.destroy()

    def conver_num_obj():
        global listbox
        global tl        
        
        tl              = toplevel('Select Features',Psx_img)
        listbox         = Listbox(tl, selectmode=MULTIPLE, highlightbackground="#004488", bg="#004488",fg='white', width =100,highlightcolor="#f80", highlightthickness=2)
        fin_num_col_lst = list(set(num_col_lst) - set(drop_feat_selections))
        values          = fin_num_col_lst
        values.sort()        
        
        for val in values:
            listbox.insert(END, val)
        listbox.place(x = 100,y = 80,width=300) 

        sub_button = Button(tl, text='SUBMIT',command=selected_item_num_cat, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sub_button.place(x = 230,y = 260,height=30)
        
        sel_button = Button(tl, text='SELECT All',command=select_all, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sel_button.place(x = 100,y = 260,height=30)

        exit_button = Button(tl, text='EXIT',command=exit_btn, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        exit_button.place(x = 330,y = 260,height=30)
    
    # Checkbox selection
    num_cat_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=num_cat_var,command =conver_num_obj)
    num_cat_cb.place(x = 235,y = 243)  
    
    ########## 
    # Binning   
    ########## 
    
    bin_selections = []
    bin_var        = StringVar()
    bin_var.set('N')

    def binning_selection():
        bin_tl          = toplevel('Select Features',Psx_img)
        bin_listbox     = Listbox(bin_tl, selectmode=MULTIPLE, highlightbackground="#004488", bg="#004488", fg='white', width =100,highlightcolor="#f80", highlightthickness=2)
        fin_num_col_lst = list(set(num_col_lst) - set(drop_feat_selections))
        values          = fin_num_col_lst
        values.sort()

        bin_val = tk.Text(bin_tl, highlightbackground="#004488", bg="#004488",fg='white', width =100, highlightcolor="#f80", highlightthickness=2)
        bin_val.place(x=320, y=80, width = 20, height=165)
        
        for i in range(len(values)-1):
            bin_val.insert(tk.INSERT, '0'+"\n")
        bin_val.insert(tk.INSERT, '0')
           
        for val in values:
            bin_listbox.insert(END, val)
        bin_listbox.place(x = 100, y = 80, width=200) 
        
        def fetchAllbins():
            re_li = []
            res   = bin_val.get("1.0", "end-1c")
            res   = res.splitlines() 
            res   = [int(x) for x in res]
            return res
    
        def selected_bin_item(window,li_box,features,var): #,di,key):
            
            global  binDict
            allBinlst     = fetchAllbins()
            bin_sel_index = []
            
            for i in li_box.curselection():
                bin_sel_index.append(i)
                features.append(li_box.get(i))
                
            if len(features) ==  0:
                var.set('N')
                messagebox.showinfo("Caution", "Select any feature for binning or click on EXIT")
            else:
                bins    = [allBinlst[x] for x in bin_sel_index]    
                binDict = { k:v for (k,v) in zip(features, bins)}      
                var.set('Y')
                bin_tl.destroy()
                
        def clear():
            bin_tl.destroy()
            binning_selection()

        def exit_bt():
            bin_tl.destroy()
            
        # Buttons
        sub_button = Button(bin_tl, text='SUBMIT', command= lambda:selected_bin_item(bin_tl,bin_listbox,bin_selections, bin_var) ,bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sub_button.place(x = 100,y = 255,height=30)
        
        exit_button = Button(bin_tl, text='EXIT',command=exit_bt, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        exit_button.place(x = 340,y = 255,height=30)
                            
        clear_button = Button(bin_tl, text='CLEAR',command=clear, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        clear_button.place(x = 220,y = 255,height=30)
    
    # Checkbox selection
    binning_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=bin_var,command_=binning_selection)
    binning_cb.place(x = 235,y = 293)  
         
    ################## 
    # Scaling Section 
    ################## 
   
    Scaling_heading = Label(root, text='SCALING',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15,anchor="e")     
    Scaling_heading.place(x = 80,y=350)  

    # Labels
    scale_label = ['Numerical Attributes','Categorical Attributes']
    for x,y in zip(range(400,545,50),scale_label):  
        
        ss_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")
        ss_labels.place(x = 3,y = x) 
        ss_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")
        ss_align.place(x = 225,y = x)   

    ###############################
    # Numerical Attributes List Box
    ############################### 
    
    global num_scalar_selections
    num_scalar_selections = []
    num_scalar_var = StringVar()
    num_scalar_var.set('N')

    def selected_item_ns():
        global listbox
        global tl

        for i in listbox.curselection():
            num_scalar_selections.append(listbox.get(i))
        tl.destroy()

    def num_scalar():  
        
        global listbox
        global tl

        tl        = toplevel('Select Scaling Method',Psx_img) 
        listbox   = Listbox(tl, selectmode=MULTIPLE, highlightbackground="#004488", bg="#004488",fg='white',width =100,highlightcolor="#f80", highlightthickness=2)
        ns_values = ['Normalization','Standardization','Robust Scaler']
        
        for val in ns_values:
            listbox.insert(END, val)
        listbox.place(x = 100,y = 80,width=300) 

        sub_button = Button(tl, text='SUBMIT',command=selected_item_ns, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sub_button.place(x = 150,y = 260,width=80,height=30) 
        exit_button = Button(tl, text='EXIT',command=exit_btn, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        exit_button.place(x = 250,y = 260,width=80,height=30) 
        
    # Checkbox selection
    num_scalar_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=num_scalar_var,command =num_scalar)
    num_scalar_cb.place(x = 235,y = 390)    
    
    #################################
    # Categorical Attributes List Box
    ################################# 
    
    oh_selection = []
    oh_var = StringVar()
    oh_var.set('N')
    
    def selected_item_oh():
        global listbox
        global tl
        for i in listbox.curselection():
            oh_selection.append(listbox.get(i))
        tl.destroy()
        
    def oh_sel():
        global listbox
        global tl
        
        tl             = toplevel('ONE-HOT ENCODING',Psx_img) 
        listbox        = Listbox(tl, selectmode=MULTIPLE, highlightbackground="#004488", bg="#004488",fg='white',width =100,highlightcolor="#f80", highlightthickness=2)
        oh_cat_col_lst = list(set(cat_col_lst) - set(drop_feat_selections) )
        oh_cat_col_lst = oh_cat_col_lst+num_cat_selections   
        values         = oh_cat_col_lst
        values.sort()
        
        for val in values:
            listbox.insert(END, val)
        listbox.place(x = 100,y = 80,width=300) 

        sub_button = Button(tl, text='SUBMIT',command=selected_item_oh, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sub_button.place(x = 230,y =260,height=30)
        
        sel_button = Button(tl, text='SELECT All',command=select_all, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sel_button.place(x = 100,y = 260,height=30)

        exit_button = Button(tl, text='EXIT',command=exit_btn, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        exit_button.place(x = 330,y = 260,height=30)

    
    ################  
    # Label Encoding
    ################ 
    
    le_selection = []
    le_var = StringVar()
    le_var.set('N')
    
    def selected_item_le():
        global listbox
        global tl
        for i in listbox.curselection():
            le_selection.append(listbox.get(i))
        tl.destroy()
        
    def le_sel():
        global listbox
        global tl

        tl             = toplevel('LABEL ENCODING',Psx_img) 
        listbox        = Listbox(tl, selectmode=MULTIPLE, highlightbackground="#004488", bg="#004488",fg='white',width =100,highlightcolor="#f80", highlightthickness=2)
        le_cat_col_lst = list(set(cat_col_lst) - set(drop_feat_selections))
        le_cat_col_lst = le_cat_col_lst+num_cat_selections   
        le_cat_col_lst = list(set(le_cat_col_lst)-set(oh_selection))
        values         = le_cat_col_lst
        values.sort()        
        
        for val in values:
            listbox.insert(END, val)
        listbox.place(x = 100,y = 80,width=300) 

        sub_button = Button(tl, text='SUBMIT',command=selected_item_le, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sub_button.place(x = 230,y = 260,height=30)
        
        sel_button = Button(tl, text='SELECT All',command=select_all, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sel_button.place(x = 100,y = 260,height=30)

        exit_button = Button(tl, text='EXIT',command=exit_btn, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        exit_button.place(x = 330,y = 260,height=30)


    ####################### 
    # Select Scaling Method
    #######################
    
    cat_scalar_var = StringVar()
    cat_scalar_var.set('N')
    
    def cat_scalar():
        global listbox
        global tl

        tl         = toplevel('Select Scaling Method',Psx_img)
        cs_submit  = {'one_hot':[],'le':[]}
        cs_label   = list(cs_submit.keys())
        cs_var_sel = []

        for x,y in zip([100,150],['One Hot Encodeing','Label Encoding']):
            cs_labels = Label(tl, text=y,bg="#004488",fg='white', font=('Poppins',12,'bold'), bd=15)  
            cs_labels.place(x = 80,y = x) 
            cs_align = Label(tl, text=':',bg="#004488",fg='white', font=('Poppins',12,'bold'), bd=15)  
            cs_align.place(x = 270,y = x)

        one_hot_cb = Checkbutton(tl,onvalue='Y',offvalue ='N',height = 3,bg="#004488", variable=oh_var,command =oh_sel)
        one_hot_cb.place(x = 300,y = 100)
        
        le_cb = Checkbutton(tl,onvalue='Y',offvalue ='N',height = 3,bg="#004488",variable=le_var,command =le_sel)
        le_cb.place(x = 300,y = 150)

        sel_button = Button(tl, text='Submit',command=tl.destroy, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3)
        sel_button.place(x = 200,y = 230,width=80)       
    
    # Checkbox selection
    cat_scalar_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048', variable=cat_scalar_var,command =cat_scalar)
    cat_scalar_cb.place(x = 235,y = 438)  

    ########################## 
    # Under Sampling  Section   
    ########################## 
    
    sampling_heading = Label(root, text='DATA PREPARATION AND FEATURE ENGINEERING METHODS',bg = "#048", fg="#f80", font=('inherit',16,'bold','underline'), bd=15)  
    sampling_heading.place(x = 320,y = 65) 

    Under_sam_heading = Label(root, text='UNDER  SAMPLING',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15,anchor="e")  
    Under_sam_heading.place(x = 360,y = 110)
    
    all_sam_submit = {'UndSmpDft':'N','RndUndSmplr':'N','UpSmpSMOTek':'N','OvrSmpDft':'N','RndOvrSmplr':'N','OvrSmpSMOT':'N','OvrSmpSMOTnc':'N'}
    
    us_submit_label = ['UndSmpDft','RndUndSmplr','UpSmpSMOTek']
    us_label        = ['Undersampling', 'RandomUnderSampler', 'SMOTETomek']
    us_var_sel      = []
    us_shade        = []
    
    # Labels
    for x,y in zip(range(150,340,45),us_label):  
        us_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        us_shade.append(us_labels)
        us_labels.place(x = 300,y = x) 
        us_align = Label(root, text=':',bg ="#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        us_shade.append(us_align)
        us_align.place(x = 535,y = x)
        
    str_lbl = Label(root, text='*',bg = "#048", fg="#f80", font=('Poppins',20,'bold'), bd=15)
    str_lbl.place(x = 445,y = 230) 
    str_lbl_sub = Label(root, text='Numeric Features Only',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    str_lbl_sub.place(x = 345,y = 262)  
        
    # Checkbox selection
    def us_selections():
        res  = us_var.get()
        all_sam_submit['UndSmpDft'] = res
        
    def rus_selections():
        res  = rus_var.get()
        all_sam_submit['RndUndSmplr'] = res
        
    def smt_tomek_selections():
        le_cat_col_lst = list(set(cat_col_lst) - set(drop_feat_selections) - set(oh_selection)-set(le_selection))
        
        if (len(le_cat_col_lst)) == 0 :
            res  = smt_tomek_var.get()
            all_sam_submit['UpSmpSMOTek'] = res
        else: 
            messagebox.showinfo("Error", "Data has Categorical attribues")
            all_sam_submit['UpSmpSMOTek'] = 'N'
        
    us_var = StringVar()
    us_var.set('N')
    us_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=us_var,command =us_selections)
    us_cb.place(x = 545,y = 140)
    
    rus_var = StringVar()
    rus_var.set('N')
    rus_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=rus_var,command =rus_selections)
    rus_cb.place(x = 545,y = 185) 

    smt_tomek_var = StringVar()
    smt_tomek_var.set('N')
    smt_tomek_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=smt_tomek_var,command =smt_tomek_selections)
    smt_tomek_cb.place(x = 545,y = 230) #x = 545,y = 240
        
    # Treshold entries  
    us_thresh_submit  = {'RndUndSmplr_thresh':0.5,'UpSmpSMOTek_thresh':0.5}
    us_thresh_label   = list(us_thresh_submit.keys())
    us_thresh_var_sel = []

    def us_thresh_selections():
        for i,j in zip(us_thresh_var_sel,us_thresh_label):
            res  = i.get()
            us_thresh_submit[j] = res

    RndUndSmplr_thresh_var = StringVar() 
    RndUndSmplr_thresh_var.set('0.5') 
    us_thresh_var_sel.append(RndUndSmplr_thresh_var) 
    RndUndSmplr_thresh_entry = Entry(root,highlightthickness=1,bg ='#048',fg="White",font=('Poppins',11,'bold'),highlightbackground="White", highlightcolor="red",textvariable=RndUndSmplr_thresh_var )  
    RndUndSmplr_thresh_entry.place(x = 575,y = 204,width=30) 
    
    UpSmpSMOTek_thresh_var = StringVar() 
    UpSmpSMOTek_thresh_var.set('0.5') 
    us_thresh_var_sel.append(UpSmpSMOTek_thresh_var) 
    UpSmpSMOTek_thresh_entry = Entry(root,highlightthickness=1,bg ='#048',fg="White",font=('Poppins',11,'bold'),highlightbackground="White", highlightcolor="red",textvariable=UpSmpSMOTek_thresh_var )  
    UpSmpSMOTek_thresh_entry.place(x = 575,y = 245,width=30) 

    ######################### 
    # Over Sampling  Section  
    ######################### 
    
    over_sam_heading = Label(root, text='OVER  SAMPLING',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15,anchor="e")  
    over_sam_heading.place(x = 360,y = 275)

    os_sub_label   = ['OvrSmpDft','RndOvrSmplr','OvrSmpSMOT','OvrSmpSMOTnc']
    os_label       = ['Oversampling', 'RandomOverSampler', 'SMOTE', 'SMOTENC']
    os_var_sel     = []
    os_shade       = []
    
    # Labels
    for x,y in zip(range(326,500,34),os_label):  
        sampling_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")   
        os_shade.append(sampling_labels)
        sampling_labels.place(x = 300,y = x) 
        sampling_align = Label(root, text=':',bg ="#048", fg="White", font=('Poppins',16,'bold'),anchor="e")  
        os_shade.append(sampling_align)  ##SR
        sampling_align.place(x = 535,y = x)

    str_lbl2 = Label(root, text='*',bg = "#048", fg="#f80", font=('Poppins',20,'bold'))
    str_lbl2.place(x = 400,y = 396)  #x = 372,y = 365
    str_lbl2_sub = Label(root, text='Numeric Features Only',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    str_lbl2_sub.place(x = 300,y = 416)  
    
    
    # Checkbox selection
    def os_selections():
        res  = os_var.get()
        all_sam_submit['OvrSmpDft'] = res
    def ros_selections():
        res  = ros_var.get()
        all_sam_submit['RndOvrSmplr'] = res
    def smote_selections(): 
        le_cat_col_lst = list(set(cat_col_lst) - set(drop_feat_selections)-set(oh_selection)-set(le_selection))
        
        if (len(le_cat_col_lst)) == 0 :
            res  = smote_var.get()
            all_sam_submit['OvrSmpSMOT'] = res
        else:
            messagebox.showinfo("Error", "Data has Categorical attribues")
            all_sam_submit['OvrSmpSMOT'] = 'N'
            
    def smtnc_selections():
        res  = smtnc_var.get()
        all_sam_submit['OvrSmpSMOTnc'] = res

    os_var = StringVar()
    os_var.set('N')
    os_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=os_var,command =os_selections)
    os_cb.place(x = 545,y = 315)

    ros_var = StringVar()
    ros_var.set('N')
    ros_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=ros_var,command =ros_selections)
    ros_cb.place(x = 545,y = 349)  
    
    smote_var = StringVar()
    smote_var.set('N')
    smote_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=smote_var,command =smote_selections)
    smote_cb.place(x = 545,y = 383)  
    
    smtnc_var = StringVar()
    smtnc_var.set('N')
    smtnc_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=smtnc_var,command =smtnc_selections)
    smtnc_cb.place(x = 545,y = 417)  
    
    # Treshold entries
    os_thresh_submit  = {'RndOvrSmplr_thresh':0.5,'OvrSmpSMOT_thresh':'minority'}
    os_thresh_label   = list(os_thresh_submit.keys())
    os_thresh_var_sel = []

    def os_thresh_selections():
        for i,j in zip(os_thresh_var_sel,os_thresh_label):
            res  = i.get()
            os_thresh_submit[j] = res

    RndOvrSmplr_thresh_var = StringVar()  
    RndOvrSmplr_thresh_var.set('0.5')  
    os_thresh_var_sel.append(RndOvrSmplr_thresh_var)  
    RndOvrSmplr_thresh_entry = Entry(root,highlightthickness=1,bg ='#048',fg="White",font=('Poppins',11,'bold'),highlightbackground="White", highlightcolor="red",textvariable=RndOvrSmplr_thresh_var )   
    RndOvrSmplr_thresh_entry.place(x = 575,y = 350,width=30)  
    
    OvrSmpSMOT_thresh_var = StringVar()  
    OvrSmpSMOT_thresh_var.set('0.5')  
    os_thresh_var_sel.append(OvrSmpSMOT_thresh_var)  
    OvrSmpSMOT_thresh_entry = Entry(root,highlightthickness=1,bg ='#048',fg="White",font=('Poppins',11,'bold'),highlightbackground="White", highlightcolor="red",textvariable=OvrSmpSMOT_thresh_var )   
    OvrSmpSMOT_thresh_entry.place(x = 575,y = 383,width=30)  

    sampling_cb = [us_cb,rus_cb,smt_tomek_cb,os_cb,ros_cb,smote_cb,smtnc_cb]
    
    ########################### 
    # Data Split Method Section  
    ########################### 
    
    cv_label = Label(root, text='DATA  SPLIT  METHODS',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15,anchor="e") 
    cv_label.place(x = 360,y = 450)   

    cv_submit       = {'dfultSplit':'N', 'rrSplit':'N','kFld':'N','strtkFld':'N'}
    cv_submit_label = list(cv_submit.keys())
    cv_label        = ['Traditional','Repeated Random','K Fold','Stratified K-fold']
    cv_var_sel      = []
    
    # Labels
    for x,y in zip(range(485,600,32),cv_label): 
        
        K_cv_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        K_cv_labels.place(x = 300,y = x) 
        
        K_cv_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        K_cv_align.place(x = 535,y = x)
        
    # Checkbox selection
    def cv_selections():
        for i,j in zip(cv_var_sel,cv_submit): 
            res  = i.get()
            cv_submit[j] = res

    for x,y in zip(range(490,600,33),cv_label):   
        y = StringVar()
        y.set('N')
        cv_var_sel.append(y)
        
        cv_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048', variable=y,command =cv_selections)
        cv_cb.place(x = 545,y = x)  
              
    def kFld_selections():
        if 'Y' not in list(all_sam_submit.values()):
            res = kFld_var.get()
            cv_submit['kFld'] = res
        else:
            messagebox.showinfo("Warning","Sampling Data Selected")
            cv_submit['kFld'] = 'N'

    kFld_var = StringVar()
    kFld_var.set('N')
    kFld_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=kFld_var,command =kFld_selections).place(x = 545,y = 539) 
    
    def skFld_selections():
        if 'Y' not in list(all_sam_submit.values()):
            res = skFld_var.get()
            cv_submit['strtkFld'] = res
        else:
            messagebox.showinfo('Warning',"Sampling Data Selected")
            cv_submit['strtkFld'] = 'N'

    skFld_var = StringVar()
    skFld_var.set('N')
    skFld_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=skFld_var,command =skFld_selections).place(x = 545,y = 571) 
        
    # Treshold entries    
    cv_thresh_submit  = {'rrSplit_thresh':5,'kFld_thresh':5,'strtKFld_thresh':5}
    cv_thresh_label   = list(cv_thresh_submit.keys())
    cv_thresh_val     = list(cv_thresh_submit.values())
    cv_thresh_var_sel = []

    def cv_thresh_selections():
        for i,j in zip(cv_thresh_var_sel,cv_thresh_label):
            res  = i.get()
            cv_thresh_submit[j] = res

    for x,y,z in zip(range(523,587,31),cv_thresh_label,cv_thresh_val): 
        
        y = StringVar()
        y.set(z)
        cv_thresh_var_sel.append(y)
        cv_thresh_entry = Entry(root,highlightthickness=1,bg ='#048',fg="White",font=('Poppins',11,'bold'),highlightbackground="#048", highlightcolor="red",textvariable=y) 
        cv_thresh_entry.place(x = 575,y = x,width=30) 
  
    ############################## 
    # Feature Enggineering Section  
    ##############################     
    
    feat_heading = Label(root, text='FEATURE  BUILD',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15,anchor="e")  
    feat_heading.place(x = 720,y = 110)

    feat_submit       = {'pearson':'N','anova':'N','extTree':'N','kbest':'N','igain':'N','CorrCoef':'N','RmvHiCorr':'N'} 
    feat_submit_label = list(feat_submit.keys())
    feat_label        = ['PccChiSq', 'ANOVA f-test','ExtraTree', 'KBest','Info Gain','CorrCoef', 'RmvHiCorr'] 
    feat_var_sel      = []
    feat_shade        = []
    
    # Labels
    for x,y in zip(range(150,400,38),feat_label): 
        fea_engg_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        feat_shade.append(fea_engg_labels)
        fea_engg_labels.place(x = 700,y = x)
        fea_engg_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        feat_shade.append(fea_engg_align)
        fea_engg_align.place(x = 883,y = x)  
        
    # Checkbox selection
    def feat_selections():
        for i,j in zip(feat_var_sel,feat_submit_label):
            res  = i.get()
            feat_submit[j] = res

    for x,y in zip(range(155,400,38),feat_label): #150,400,38
        y = StringVar()
        y.set('N')
        feat_var_sel.append(y)
        feat_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048', 
                        variable=y,command =feat_selections)
        feat_cb.place(x = 890,y = x)  ###897   
        
    # Treshold entries 
    feat_thresh_submit = {'corr_trgt_thresh':0.05,'corr_rmvhi_thresh':0.8}  
    
    def feat_thresh_selections():
        res  = feat_thresh_var.get()
        feat_thresh_submit['feaSelCount'] = ''
    
    def corr_thresh_selections():
        res  = corr_thresh_var.get()
        feat_thresh_submit['corr_trgt_thresh'] = res

    corr_thresh_var = StringVar()
    corr_thresh_var.set(0.05)
    corr_thresh_entry = Entry(root,highlightthickness=1,bg ='#048',fg="White",font=('Poppins',11,'bold'),highlightbackground="#048", highlightcolor="red",textvariable=corr_thresh_var) 
    corr_thresh_entry.place(x = 920,y = 345,width=40)    
    
    def corr_rmvhi_thresh_selections():
        res  = corr_rmvhi_thresh_var.get()
        feat_thresh_submit['corr_rmvhi_thresh'] = res

    corr_rmvhi_thresh_var = StringVar()
    corr_rmvhi_thresh_var.set(0.8)
    corr_rmvhi_thresh_entry = Entry(root,highlightthickness=1,bg ='#048',fg="White",font=('Poppins',11,'bold'),highlightbackground="#048", highlightcolor="red",textvariable=corr_rmvhi_thresh_var) 
    corr_rmvhi_thresh_entry.place(x = 920,y = 385,width=30)    
        
    def kbest_selections(): 
        
        global num_scalar_selections
    
        if (num_scalar_selections != []):
            if ( 'Normalization' in num_scalar_selections) :
                res  = kbest_var.get()
                feat_submit['kbest'] = res
            else:
                messagebox.showinfo("Warning", "Selected numeric scaling will lead to negative values. De-select this option")
                feat_submit['kbest'] = 'N'
          
        if (num_scalar_selections == []):
            res  = kbest_var.get()
            feat_submit['kbest'] = res

    kbest_var = StringVar()
    kbest_var.set('N')
    kbest_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=kbest_var,command =kbest_selections)
    kbest_cb.place(x = 890,y = 268)   
    
    def extree_selections():
        res  = extree_var.get()
        feat_submit['extTree'] = res
        
    extree_var = StringVar()
    extree_var.set('N')
    extree_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=extree_var,command =extree_selections)
    extree_cb.place(x = 890,y = 226)
    
    def toplevel(title,header_image_=None,bg_='#004488',width_=500,height_=300, header_height=60,header_x_cord=100):
        
        tl_window      = Toplevel(bg=bg_,width=width_,height=height_)
        window_width,window_height, x_cordinate, y_cordinate = display_center(tl_window,height_,width_)
        tl_window.geometry("{}x{}+{}+{}".format( window_width,window_height, x_cordinate, y_cordinate)) ##SR
        tl_window.title(title)
        tl_window.grid()
        header = Frame(tl_window, width=width_, height=header_height, bg="white")
        header.grid(columnspan=3, rowspan=2, row=0)
        header_heading = Label(tl_window,text = title,bg ='#FFFFFF',fg= '#004488', font = ('Poppins',20,"bold"))   
        header_heading.place(x = header_x_cord,y = 15)
        header_image = Label(tl_window,image = header_image_)
        header_image.place(x=0,y=1)
        main_content = Frame(tl_window,highlightbackground="#f80", bg="#048", highlightcolor="#f80", highlightthickness=2,
                              width=width_, height=height_-header_height).grid()
        return tl_window
    
    def display_center(master,window_height_,window_width_):
        master.resizable(False, False) 
        window_height = window_height_
        window_width  = window_width_
        screen_width  = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_cordinate   = int((screen_width/2) - (window_width/2))
        y_cordinate   = int((screen_height/2) - (window_height/2))
        return (window_width,window_height, x_cordinate, y_cordinate)

        
    ########################################
    # Routine to select correlation checkbox
    ########################################
     
    def get_correlation(data,targetName):
        
        from sklearn import preprocessing
        from pylab import savefig
        
        scaler              = preprocessing.MinMaxScaler()
        cat_var             = [c for c in data.columns if data[c].dtype=='O']
        num_var             = [n for n in data.columns if n not in cat_var]
        num_var             = data[num_var]       
        num_var             = pd.DataFrame(scaler.fit_transform(num_var), columns =num_var.columns)
        num_var[targetName] = data[targetName]
        corr                = num_var.corr()
        
        #### Heatmap
        fig = plt.figure(figsize=(15,10))
        hm  = sns.heatmap(corr,cmap='coolwarm',linecolor='w',square=True,linewidths=1,annot=True, annot_kws={"size":12})
        plt.title('Heatmap', fontsize = 20)  
        plt.xticks(fontstyle='normal',fontsize=10)
        plt.yticks(fontsize=10)
        fig.savefig('hm.png')
        plt.close(fig)
        
        corr[targetName]    = corr[targetName].apply(lambda x: round(x,3))
        corr                = pd.DataFrame(corr[targetName])
        corr                = corr.rename({'Loan_Status':'Score'},axis = 'columns')
        corr                = corr.reset_index().rename({'index':'Feature'},axis = 'columns')
        return (corr)
    
    ##################################
    # Routine for Correlation List box
    ##################################
      
    def corr_lstBox():  
        
        from tabulate import tabulate
        
        global tl  
        
        corr  = get_correlation(train_df,target_col)
        if len(corr)>0:
            tl   = toplevel('Feature Correlation',Psx_img,)  
            txt  = Text(tl,bg='#004488',fg='white',highlightbackground="#f80", highlightcolor="#f80", highlightthickness=3,height=10,width=42)
            txt.place(x=80,y=80)

            class PrintToTXT(object): 
                def write(self, s):
                    txt.insert(END, s)
                    
            stdout_obj = sys.stdout  
            sys.stdout = PrintToTXT() 
            print(tabulate(corr, headers='keys', tablefmt='simple', showindex=False))
            sys.stdout = stdout_obj  
            exit_button = Button(tl, text='EXIT',command=exit_btn, bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14), relief=RAISED,bd=3) 
            exit_button.place(x = 230,y = 260,height=30)    
        else:
            pass
    
    
    ##############
    # All Features
    ##############     
    
    all_feat_label = Label(root, text='ALL',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
    all_feat_label.place(x = 700,y = 418) ###
    all_feat_align = Label(root, text=':',bg ="#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
    all_feat_align.place(x = 883,y = 418)   

    # Checkbox selectio
    all_feat_submit = feat_submit
    all_feat_labels = list(all_feat_submit.keys())
    
    def all_feat_selections():
        
        for i in all_feat_labels:
            res  = all_feat_var.get()
            all_feat_submit[i] = res
                 
        if (num_scalar_selections != []):
            if ( 'Normalization' in num_scalar_selections) :
                res                      = all_feat_var.get()
                all_feat_submit['kbest'] = res
            else:
                
                if problem == 'Classification':
                    messagebox.showinfo("Warning", "Selected numeric scaling will lead to negative values. De-select this option")
                    all_feat_submit['kbest'] = 'N'
        
        if (num_scalar_selections == []):
            res                      = all_feat_var.get()
            all_feat_submit['kbest'] = res
            
        if problem == 'Regression':
            feat_submit['extTree'] ='N'    
            feat_submit['kbest']   ='N'       

        
    all_feat_var = StringVar()
    all_feat_var.set('N')

    all_feat_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=all_feat_var,command =all_feat_selections)
    all_feat_cb.place(x = 890,y = 420)   
    
    all_feat_label = Label(root, text='Filtering Methods',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    all_feat_label.place(x = 700,y = 442)  
    
    
    ################ 
    # Feature Select  
    ################     
        
    feat_heading = Label(root, text='FEATURE  SELECT',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15,anchor="e")  
    feat_heading.place(x = 720,y = 458)

    feat_submit_new       = {'ComnFeat':'N', 'DefaultFeat':'N','ComnCorFeat':'N'} 
    feat_submit_new_label = list(feat_submit_new.keys())
    feat_label_new        = ['Common','Default','Correlation']
    feat_var_sel_new      = []
    
    # Labels
    for x,y in zip(range(498,700,38),feat_label_new): 
        fea_engg_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")  
        fea_engg_labels.place(x = 700,y = x)
        fea_engg_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")   
        fea_engg_align.place(x = 883,y = x)

    # Checkbox selection
    def feat_selections_new():
        for i,j in zip(feat_var_sel_new,feat_submit_new_label):
            res  = i.get()
            feat_submit_new[j] = res

    for x,y in zip(range(488,700,37),feat_label_new):  
        y = StringVar()
        y.set('N')
        feat_var_sel_new.append(y)
        feat_new_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 3,bg ='#048',variable=y,command =feat_selections_new)
        feat_new_cb.place(x = 890,y = x)  
    
    ################ 
    # Tree Based  
    ################ 
    
    model_sel_heading = Label(root, text='MODEL SELECTION',bg = "#048", fg="#f80", font=('inherit',16,'bold','underline'), bd=15)  
    model_sel_heading.place(x = 1050,y = 65) 
    
    TBM_heading = Label(root, text='TREE  BASED  MODELS',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15)  
    TBM_heading.place(x = 1060,y = 109)
    
    TBM_sub_heading = Label(root, text='* Scaling Not Recommended',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    TBM_sub_heading.place(x = 1137,y = 140)   
        
    tb_model = {'DT':'N','RF':'N','GB':'N','XGB':'N','ADA':'N'} 
    tb_label = ['DT','RF','GB','XGBoost','AdaBoost']
     
    # Labels
    for x,y in zip(range(152,300,31),tb_label):  
        tb_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")   
        tb_labels.place(x = 1005,y = x) 
        tb_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        tb_align.place(x = 1170,y = x)

    # Checkbox selection
    def tb_selections_DT():
        #messagebox.showinfo("Please Verify", "Scaling Not Recommended")
        res  = dt_var.get()
        tb_model['DT'] = res 

    dt_var = StringVar()
    dt_var.set('N')
    dt_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=dt_var,command =tb_selections_DT)
    dt_cb.place(x = 1180,y = 155)  

    def tb_selections_RF():
        #messagebox.showinfo("Please Verify", "Scaling Not Recommended")
        res  = RF_var.get()
        tb_model['RF'] = res  ###

    RF_var = StringVar()
    RF_var.set('N')
    RF_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=RF_var,command =tb_selections_RF)
    RF_cb.place(x = 1180,y = 186)

    def tb_selections_GB():
        #messagebox.showinfo("Please Verify", "Scaling Not Recommended")
        res  = GB_var.get()
        tb_model['GB'] = res  ###

    GB_var = StringVar()
    GB_var.set('N')
    GB_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=GB_var,command =tb_selections_GB)
    GB_cb.place(x = 1180,y = 217)

    def tb_selections_xgb():
        #messagebox.showinfo("Please Verify", "Scaling Not Recommended")
        res  = xgb_var.get()
        tb_model['XGB'] = res  ###

    xgb_var = StringVar()
    xgb_var.set('N')
    xgb_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=xgb_var,command =tb_selections_xgb)
    xgb_cb.place(x = 1180,y = 248)

    def tb_selections_ada():
        #messagebox.showinfo("Please Verify", "Scaling Not Recommended")
        res  = ada_var.get()
        tb_model['ADA'] = res   

    ada_var = StringVar()
    ada_var.set('N')
    ada_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=ada_var,command =tb_selections_ada)
    ada_cb.place(x = 1180,y = 279)

    ################ 
    # Non Tree Based  
    ################ 
    
    N_TBM_heading = Label(root, text='NON  TREE  BASED  MODELS',bg = "#048", fg="orange", font=('Poppins',12,'bold'), bd=15)  
    N_TBM_heading.place(x = 1030,y = 300)  
    
    N_TBM_sub_heading = Label(root, text='* Scaling and Outlier Treatment Recommended',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    N_TBM_sub_heading.place(x = 1075,y = 331)   
    
    ntb_model = {'LN':'N','LR':'N','NB':'N','SVM':'N','NN':'N'} 
    ntb_label = ['Linear','Logistic','Naivye Bayes','SVM','NN']  
    ntb_shade = []
    
    # Labels
    for x,y in zip(range(344,550,34),ntb_label):  
        ntb_labels = Label(root, text=y,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")    
        ntb_shade.append(ntb_labels)
        ntb_labels.place(x = 1005,y = x) 
        ntb_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")   
        ntb_shade.append(ntb_align)
        ntb_align.place(x = 1170,y = x)

    # Checkbox selection
    def ntb_selections_Nn():
            #messagebox.showinfo("Please Verify", "Normalization Recommended")
            res  = NN_var.get()
            ntb_model['NN'] = res
            
    def ntb_selections_Nb():
            #messagebox.showinfo("Please Verify", "Scaling Recommended")
            res  = Nb_var.get()
            ntb_model['NB'] = res   
        
    def ntb_selections_lin():
            #messagebox.showinfo("Please Verify", "Scaling and Outlier Treatment Recommended")
            res  = lin_var.get()
            ntb_model['LN'] = res   
    def ntb_selections_log():
            #messagebox.showinfo("Please Verify", "Scaling and Outlier Treatment Recommended")
            res  = log_var.get()
            ntb_model['LR'] = res   
    def ntb_selections_svm():
            #messagebox.showinfo("Please Verify", "Scaling and Outlier Treatment Recommended")
            res  = svm_var.get()
            ntb_model['SVM'] = res   
    
    lin_var = StringVar()
    lin_var.set('N')
    ntb_cb_lin = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=lin_var,command =ntb_selections_lin)
    ntb_cb_lin.place(x = 1180,y = 347)

    log_var = StringVar()
    log_var.set('N')
    log_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=log_var,command =ntb_selections_log)
    log_cb.place(x = 1180,y = 381)
    
    Nb_var = StringVar()
    Nb_var.set('N')
    nb_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=Nb_var,command =ntb_selections_Nb)
    nb_cb.place(x = 1180,y = 415)
  
    svm_var = StringVar()
    svm_var.set('N')
    svm_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=svm_var,command =ntb_selections_svm)
    svm_cb.place(x = 1180,y = 449)

    NN_var = StringVar()
    NN_var.set('N')
    nn_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=NN_var,command =ntb_selections_Nn)
    nn_cb.place(x = 1180,y = 483)
   
    all_modl_label = Label(root, text='ALL',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")    
    all_modl_label.place(x = 1005,y = 520)
    all_modl_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")      
    all_modl_align.place(x = 1170,y = 520)   
    
    # Checkbox selection
    all_modl_submit = {**tb_model, **ntb_model}
    all_modl_labels = list(all_modl_submit.keys())
    def all_modl_selections():
        for i in all_modl_labels:
            res                = all_modl_var.get()
            all_modl_submit[i] = res
            
            if problem == 'Regression':
                all_modl_submit['LR'] = 'N'
            else:
                all_modl_submit['LN'] = 'N'

    all_modl_var = StringVar()
    all_modl_var.set('N')

    all_modl_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=all_modl_var,command =all_modl_selections)
    all_modl_cb.place(x = 1180,y = 523)
    
    
    tuning_label = Label(root, text='Hyper Tuning',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")    
    tuning_label.place(x = 1005,y = 560)
    tuning_align = Label(root, text=':',bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e")      
    tuning_align.place(x = 1170,y = 560)
    
    # Hyper Parameter Selection
    tuning_submit = {'parmTuning':'N'}
    tuning_labels = list(tuning_submit.keys())
    
    def tuning_selections():
        for i in tuning_labels:
            res  = tuning_var.get()
            tuning_submit[i] = res
            
    tuning_var = StringVar()
    tuning_var.set('N')
    tuning_cb = Checkbutton(root,onvalue='Y',offvalue ='N',height = 1,bg ='#048',variable=tuning_var,command =tuning_selections)
    tuning_cb.place(x = 1180,y = 565)
    
    global final_res
    final_res = {}
    
    def Clear(): 
        try:
            os.remove('hm.png')
        except OSError:
            pass
        Auto_ml()

    def submit():   
        
        validate()
        if (errFlg == 'N'):
            result()
            root.destroy()
        
    def Exit():  
        try:
            os.remove('hm.png')
        except OSError:
            pass
        root.destroy()
        
    def validate(): 
        from sklearn.model_selection import train_test_split
        
        global errFlg
        
        errFlg    = 'N'
        sels      = result()
        dependCol = train_df.columns[-1]
        X         = train_df.drop(dependCol, axis=1)
        y         = train_df[dependCol]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        if (sels['UndSmpDft'] == 'Y'):
             X_train,y_train = downUpSample(X_train,y_train,'d',dependCol)
        if (sels['OvrSmpDft'] == 'Y'):
             X_train,y_train = downUpSample(X_train,y_train,'u',dependCol)
        
    def result():
        global final_res
        global  binDict
        
        us_thresh_selections()
        os_thresh_selections()
        cv_thresh_selections() 
        corr_thresh_selections()
        corr_rmvhi_thresh_selections()
        pp_submit['ChgNumChr'] = num_cat_selections
        pp_submit['RmvCols']   = drop_feat_selections
        #print(drop_feat_selections) 
        
        if 'Y' in list(all_modl_submit.values()):
            res_li = [pp_submit,all_sam_submit,us_thresh_submit,os_thresh_submit,feat_submit,feat_submit_new,
                feat_thresh_submit,cv_submit,cv_thresh_submit,all_modl_submit,tuning_submit]  
        else:
            res_li = [pp_submit,all_sam_submit,us_thresh_submit,os_thresh_submit,feat_submit,feat_submit_new,
                        feat_thresh_submit,cv_submit,cv_thresh_submit,tb_model,ntb_model,tuning_submit]
            
        for d in res_li:
            for k, v in d.items():
                final_res[k] = v

        if imb_check(train_df,target_col) == 'Y':
            final_res['ImbFlg'] ='Y'
        else:
            final_res['ImbFlg'] ='N'
                
        ## Feature Flag        
        if 'Y' in list(feat_submit.values()) :
            final_res['feaSelFlg'] ='Y'
        else:
            final_res['feaSelFlg'] ='N'
                                
        ## Bining
        try:
            if binDict:
                final_res['Binning'] = binDict
            else:
                final_res['Binning'] = ''
        except Exception as e:
            final_res['Binning'] = ''
            pass
        
        ## Numerical Scalar Flag        
        if num_scalar_selections != []:
            final_res['NumSclFlg'] ='Y'
            final_res['NumScl'] = num_scalar_selections[0]
        else:
            final_res['NumSclFlg'] ='N'
        
        ## Categorical Scalar Flag        
        if (oh_selection != []) or (le_selection != []) :
            final_res['CatSclFlg'] ='Y'
            final_res['OneHotEnc'] = oh_selection
            final_res['LblEnc']     = le_selection
        else:
            final_res['CatSclFlg'] ='N'
        
        return final_res

    ######################################################   
    # Grayout the labels and options in case of Regression    
    ######################################################  

    def destroy_label(li):   
        for i in li:  
            i.destroy()  

    def disable(li):  
        for x in li:  
            x.config(state=DISABLED)  

    def grayout(li):  
        for x in li:  
            x.config(fg='#808080')  
            
    fea_model_grayout = [x for x in feat_shade[3:8]]+[x for x in ntb_shade[2:4]]+us_shade+os_shade  
    sampling_grayout  = [kbest_cb,extree_cb,log_cb]+sampling_cb 
    thresh_grayout    = [RndUndSmplr_thresh_entry,UpSmpSMOTek_thresh_entry,RndOvrSmplr_thresh_entry,OvrSmpSMOT_thresh_entry]        

    if problem == 'Regression':  
        destroy_label(thresh_grayout)  
        disable(sampling_grayout)
        grayout(fea_model_grayout)
    else:  
        disable([ntb_cb_lin])  
        grayout(ntb_shade[:2])  

        
    ################
    # Footer Section  
    ################
    
    #####################################
    # Show the Class Distribution Diagram
    #####################################
    
    if problem == 'Classification':
        header_imbal = Label(root,image = Psx_imbal,bg = "#048", fg="White", font=('Poppins',16,'bold'),anchor="e") 
        header_imbal.place(x=3,y=650)
        header_imbal_label = Label(root, text='Class Balance',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
        header_imbal_label.place(x = 1,y = 675)  
    
    ########################################
    # Routine to select correlation checkbox    
    ########################################
    
    def corr_chkBox():
        
        res  = corr_var.get()
        corr_lstBox()

    def heatmap_chkBox():
        
        import cv2
                
        res  = hm_var.get()
        image_cv2= cv2.imread('hm.png')
        cv2.imshow("Heat Map", image_cv2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    corr_var = StringVar()
    corr_var.set('1')
    corr_rb = Radiobutton(root,height = 3,bg ='#048',variable=corr_var,command =corr_chkBox)
    corr_rb.place(x = 100,y = 660,height=15)  
    corr_var_label = Label(root, text='Correlation',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    corr_var_label.place(x = 87,y = 675) 

    hm_var = StringVar()
    hm_var.set('0')
    hm_rb = Radiobutton(root,height = 3,bg ='#048',variable=hm_var,command =heatmap_chkBox)
    hm_rb.place(x = 370,y = 660,height=15)  
    
    hm_var_label = Label(root, text='Heatmap',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    hm_var_label.place(x = 360,y = 675) 

    
    #############################
    # Missing Values Notification     
    #############################
    
    misVal_var_label = Label(root, text='Missing Values Handled',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    misVal_var_label.place(x = 162,y = 675) 
    
    if impute_missing == 'Y':
        misVal_text = Label(root, text=impute_missing,bg = "#048", fg="White", font=('Poppins',7,"bold"),anchor="e") 
    if impute_missing == 'N':
        misVal_text = Label(root, text='NA',bg = "#048", fg="White", font=('Poppins',7,"bold"),anchor="e") 
    misVal_text.place(x = 210,y = 660)
    
    ################################
    # Outliers Presence Notification     
    ################################
    Outlyr_var_label = Label(root, text='Outliers',bg = "#048", fg="White", font=('Poppins',7),anchor="e") 
    Outlyr_var_label.place(x = 295,y = 675)  
    Outlyr_text = Label(root, text=Outlyrflag,bg = "#048", fg="White", font=('Poppins',7,"bold"),anchor="e") 
    Outlyr_text.place(x = 308,y = 660)
       
    ################  
    # Button Section
    ################  

    Clear_Button = Button(root, text="Clear",command=Clear,bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14),
                          bd=6,relief=RAISED )

    Clear_Button.place(x = 520, y =650,height=40)  


    Submit_Button = Button(root, text="Submit",command=submit,bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14),
                         relief=RAISED,bd=6)
    Submit_Button.place(x = 605, y =650, height=40)  

    Exit_Button = Button(root, text="EXIT",command=Exit,bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14),
                         relief=RAISED,bd=6)  
    Exit_Button.place(x = 700, y =650, height=40)   

    problem_label = Label(root, text='Problem Type : '+ problem,bg = "#048", fg="White",font = ('Poppins',16,"bold"))
    problem_label.place(x = 800,y = 650)  
    
    ## Hover Effects
    def but_1_onbutton(e):
        Submit_Button.config( bg ='#f80',fg= 'white')
    def but_1_leavebutton(e):
        Submit_Button.config( bg ='#FFFFFF',fg= '#004488')
    def but_2_onbutton(e):
        Exit_Button.config( bg ='#f80',fg= 'white')
    def but_2_leavebutton(e):
        Exit_Button.config( bg ='#FFFFFF',fg= '#004488')
    def but_3_onbutton(e):
        Clear_Button.config( bg ='#f80',fg= 'white')
    def but_3_leavebutton(e):
        Clear_Button.config( bg ='#FFFFFF',fg= '#004488')

    Submit_Button.bind('<Enter>',but_1_onbutton)
    Submit_Button.bind('<Leave>',but_1_leavebutton)
    Exit_Button.bind('<Enter>',but_2_onbutton)
    Exit_Button.bind('<Leave>',but_2_leavebutton)
    Clear_Button.bind('<Enter>',but_3_onbutton)
    Clear_Button.bind('<Leave>',but_3_leavebutton)

    root.mainloop()
    return result()

############################
#  END OF GUI SCREEN DESIGN    
############################

#############################################################
# Routine to seperate numerical and cetegorical feature names
#############################################################

def features(data):
    
    cat_cols = [c for c in data.columns if data[c].dtype=='O']
    num_cols = [n for n in data.columns if n not in cat_cols]
    
    return [cat_cols,num_cols]

##################################### 
# Routine to check for missing values
##################################### 

def missingValues(data):
       
    col_null      = data.isnull().sum()
    col_null_frac = col_null / data.shape[0]
    missing       = 'N'

    # Get the missing column count
    temp = pd.DataFrame(col_null)
    if (temp[0].sum()) > 0:
        missing = 'Y'
    
    return(missing)


#############################################################
# Routine to check for zero values
#############################################################

def zeroValues(data):
    
    col_zero      = (data.iloc[:] == 0).sum()
    col_zero_frac = col_zero / data.shape[0]
    fig = plt.figure(figsize= (15,4))
    fig.suptitle('ZERO VALUES')
    col_zero_frac.plot(kind='bar')
    
######################################## 
# Routine to plot the class distribution
######################################## 

def pltClsDist(data,target):
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    data[target].value_counts().plot(kind='pie', subplots=True, autopct='%1.2f%%', explode= (0.05, 0.05), startangle=80, legend=True,fontsize=12, figsize=(14,6), textprops={'color':"black"},ax=ax1)

    plt.legend(["0: paid loan","1: not paid loan"])
    ax=sns.countplot(y=target,data=data, palette='hls',order = data[target].value_counts().index,ax=ax2)
    plt.title('Loan Status Distribution')
    total = len(data[target])
     
    for p in ax.patches:
        percentage = '{:.3f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

    Class0, Class1 = pd.DataFrame(data[target]).value_counts()
    Cls_imbal      = 'Y' if ((Class1/Class0) <0.65) else 'N'  

    plt.tight_layout()
    plt.show()
    
    
    #fig1 = plt.figure(figsize=(0.75,0.35))
    fig1 = plt.figure(figsize=(15,10))
    
    g = sns.countplot(data[target],palette=['green',"red"])
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.set(xticklabels=[])  
    g.set(yticklabels=[])  
    fig1.savefig('imbal.png')
   
    return(Cls_imbal)
    

################################ 
# Routine to detect the outliers
################################ 

def detectOutliers(num_cols,dataType): 
    
    global Outlyrflag
    
    start       = "\033[1m"
    end         = "\033[0m"
    Outlyrflag  = 'N'
    #f, ax = plt.subplots(figsize=(12, 8))
    #sns.boxplot(data=num_cols,orient='h',palette="Set2")
    #plt.show()
    
    ######################################################
    # Get the Outliers for each feature based on IQR Score
    ######################################################
    
    lst         = []

    for col_name in num_cols.columns:
        Q3, Q1      = np.percentile(num_cols[col_name], [75 ,25])
        IQR         = Q3-Q1
        Uwhisker    = Q3 + (1.5*IQR) 
        Lwhisker    = Q1 - (1.5*IQR)  
        new_row     = {'Feature':col_name, 'Outliers Count':len(num_cols.loc[(num_cols[col_name] < Lwhisker) | (num_cols[col_name] > Uwhisker),col_name])}
        #df_Outlier  = df_Outlier.append(new_row, ignore_index=True)
        lst.append([col_name,len(num_cols.loc[(num_cols[col_name] < Lwhisker) | (num_cols[col_name] > Uwhisker),col_name])])
    
    df_Outlier  = pd.DataFrame(lst, columns=['Feature','Outliers Count'])
    
    #if dataType !='CHK':
    #    print(start +'                                            ' + dataType +' Outliers'+ end)
    #    display(df_Outlier.set_index('Feature').T)
    
    if df_Outlier['Outliers Count'].sum() > 0:
        Outlyrflag = 'Y'
    
    return()

#############################################################
# Routine to handle the outliers
#############################################################

def treatOutliers(num_cols,forData):
    
    for col_name in num_cols.columns:
        
        Coltype     = num_cols[col_name].dtype
        Q3, Q1      = np.percentile(num_cols[col_name], [75 ,25])
        IQR         = Q3-Q1
        Uwhisker    = Q3 + (1.5*IQR) 
        Lwhisker    = Q1 - (1.5*IQR)  
        #print(col_name,Lwhisker,Uwhisker)
        Outliers_Values = [x for x in num_cols[col_name] if x < Lwhisker or x > Uwhisker]

        if len(Outliers_Values) > 0:
            num_cols.loc[(num_cols[col_name] < Lwhisker) | (num_cols[col_name] > Uwhisker),col_name]  = num_cols[col_name].median()
            
            if ( (Coltype == 'int64') | (Coltype == 'int32') ):
                num_cols[col_name]  = num_cols[col_name].apply(int)
            
            #print(f'Outliers cleanup done on the feature {col_name} for dataset {forData}')
    
    return(num_cols)


#################
# End of GUI Code
#################

##########################################################################################
# Routine to perform various feature engineering techniques by means of Filtering Methods
##########################################################################################

def feaEng(catData,numData,dependent,Options,targetName): 
    
    global PearsonPredictors          
    global AnovaPredictors            
    global KBestChi2Predictors        
    global KBestMuInfoClsfPredictors   
    global ExttreePredictors          
    global NonLowCorrPredictors
    global NonHiCorrPredictors
    global IgainPredictors               
       
    CatPredictors                = []   
    PearsonPredictors            = []
    AnovaPredictors              = []
    KBestChi2Predictors          = []
    KBestMuInfoClsfPredictors    = []
    ExttreePredictors            = []
    NonLowCorrPredictors         = []
    NonHiCorrPredictors          = []
    IgainPredictors              = []
    
    numData_ = numData
    if (Options['pearson'] == 'Y'):
        PearsonPredictors = feaPearsonChiSq(numData,catData,dependent,'Pearson',targetName)
        
    if (Options['anova'] == 'Y'):
        AnovaPredictors = feaAnova(numData,catData,dependent,Options)
             
    if (Options['kbest'] == 'Y'):
        KBestChi2Predictors       = feaEngKBest(numData, catData, dependent,(Options), 'chi2')    
        KBestMuInfoClsfPredictors = feaEngKBest(numData, catData, dependent,(Options), 'mutual_info_classif')
    
    if (Options['extTree'] == 'Y'):
        ExttreePredictors = feaExtTree(numData, catData,dependent,(Options))
    
    if (Options['CorrCoef'] == 'Y'):
        NonLowCorrPredictors = feaCorrOLS(numData,catData,dependent,'CorrCoef',Options,targetName)
    
    if (Options['RmvHiCorr'] == 'Y'):
        NonHiCorrPredictors = feaCorrOLS(numData,catData,dependent,'RmvHiCorr',Options,targetName)
    
    if (Options['igain'] == 'Y'):
        numData = numData_
        IgainPredictors = feaiGain(numData, catData,dependent,Options,targetName)
    
    return (PearsonPredictors,AnovaPredictors,KBestChi2Predictors,KBestMuInfoClsfPredictors,ExttreePredictors,NonLowCorrPredictors,NonHiCorrPredictors,IgainPredictors)

##########################################################################################
# Routine to select all top features
##########################################################################################

def selFeatures(features,N):    
    N                     = N.strip()
    features              = pd.DataFrame(features)
    #features             = features.reset_index() 
    features.columns      = ['Features', 'Score']
    features              = features.sort_values(by=['Score'],ascending=False)
    features              = features.reset_index()
    features              = features.drop(['index'], axis=1)

    ###################################### 
    # Select features based on the value N
    ######################################
    if N == '':
        return(features[features.Score > 0]) # features[features['Score'] != 0]
    else:
        N = int(N)
        return(features.head(N))
    
##########################################################################################
# Routine to convert categorical to numerical
##########################################################################################

def convCatNum(catData):
    
    from sklearn.preprocessing import LabelEncoder
    
    le          = LabelEncoder()    
    catDataEnc = catData.apply(lambda col: le.fit_transform(col))  
    
    return catDataEnc

##########################################################################################
# Routine to convert categorical to numerical and append to the numeric attributes
##########################################################################################

def mergeData(catData,numData):

    catDataEnc                  = convCatNum(catData) 
    catDataEnc                  = catDataEnc.reset_index() 
    catDataEnc                  = catDataEnc.drop(['index'], axis=1)
    numData                     = numData.reset_index()
    numData                     = numData.drop(['index'], axis=1)
    catDataEnc[numData.columns] = numData
    data                        = catDataEnc
    
    return data

#######################################################################################
# Routine to select numerial and categorical features based on Pearson & ChiSq p-value
#######################################################################################
def feaPearsonChiSq(numData,catData,target,ftype,targetName):
    
    from scipy.stats import chi2_contingency
    
    start                 = "\033[1m"
    end                   = "\033[0m"
    cols                  = ['Column','p-value']
    #df                    = pd.DataFrame(columns=cols)
    target                = target[targetName]
    SelectedPredictors    = []
    lst                   = []
    
    
    #################################
    # Select best numerical features
    #################################
              
    if (ftype == "Pearson" ):  
        
        for col in numData.columns:
            
            pearson_coef, pval         = scipy.stats.pearsonr(numData[col], target)
            #F_statistic, pval          = scipy.stats.f_oneway(*numData.groupby(target)[col].apply(list))  
            new_row                    = {'Column':col, 'p-value':pval}
            
            ######################################
            # Assumes a null hypothesis where two variables are independent & alternative hypothesis two variables are dependent
            # Select features whose p-value < 0.05
            ######################################
            if (pval <0.05):
                #df = df.append(new_row, ignore_index=True)
                lst.append(new_row)
                SelectedPredictors.append(col)
        df  = pd.DataFrame(lst, columns=['p-value','Column'])
        
        ############################################################################################################
        # Select best categorical features using ChiSq method and append to the numerical selected pearson features
        ############################################################################################################
         
        if (catData.shape[1] != 0):
            catDataEnc = convCatNum(catData)   

            for col in catDataEnc.columns:
                table                      = pd.crosstab(columns=catDataEnc[col],index=target)
                _, pval, _, expected_table = scipy.stats.chi2_contingency(table) 
                new_row                    = {'Column':col, 'p-value':pval}

                ##################################################  
                # Select categorical features whose p-value < 0.05  
                ################################################## 
                if (pval <0.05):
                    #df = df.append(new_row, ignore_index=True)
                    lst.append(new_row)
                    SelectedPredictors.append(col)
                    
        df  = pd.DataFrame(lst)
                
    df = df.T
    df.columns = [''] * len(df.columns)    
    
    return (SelectedPredictors)

######################################################################################################
# Routine to select numerial and categorical features based on correlation coefficient and OLS p-value
######################################################################################################

def feaCorrOLS(numData,catData,target,ftype,Options,targetName):   
    
    import statsmodels.api as sm
    from scipy import stats
    
    start                 = "\033[1m"
    end                   = "\033[0m"
    cols                  = ['Column','Coeff-value']
    SelectedPredictors    = []
    lst                   = []
    
    ############################################################################
    # Select best numerical features based on correlation with dependent feature
    ############################################################################ 
    if (ftype == 'CorrCoef'):          
        X                = numData
        X[targetName]    = target  
        threshold        = float(Options['corr_trgt_thresh'])
        corrWithdepndt   = pd.DataFrame(X.corrwith(X[targetName]),columns=['coeff'])
        corrWithdepndt   = corrWithdepndt.reset_index().rename({'index':'Column'},axis = 'columns') 
        corrWithdepndt   = corrWithdepndt[:-1] # drop last row
        
        #######################################################################################################
        # Select features whose Correlation coefficient > threshold (i.e) drop less correlated ones with target
        #######################################################################################################
        for i in range(0,len(corrWithdepndt['Column'])):
            if (abs(corrWithdepndt['coeff'][i]) > threshold):
                SelectedPredictors.append(corrWithdepndt['Column'][i])
            else:
                new_row = {'Column':corrWithdepndt['Column'][i], 'Coeff-value':abs(corrWithdepndt['coeff'][i])}
                lst.append(new_row)
    
    df  = pd.DataFrame(lst, columns=cols)
    
    ############################################
    # Find highly correlated features and remove 
    ############################################
    if (ftype == 'RmvHiCorr'):   
        corr_features      = set()
        corr_matrix        = numData.corr().abs()
        corr_matrix[corr_matrix == 1] == 0
        threshold          = float(Options['corr_rmvhi_thresh'])
        df                 = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool_)) # selecting the upper traingular 
        to_drop            = [column for column in df.columns if any(df[column] > threshold)]
        numData            = numData.drop(to_drop, axis=1)
        
        if targetName in numData.columns:
            numData = numData.drop([targetName], axis=1)
        df                 = df.max().sort_values(ascending=False) # for each feature, find the max corr and sort
        df                 = pd.DataFrame(df,columns=['hicoeff'])
        df                 = df.reset_index().rename({'index':'Column'},axis = 'columns') 
        df                 = df.where(df['hicoeff'] > threshold).dropna()
        SelectedPredictors = list(numData)
        
        #SelectedPredictors = [item for item in numData.columns if item not in corr_features] # Drop the Highly correlated features
    
    #####################################################################################################################
    # Select categorical features based on statsmodel OLS (p-value) and append to numerical selected correlation features
    #####################################################################################################################
    
    if (catData.shape[1] != 0):
        X          = convCatNum(catData)   
        y          = target
        model      = sm.OLS(y, X)
        result     = model.fit()
        df_cf      = pd.DataFrame(result.pvalues,columns=['p-value'])
        df_cf      = df_cf.reset_index().rename({'index':'Column'},axis = 'columns') 
        ##################################################################
        # Select categorical features whose p-value < 0.05 and drop > 0.05
        ##################################################################
        for i in range(0,len(df_cf['Column'])):
            if ( (abs(df_cf['p-value'][i]) > 0.05) ):

                if (ftype == 'CorrCoef'):
                    new_row = {'Column':df_cf['Column'][i], 'Coeff-value':abs(df_cf['p-value'][i])}
                else:
                    new_row = {'Column':df_cf['Column'][i], 'hicoeff':abs(df_cf['p-value'][i])}

                lst.append(new_row)
                #df      = df.append(new_row, ignore_index=True)    

            else:
                SelectedPredictors.append(df_cf['Column'][i])

    df  = pd.DataFrame(lst)
    df = df.T
    df.columns = [''] * len(df.columns)    
    
    #if (ftype == 'RmvHiCorr'):
    #    print(start +'Removing the following Highly Correlated features'+ end)
    #else:
    #    print(start +'Removing the following Low Correlated features'+ end)
          
    #display(df)
    #print(start +'Remaining features are'+ end) 
    #display(pd.DataFrame(SelectedPredictors,columns=['Feature Name']).T)
    
    return (SelectedPredictors)

#######################################################
# Routine to extract features based on KBest algorithm
#######################################################
    
def feaEngKBest(numData,catData,dependent,Options,func):
        
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import mutual_info_classif
    
    start                 = "\033[1m"
    end                   = "\033[0m"
    SelectedPredictors    = []       
    data = mergeData(catData,numData) if catData.shape[1] != 0  else numData
    #N                     = Options['feaSelCount']
    N = ''
    X                     = data
    y                     = dependent
    #selector              = SelectKBest(score_func=eval(func),k=N) 
    selector              = SelectKBest(score_func=eval(func),k='all') 
    model                 = selector.fit(X,y)

    ################################################################
    # Select best numerical features. Higher the score more relevant
    ################################################################
    
    dfscores              = pd.DataFrame(model.scores_,columns=["Score"])
    dfcolumns             = pd.DataFrame(X.columns)
    features              = pd.concat([dfcolumns,dfscores],axis=1)
    features              = selFeatures(features,N)
    df                    = features.T
    df.columns            = [''] * len(df.columns)   
    SelectedPredictors = list(features['Features'])
    
    #print(start +'KBest features of ' + func + ' are ' + end)
    #display(df)

    return(SelectedPredictors)

#######################################################
# Routine to extract features based on Extra Tree
#######################################################

def feaExtTree(numData,catData,dependent,Options):
    
    from sklearn.ensemble import ExtraTreesClassifier

    start                 = "\033[1m"
    end                   = "\033[0m"
    SelectedPredictors    = []
    data = mergeData(catData,numData) if catData.shape[1] != 0  else numData
    #N                     = Options['feaSelCount']
    N = ''
    X                     = data
    y                     = dependent
    model                 = ExtraTreesClassifier()
    model.fit(X,y)  
    #topN                 = pd.Series(model.feature_importances_,index=X.columns).nlargest(N)
    features              = pd.Series(model.feature_importances_,index=X.columns)
    features              = features.reset_index()
    
    features              = selFeatures(features,N)
    df                    = features.T
    df.columns            = [''] * len(df.columns)    
    #print(start +'ExtraTrees features are ' + end)
    #display(df)
    SelectedPredictors    = list(features['Features'])

    return (SelectedPredictors)
    
#######################################################
# Routine to extract features based on Information Gain
#######################################################

def feaiGain(numData,catData,dependent,Options,targetName): 

    from sklearn.feature_selection import mutual_info_classif

    start                 = "\033[1m"
    end                   = "\033[0m"
    SelectedPredictors    = []
    data = mergeData(catData,numData) if catData.shape[1] != 0  else numData
    if targetName in data.columns:
        data = data.drop([targetName], axis=1)
    #N                     = Options['feaSelCount']
    N = ''
    X                     = data
    y                     = dependent
    model                 = mutual_info_classif(X,y)
    features              = pd.Series(model,index=X.columns)
    features              = features.reset_index()
    features              = selFeatures(features,N)
    df                    = features.T
    df.columns            = [''] * len(df.columns)    
    SelectedPredictors    = list(features['Features'])

    #print(start +'Information Gain features are ' +end)
    #display(df)

    return (SelectedPredictors)

####################################################### 
# Routine to extract high_score_features based on ANOVA
#######################################################

def feaAnova(numData,catData,dependent,Options):

    from sklearn.feature_selection import f_classif
    
    start                 = "\033[1m"
    end                   = "\033[0m"
    SelectedPredictors    = []    
    data = mergeData(catData,numData) if catData.shape[1] != 0  else numData
    #N                     = Options['feaSelCount']
    N = '' 
    X                     = data
    y                     = dependent
    F_score               = pd.DataFrame(f_classif(data,dependent)[0],columns=["Score"])
    dfcolumns             = pd.DataFrame(X.columns)
    features              = pd.concat([dfcolumns,F_score],axis=1)
    features              = selFeatures(features,N)
    df                    = features.T
    df.columns            = [''] * len(df.columns)    
    #print(start +'Anova features are ' + end)
    #display(df)

    SelectedPredictors     = list(features['Features'])

    return (SelectedPredictors)
 
##################################################################################### 
# Consolidate best common numerical features from all the feature engineering methods
#####################################################################################

def getConsolidatedFeatures(PearsonPredictors,AnovaPredictors,KBestChi2Predictors,KBestMuInfoClsfPredictors,ExttreePredictors,NonLowCorrPredictors,NonHiCorrPredictors,IgainPredictors):
        
    if ( (KBestChi2Predictors != []) & (KBestMuInfoClsfPredictors != [])):
        methods     = [PearsonPredictors,AnovaPredictors,KBestChi2Predictors,KBestMuInfoClsfPredictors,ExttreePredictors,NonLowCorrPredictors,NonHiCorrPredictors,IgainPredictors]
        methodnames = ['PearsonPredictors','AnovaPredictors','KBestChi2Predictors','KBestMuInfoClsfPredictors','ExttreePredictors','NonLowCorrPredictors','NonHiCorrPredictors','IgainPredictors']
        names       = ['Pearson','Anova','KBestChi2','KBestMuInfoCls','ExtraTree','NonLowCorrelation','NonHiCorrelation','InformationGain',]
    
    if ( (KBestChi2Predictors == []) & (KBestMuInfoClsfPredictors == [])):
        methods     = [PearsonPredictors,AnovaPredictors,ExttreePredictors,NonLowCorrPredictors,NonHiCorrPredictors,IgainPredictors]
        methodnames = ['PearsonPredictors','AnovaPredictors','ExttreePredictors','NonLowCorrPredictors','NonHiCorrPredictors','IgainPredictors']
        names       = ['Pearson','Anova','ExtraTree','NonLowCorrelation','NonHiCorrelation','InformationGain']
    
    cols        = ['FeaturesCount','Method','Name']  
    final_list  = []
    name        = []
    method      = []
    feaCount    = []
    lst         = []

    for i in range(len(methods)): 
        
        new_row  = {'FeaturesCount':len(methods[i]), 'Method':methods[i],'Name':names[i],'FeaEng':methodnames[i]}
        lst.append(new_row)
            
    df_feng      = pd.DataFrame(lst)
    df_feng      = df_feng.sort_values(by='FeaturesCount',ascending=False, ignore_index=True)
    df_feng      = df_feng[df_feng['FeaturesCount'] != 0]
    
    for i in range(len(df_feng)):
        final_list.append(df_feng['Method'][i])
        feaCount.append(len(final_list[i]))
        name.append(df_feng['Name'][i])
        method.append(df_feng['FeaEng'][i])
        
    consolidated = pd.DataFrame(list(zip(method,name,feaCount,final_list)),columns=['FeaEng','Name','Total Features','Features'])
    #display(consolidated)
    
    return (consolidated)


#######################################################
# Routine to best features for categorical features
#######################################################

def feaEngCate(data,target):
    
    cols                  = ['Column','p-value']
    df_cat                = pd.DataFrame(columns=cols)
    SelectedCatPredictors = []

    for col in data.columns:

        table = pd.crosstab(data[col], target)

        ################
        # display(table)
        ################

        _, pval, _, expected_table = scipy.stats.chi2_contingency(table)
        new_row  = {'Column':col, 'p-value':pval}
        df_cat   = df_cat.append(new_row, ignore_index=True)

        #########################################################
        # Select features whose ChiSq p-value < 0.05 (reject H0)
        #########################################################

        if (pval <0.05):
            SelectedCatPredictors.append(col)

    display(df_cat.T)
    
    return (SelectedCatPredictors)


########################################################################################################################################### 
# Routine to Down - Down sample Majority Class to match Minority class
# Routine to Up   - Up sample Minority Class to match Majority class
########################################################################################################################################### 

def downUpSample(X_train,y_train,sample,targetName):
    
    from sklearn.utils import resample

    global errFlg
       
    target   = targetName
    df       = pd.concat([X_train,y_train], axis=1)
    
    ########################################
    # Separate majority and minority classes
    ########################################
    
    train_major=df[df[target]==0]
    train_minor=df[df[target]==1]
      
    ############################
    # Down-sample Majority Class
    ############################
    
    if (sample == 'd'):
        
        try: 
            x = (df[df[target]==1][target].count()) # get the minority count
        
            Dn_sample_major=resample(train_major,
                                     replace   = False,  # sample with no replacement
                                     n_samples = x,      # to match minority class
                                     random_state=42)    # reproducible results
            df=pd.concat([Dn_sample_major,train_minor])  # Combine miniority class with downsampled majority class
            errFlg  = 'N'
            
        except Exception as e:
            errFlg  = 'Y'
            messagebox.showinfo("Warning", "Insufficient Data for Down Sampling the Majority Class. De-select this option")
            
    ##########################
    # Up-sample Minority Class
    ##########################
    
    if (sample == 'u'):
        
        try: 
            x = (df[df[target]==0][target].count()) # get the majority count
        
            Up_sample_minor=resample(train_minor,
                                     replace   = True,  # sample with no replacement
                                     n_samples = x,     # to match majority class
                                     random_state=42)   # reproducible results
            df=pd.concat([Up_sample_minor,train_major]) # Combine majority class with upsampled minority class
            errFlg  = 'N'
            
        except Exception as e:
            errFlg  = 'Y'
            messagebox.showinfo("Warning", "Insufficient Data for Up Sampling the Miniority Class. De-select this option")

    del train_major,train_minor

    if (errFlg == 'N'):
        
        y  = df[target]
        df = df.drop([target], axis=1)
        X  = df
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    
        return (X,y)
    else:
        return('','')


#########################################################################
# Random Oversampling: Randomly duplicate examples in the minority class.
# Random Undersampling: Randomly delete examples in the majority class.
#########################################################################
    
def RndUndOvrSample(X,y,smplg_strategy,sample):
    
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    
    if (sample == 'd'):
        ruos = RandomUnderSampler(sampling_strategy=smplg_strategy,random_state=42, replacement=True)
        
    if (sample == 'u'):
        ruos = RandomOverSampler(sampling_strategy=smplg_strategy,random_state=42)
    
    X_ruos, y_ruos = ruos.fit_resample(X, y)
    X_ruos.reset_index(drop=True, inplace=True)
    y_ruos.reset_index(drop=True, inplace=True)
    
    return (X_ruos,y_ruos)

###########
#    SMOTE  
###########

def smote(X,y,smplg_strategy):
    
    from imblearn.over_sampling import SMOTE
    
    smote        = SMOTE(sampling_strategy=smplg_strategy)
    X_smT, y_smT = smote.fit_resample(X,y)

    X_smT.reset_index(drop=True, inplace=True)
    y_smT.reset_index(drop=True, inplace=True)
       
    return (X_smT, y_smT)

################
#    SMOTETomek  
################

def smoteTomek(X,y,smplg_strategy):
    
    from imblearn.combine import SMOTETomek

    smk          = SMOTETomek(sampling_strategy=smplg_strategy,random_state=42)
    X_smk, y_smk = smk.fit_sample(X, y)
    #print(' After Sampling X length:',len(X_smk))
    #print(' After Sampling y length:',len(y_smk))

    X_smk.reset_index(drop=True, inplace=True)
    y_smk.reset_index(drop=True, inplace=True)
    
    return (X_smk, y_smk)

###########
#  SMOTENC     
###########

def smoteTenc(X,y):
    
    from imblearn.over_sampling import SMOTENC
    
    cat_features,num_features = features(X)

    Smotenc = SMOTENC(categorical_features=[i for i, e in enumerate(X) if e in cat_features],random_state=42)
    X_nc, y_nc = Smotenc.fit_sample(X,y)
    #print(' After Sampling X length:',len(X_nc))
    #print(' After Sampling y length:',len(y_nc))

    X_nc.reset_index(drop=True, inplace=True)
    y_nc.reset_index(drop=True, inplace=True)
   
    #################
    # Split the data
    #################
    #X_train, X_test, y_train, y_test = train_test_split(X_smT, y_smT, test_size=0.3, stratify=y,random_state=42)
    
    return (X_nc, y_nc)


######################################
# Routine to handle imbalanced process
######################################

def procesImbal(Options,X_train,y_train,targetName):
    
    if (Options['UndSmpDft'] == 'Y'):        
        X_train,y_train = downUpSample(X_train,y_train,'d',targetName)

    if (Options['OvrSmpDft'] == 'Y'):        
        X_train,y_train = downUpSample(X_train,y_train,'u',targetName)
        
    if (Options['RndUndSmplr'] == 'Y'):        
        X_train, y_train = RndUndOvrSample(X_train,y_train,float(Options['RndUndSmplr_thresh']),'d')
               
    if Options['RndOvrSmplr'] == 'Y':        
        X_train, y_train = RndUndOvrSample(X_train,y_train,float(Options['RndOvrSmplr_thresh']),'u')
        
    if Options['OvrSmpSMOT'] == 'Y':        
        X_train,y_train = smote(X_train,y_train,float(Options['OvrSmpSMOT_thresh']))

    if Options['OvrSmpSMOTnc'] == 'Y':        
        X_train,y_train = smoteTenc(X_train,y_train)
        
    if Options['UpSmpSMOTek'] == 'Y': 
        X_train,y_train = smoteTomek(X_train,y_train,float(Options['UpSmpSMOTek_thresh']))

    return (X_train,y_train)

#################################### 
# Routine to Delete unwanted columns 
#################################### 

def delCols(Options, train_df,test_df):

    if (Options['RmvCols'] != []):

        train_df       = train_df.drop(Options['RmvCols'],axis=1)
        test_df        = test_df.drop(Options['RmvCols'],axis=1)    

    #print('Removed unwanted columns')

    return(train_df,test_df)


############################################################# 
# Routine to Scale the numerical and categorical attributes     
#############################################################

def procesScaling(Options,X_train,X_test,test_df,num_cols_names,cat_cols_names):
    
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder

    scaleFlg = 'N'
    ###################
    # Numerical columns
    ###################
    if (Options['NumSclFlg'] == 'Y'):  

        if (Options['NumScl']  == 'Standardization'):  
            scaler = preprocessing.StandardScaler()
        if (Options['NumScl']  == 'Normalization'):   
            scaler = preprocessing.MinMaxScaler()
        if (Options['NumScl']  == 'Robust Scaler'):   
            scaler = preprocessing.RobustScaler()

        X_train[num_cols_names] = scaler.fit_transform(X_train[num_cols_names])
        X_train                 = pd.DataFrame(X_train, columns = X_train.columns)
        
        if (Options['dfultSplit'] == 'Y'):
            X_test[num_cols_names]  = scaler.transform(X_test[num_cols_names])
            X_test                  = pd.DataFrame(X_test, columns = X_test.columns)
        
        test_df[num_cols_names] = scaler.transform(test_df[num_cols_names])
        test_df                 = pd.DataFrame(test_df, columns = X_test.columns)
        scaleFlg                = 'Y'
    ####################
    # Categorical OneHot
    ####################
    if ( (Options['CatSclFlg'] == 'Y') and (Options['OneHotEnc'] != [])): 
        
        X_train  = pd.get_dummies(X_train, columns= Options['OneHotEnc'],drop_first=True)
        
        if (Options['dfultSplit'] == 'Y'):
            X_test   = pd.get_dummies(X_test,  columns= Options['OneHotEnc'],drop_first=True)
        
        test_df  = pd.get_dummies(test_df, columns= Options['OneHotEnc'],drop_first=True)
        scaleFlg = 'Y'
        
    ##########################
    # Categorical Label Encode   
    ##########################  
    
    if ( (Options['CatSclFlg'] == 'Y') and (Options['LblEnc'] != [])):   

        le                         = LabelEncoder()    
        X_train[Options['LblEnc']] = X_train[Options['LblEnc']].apply(lambda col: le.fit_transform(col))  
        
        if (Options['dfultSplit'] == 'Y'):
            X_test[Options['LblEnc']]  = X_test[Options['LblEnc']].apply(lambda col: le.fit_transform(col)) 
        
        test_df[Options['LblEnc']] = test_df[Options['LblEnc']].apply(lambda col: le.fit_transform(col)) 
        scaleFlg                   = 'Y'
        
    #####################################################################################################
    # Recreate category and numerical columns names as label and One-hot will convert category to Numeric
    #####################################################################################################
    if (scaleFlg =='Y'):
        
        cat_cols_names,num_cols_names = features(X_train)
        #test_cat_cols,test_num_cols   = features(X_test)
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        
    #print('Data Scaling Process Completed')
    
    return (X_train,X_test,test_df,num_cols_names,cat_cols_names)
  
###############################################
# Seperate Numerical and categorical attributes
###############################################

def splitNumCat(X_train,Options):
    
    ###########################################################
    # Check for conversion from numeric to categorical, if any)
    ###########################################################

    if (Options['ChgNumChr'] != []):
        
        X_train[Options['ChgNumChr']] = X_train[Options['ChgNumChr']].astype('object',copy=False)
        
    cat_cols_names,num_cols_names = features(X_train)
   
    return (cat_cols_names,num_cols_names)

################################ 
# Detect and treat the Outliers
################################ 

def impOutlyr(X_train,X_test,test_df,num_cols_names,Options):
    
    if (Options['OutlyrImp'] == 'Y'): 
        detectOutliers(X_train[num_cols_names],'Train Data')
        detectOutliers(test_df[num_cols_names],'Test Data')
        
        X_train[num_cols_names]  = treatOutliers(X_train[num_cols_names],'train')
        
        if (Options['dfultSplit'] == 'Y'):
            X_test[num_cols_names]   = treatOutliers(X_test[num_cols_names],'test')
        
        test_df[num_cols_names]  = treatOutliers(test_df[num_cols_names],'Test')
            
    return (X_train,X_test,test_df)

        
################################    
# Routine to impute missing data
################################

def impute_missing_data(data):
    
    global impute_missing
    
    for col in data.columns:
        if data[col].dtype !='object':
            if data[col].isna().any():
                data[col].fillna(data[col].median(),inplace=True)
                impute_missing = 'Y'
                #print(f'Missinfg data in {col} imputed with median value')
        elif data[col].dtype =='object':
            if data[col].isna().any():
                data[col].fillna(data[col].mode()[0],inplace=True)
                impute_missing = 'Y'
                #print(f'Missinfg data in {col} imputed with mode')
                
                
def Label_encoding(data,view_classes = False):
    """Returns Label Endoded data frame ."""
    classes = {}
    le = LabelEncoder()
    for c in data.columns:
        data[c] = le.fit_transform(data[[c]])
        classes[c] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    if view_classes:
        display(classes)
    return data
    
######################################
# Routine to process various ML models
######################################

def buildML(X_train, y_train, X_test, y_test,Options,problem):
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from xgboost import XGBClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn import linear_model

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from xgboost import XGBRegressor

    if (Options['dfultSplit'] == 'Y'):
        splitName    = 'DFT'

    if ( (Options['dfultSplit'] == 'Y') & (Options['UndSmpDft'] == 'Y')):
        splitName    = 'DFT-DS'

    if ((Options['dfultSplit'] == 'Y') & (Options['RndUndSmplr'] == 'Y')):
        splitName    = 'DFT-RDS'

    if ((Options['dfultSplit'] == 'Y') & (Options['OvrSmpDft'] == 'Y')):
        splitName    = 'DFT-US'

    if ((Options['dfultSplit'] == 'Y') & (Options['RndOvrSmplr'] == 'Y')):
        splitName    = 'DFT-RUS'

    if ((Options['dfultSplit'] == 'Y') & (Options['OvrSmpSMOT'] == 'Y')):
        splitName    = 'DFT-SMO'

    if ((Options['dfultSplit'] == 'Y') & (Options['OvrSmpSMOTnc'] == 'Y')):
        splitName    = 'DFT-SMO'

    if ((Options['dfultSplit'] == 'Y') & (Options['UpSmpSMOTek'] == 'Y')):
        splitName    = 'DFT-SMOEK'

    if ((Options['strtkFld'] == 'Y')):
        splitName    = 'SKF'

    if ((Options['kFld'] == 'Y')):
        splitName    = 'KF'
        
    if (Options['rrSplit'] == 'Y'):
        splitName    = 'RRS'
    
    if Options['NN'] == 'Y':
        params = {}
        processData(X_train, y_train, X_test, y_test, '','NN',splitName,params,Options,problem) 
        Options['NN'] = 'N'
   
    if Options['DT'] == 'Y':
        params = {}    
        processData(X_train, y_train, X_test, y_test, DecisionTreeClassifier,'DT',splitName,params,Options,problem) if problem == 'Classification' else processData(X_train, y_train, X_test, y_test, DecisionTreeRegressor,'DT',splitName,params,Options,problem)

    if Options['LR'] == 'Y':
        params = {}
        processData(X_train, y_train, X_test, y_test, LogisticRegression,'LR',splitName,params,Options,problem) if problem == 'Classification' else processData(X_train, y_train, X_test, y_test, LinearRegression,'LR',splitName,params,Options,problem)

    if Options['RF'] == 'Y':
        params = {}
        processData(X_train, y_train, X_test, y_test, RandomForestClassifier,'RF',splitName,params,Options,problem) if problem == 'Classification' else processData(X_train, y_train, X_test, y_test, RandomForestRegressor,'RF',splitName,params,Options,problem)

    if Options['XGB'] == 'Y':
        params = {}
        processData(X_train, y_train, X_test, y_test, XGBClassifier,'XGB',splitName,params,Options,problem) if problem == 'Classification' else processData(X_train, y_train, X_test, y_test, XGBRegressor,'XGB',splitName,params,Options,problem)

    if Options['GB'] == 'Y':
        params = {}
        processData(X_train, y_train, X_test, y_test, GradientBoostingClassifier,'GB',splitName,params,Options,problem) if problem == 'Classification' else processData(X_train, y_train, X_test, y_test, GradientBoostingRegressor,'GB',splitName,params,Options,problem)

    if Options['ADA'] == 'Y':
        params = {}        
        processData(X_train, y_train, X_test, y_test, AdaBoostClassifier,'ADA',splitName,params,Options,problem) if problem == 'Classification' else processData(X_train, y_train, X_test, y_test, AdaBoostRegressor,'ADA',splitName,params,Options,problem)

    if Options['NB'] == 'Y':
        params = {}
        processData(X_train, y_train, X_test, y_test, GaussianNB,'NB',splitName,params,Options,problem) 
        
##############################################################################
# Routine to process Stratified & K Fold Cross Validation data for model build   
##############################################################################

def processData(X_train, y_train, X_test, y_test, classifier, mlName,splitName,param_grid, Options,problem, n_jobs=-1):
    
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.model_selection import ShuffleSplit
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from xgboost import XGBClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn import linear_model

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from xgboost import XGBRegressor
  
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping

    ###############################################################################
    # For Stratified & K Fold Cross Validation or Repeated Random Test-Train Splits
    ###############################################################################
    
    if (Options['strtkFld'] == 'Y'):
        Cv = StratifiedKFold(n_splits=int(Options['strtKFld_thresh']),random_state=None,shuffle=True)
        
    if (Options['kFld'] == 'Y'):
        Cv = KFold(n_splits=int(Options['kFld_thresh']),shuffle=True,random_state=None)
        
    if (Options['rrSplit'] == 'Y'):
        Cv = ShuffleSplit(n_splits=int(Options['rrSplit_thresh']),test_size=0.30,random_state=None)
    
    if ((Options['strtkFld'] == 'Y') | (Options['kFld'] == 'Y') | (Options['rrSplit'] == 'Y')):
        
        accuracy_  = []
        recall_    = []
        precision_ = []
        f1_        = []
        logloss_   = []
        rmse_      = []
        rsqu_      = []
        score_     = []
        
        X          = X_train
        y          = y_train
    
        for train_index , test_index in Cv.split(X,y):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
            
            ##################
            # Construct  model  
            ##################

            if (Options['NN'] == 'Y'):

                model = build_ann_model(len(X_train.columns),problem)
                
                if problem == 'Classification':
                    eS  = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5) 
                    model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test),verbose=0, 
                              callbacks   = [eS,accuracyTresholdCallback()])
                    
                if problem == 'Regression':
                    model.fit(X_train, y_train)
            
            else:
                model = classifier(**param_grid).fit(X_train,y_train)
            
            if problem == 'Classification':
                rmse_ = rsqu_ = 0
                
                if (Options['NN'] == 'Y'):
                    
                    pred  = model.predict(X_test).round()
                    score_.append(0)
                    accuracy_.append(accuracy_score(y_test, model.predict(X_test).round()))
                    recall_.append(recall_score(y_test, model.predict(X_test).round()))
                    precision_.append(precision_score(y_test, model.predict(X_test).round()))
                    f1_.append(f1_score(y_test, model.predict(X_test).round()))
                    logloss_.append(log_loss(y_test, pred))
                else:
                    pred  = model.predict_proba(X_test) 
                    score_.append(cross_val_score(model, X_train, y_train, cv=Cv))
                    accuracy_.append(accuracy_score(y_test, model.predict(X_test)))
                    recall_.append(recall_score(y_test, model.predict(X_test)))
                    precision_.append(precision_score(y_test, model.predict(X_test)))
                    f1_.append(f1_score(y_test, model.predict(X_test)))
                    logloss_.append(log_loss(y_test, pred))                
            else:
                score_ = accuracy_ = recall_ = precision_ = f1_ = logloss_ = 0
                pred = model.predict(X_test)
                rmse_.append(np.sqrt(mean_squared_error(y_test, pred)))
                rsqu_.append(r2_score(y_test, pred))
                
        Get_Metrics(mlName,splitName,round(np.mean(score_),2),round(np.mean(accuracy_),2),round(np.mean(recall_),2),round(np.mean(precision_),2),round(np.mean(f1_),2),round(np.mean(logloss_),2),round(np.mean(rmse_),2),round(np.mean(rsqu_),2),problem)
                  
    ####################
    # Traditional  Split
    ####################
    if (Options['dfultSplit'] == 'Y'):
        
        ##################
        # Construct  model
        ##################
        
        if (Options['NN'] == 'Y'):

            model = build_ann_model(len(X_train.columns),problem)
            
            if problem == 'Classification':
                eS = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5,restore_best_weights = True) 
                model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=10, validation_data=(X_test, y_test), 
                          callbacks=[eS,accuracyTresholdCallback()])    
        
            if problem == 'Regression':
                model.fit(X_train, y_train)            
        
        else:
            model = classifier(**param_grid).fit(X_train,y_train)
        
        
        if problem == 'Classification':
            rmse = rsqu = 0
            
            if (Options['NN'] == 'Y'):
                pred      = model.predict(X_test).round()
                accuracy  = accuracy_score(y_test, model.predict(X_test).round())
                recall    = recall_score(y_test, model.predict(X_test).round())
                precision = precision_score(y_test, model.predict(X_test).round())
                f1        = f1_score(y_test, model.predict(X_test).round())
                logloss   = log_loss(y_test, pred)
                score     = 0
            else:
                pred      = model.predict_proba(X_test)#[:,1]
                score     = model.score(X_train, y_train)
                accuracy  = accuracy_score(y_test, model.predict(X_test))
                recall    = recall_score(y_test, model.predict(X_test))
                precision = precision_score(y_test, model.predict(X_test))
                f1        = f1_score(y_test, model.predict(X_test))
                logloss   = log_loss(y_test, pred)   
        else:
            score = accuracy = recall = precision = f1 = logloss = 0
            pred  = model.predict(X_test)
            #pred  = model.predict_proba(X_test)
            #y_test = y_test.values
            rmse  = np.sqrt(mean_squared_error(y_test, pred))
            rsqu  = r2_score(y_test, pred)

        Get_Metrics(mlName,splitName,round(score,2),round(accuracy,2),round(recall,2),round(precision,2),round(f1,2),round(logloss,2),round(rmse,2),round(rsqu,2),problem)
        
    return model

######################################################################
# Routine to build a simple ANN structure with a binary classification
######################################################################

def build_ann_model(inputs,problem):
        
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    from keras import backend as K
    #from sklearn.neural_network import MLPRegressor
    from keras import regularizers

    def coeff_determination(y_test, pred):
        SS_res =  K.sum(K.square(y_test-pred))
        SS_tot = K.sum(K.square(y_test - K.mean(y_test) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    
    if problem == 'Classification':
        model = Sequential()
        model.add(Dense(32, kernel_initializer = 'he_uniform', input_dim=inputs, activation='relu'))                         
        model.add(Dense(16, kernel_initializer = 'he_uniform',activation='relu'))                                          
        model.add(Dense(1, activation='sigmoid'))                                         
        model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    
    if problem == 'Regression':       

        model = Sequential()
        model.add(Dense(100, input_dim=inputs, activation='relu',kernel_regularizer=regularizers.l2(0.01)))   
        #Dropout(0.3) 
        model.add(Dense(50,activation='relu',kernel_regularizer=regularizers.l2(0.01))) 
        model.add(Dense(1,activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss= "mean_squared_error", metrics = ['mean_squared_error'])       

    return model

#################################################################
# Build Model from train data based on the final metric selection
#################################################################

def finalProcess(X_train, y_train, classifier, dataSplit, param_grid, Options, n_jobs=-1):
        
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    
    ##################################################
    # Routine for Stratified & K Fold Cross Validation
    ##################################################
    
    if (dataSplit[0] == 'SKF'):
        Cv = StratifiedKFold(n_splits=int(Options['strtKFld_thresh']),random_state=None,shuffle=True)
        
    if (dataSplit[0] == 'KF'):
        Cv = KFold(n_splits=int(Options['kFld_thresh']),random_state=None)
    
    if ((dataSplit[0] == 'SKF') | (dataSplit[0] == 'KF')):
        
        X          = X_train
        y          = y_train
    
        for train_index , test_index in Cv.split(X,y):
            X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
            model            = classifier(**param_grid).fit(X_train,y_train)
                       
    if (dataSplit[0] == 'DFT'):
                
        model     = classifier(**param_grid).fit(X_train,y_train)
        
    return model

#########################################
# Subroutine to build the metric outfile
#########################################
  
def Get_Metrics(classifier,dataSplit,cv,accuracy,recall,precision,f1,logloss,rmse,rsqu,problem):
        
    Classifier.append(classifier)
    DataSplit.append(dataSplit)
    cvScore.append(cv)
    Accuracy.append(accuracy)
    Recall.append(recall)
    Precision.append(precision) 
    F1.append(f1)
    Logloss.append(logloss)  
    Rmse.append(rmse)
    R2sq.append(rsqu) 

#Options = Auto_ml()