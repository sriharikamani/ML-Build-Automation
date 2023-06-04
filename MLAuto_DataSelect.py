##########################################################################################################################
# Program to select the database for Auto ML processing
##########################################################################################################################

import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import tensorflow as tf

def Get_datasets():
    
    tf.keras.backend.clear_session()
    root = Tk()
    root.geometry('+%d+%d'%(500,10))    
    path = os.getcwd()
    root.iconbitmap(path+ '\psx.ico') 
    Psx_img = PhotoImage(file = path + '\psx.png')   
    root.title('Select the Database to Process')
    final_res = {}      
    ##################################
    # header area - Logo, Title & Time
    ##################################
    header = Frame(root, width=900, height=195, bg="white")
    header.grid(columnspan=3, rowspan=2, row=0)

    head_title = Label(root,text = "Database Select",bg ='#FFFFFF',fg= '#004488', font = ('Poppins',30,"bold"))
    head_title.place(x = 350,y=90)
    header = Label(root,image = Psx_img)
    header.place(x=0,y=1)
    
    Org_title = Label(root,text = "POSIDEX",bg ='#FFFFFF',fg= '#004488', font = ('sans-serif',34,"bold") )
    Org_title.place(x=0,y=139, height=38)
    
    Info_title = Label(root,text = "Select the Test and Train datasets",bg ='#FFFFFF',fg= '#004488', font = ('Poppins',8,"bold"))
    Info_title.place(x = 2,y=175)
    
    time = Label(root, text=f"{'{0:%d-%m-%Y %H:%M %p}'.format(datetime.now())}",bg ='#FFFFFF',fg= '#004488',
                 font = ('Poppins',12,"bold"))
    time.place(x = 735,y=-1)
    
    # MainFrame Area
    main_content = Frame(root,highlightbackground="#f80", highlightcolor="#f80", highlightthickness=2, width=900, height=300, bg="#048")
    main_content.grid(columnspan=3, rowspan=2, row=2)
   
    ######################################### 
    # Routine to get the path of the document                               
    ######################################### 
    
    def Submit():
        result()
        root.destroy()
       
    def result():
        
        train     = filedialog.askopenfilename(title = "Select File") 
        test      = filedialog.askopenfilename(title = "Select File") 
        final_res['train'] = train
        final_res['test']  = test
        return final_res
    
    def Clear(): 
        root.destroy()
        Get_datasets()
    
    def Exit():
        root.destroy()
       
    Exit_Button = Button(root, text=" Exit",command=Exit,bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14),
                         relief=RAISED,bd=6)
    Exit_Button.place(x = 350, y =200)

    Submit_Button = Button(root, text=" Submit",command=Submit,bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14),
                         relief=RAISED,bd=6)
    Submit_Button.place(x = 450, y =200)
    
    Clear_Button = Button(root, text="Clear",command=Clear,bg ='#FFFFFF',fg= '#004488',font = ('Poppins',14),relief=RAISED,bd=6)
    Clear_Button.place(x = 220, y =200)
    
    ## Hover Effects
    def but_1_onbutton(e):
        Submit_Button.config( bg ='#f80',fg= 'white')
    def but_1_leavebutton(e):
        Submit_Button.config( bg ='#FFFFFF',fg= '#004488')
        
    Submit_Button.bind('<Enter>',but_1_onbutton)
    Submit_Button.bind('<Leave>',but_1_leavebutton)
        
    root.mainloop()
    
    return (final_res)