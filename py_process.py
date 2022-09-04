import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#from collections import Counter
#import hypertools as hyp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
#from sklearn.feature_selection import SelectFromModel

import pickle

#####################

def import_excel(path,sheet1,sheet2):
    '''Takes as input filepath and the name of the two sheets to create 2 dataframes'''
    data = pd.read_excel(path,sheet_name=sheet1)
    df = pd.read_excel(path,sheet_name=sheet2)
    return data,df


def create_nlst_column(data,dataframe,n,column):
    
    '''Creates pivot table for the number of 'n' last chronological transactions , then uses that to create unique culumn per chronological transaction on the given attribute.
    Input:
        -data : Customer Info
        -df: Chronological Transaction per customer
        -n: number of last chronological occurences
        -column: the attribute to be transformed
    Output: dataframe with n number of columns for specific attribute
    '''

    table = pd.pivot_table(dataframe[dataframe['cum_count']<=n], values=column, index="Customer",
                    columns='cum_count', fill_value=0)
    data=data.merge(table,how='left',left_on="Customer",right_on="Customer")
    x=[]
    x.append('Customer')
    for i in range(1,n+1):
        x.append(i)
    data=data[x]
    x=[]
    for i in range(1,n+1):
        x.append(column+"_from_last_"+str(i))
        data.rename(columns={i:x[i-1]},inplace=True)
    return data



def pre_process(df,data,today,n_months):
    
    '''All the preprocess required for the data to be used in classification.Uses function "create_nlst_column". '''

    df1=df.groupby(['Customer','Month']).agg({
    "# of Hires":'sum',
    'Headcount/ Assessments Included in Subscription Plan':'max',
    '# of Assessments This Month':'max',
    '# of Invitations Sent':'sum'})

    df1.reset_index(level=0, inplace=True)
    df1.reset_index(level=0, inplace=True)
    dataframe=df1.merge(data, how='left' , left_on='Customer', right_on='Customer')
    end = pd.to_datetime(today)
    #
    #
    dataframe['End Date']= pd.to_datetime(dataframe['End Date'])
    dataframe['End Date'] = dataframe['End Date'].fillna(end)
    
    for i in dataframe['End Date']:
        if i>end:
            i=end
        else:
            pass
    
    
    
    dataframe['Start Date']= pd.to_datetime(dataframe['Start Date'])
    dataframe['Since_Start']=abs(dataframe['Month']-dataframe['Start Date']).dt.days.astype('int16')
    dataframe['Since_End']=abs(dataframe['End Date']-dataframe['Month']).dt.days.astype('int16')
    dataframe.sort_values(by='Month', inplace=True)
    dataframe['Month_diff'] =         [str(n.days)  if n > pd.Timedelta(days=1) else '0' if pd.notnull(n) else "0" 
        for n in dataframe.groupby('Customer', sort=False)['Month'].diff()]

    dataframe=dataframe[['Customer','Month','Month_diff', 'Start Date','Since_Start','End Date','Since_End',
          '# of Hires', 'Headcount/ Assessments Included in Subscription Plan', 
          '# of Assessments This Month', '# of Invitations Sent',  'Status',
          ]]
    dataframe['Month_diff']=dataframe['Month_diff'].astype(int)
    dataframe.sort_values(by='Month', inplace=True,ascending=False)
    dataframe['cum_count'] = dataframe.groupby("Customer").cumcount()+1
    dataframe["plan_full"]=dataframe["# of Assessments This Month"]/dataframe["Headcount/ Assessments Included in Subscription Plan"]
    dataframe['hire_metr']=dataframe["# of Hires"]/dataframe["# of Assessments This Month"]
    
    a=dataframe[dataframe['cum_count']==1]
    pattern=data[['Status','Customer']].merge(a[['Customer','Since_Start','Since_End']],how='left',left_on="Customer",right_on="Customer")
    df=create_nlst_column(data,dataframe,n_months,'plan_full')
    pattern=pattern.merge(df,how='left',left_on="Customer",right_on="Customer")
    #df=create_nlst_column(data,dataframe,n_months,'hire_metr')
    #pattern=pattern.merge(df,how='left',left_on="Customer",right_on="Customer")
    df=create_nlst_column(data,dataframe,n_months,'# of Invitations Sent')
    pattern=pattern.merge(df,how='left',left_on="Customer",right_on="Customer")
    df=create_nlst_column(data,dataframe,n_months,'Month_diff')
    pattern=pattern.merge(df,how='left',left_on="Customer",right_on="Customer")
    return pattern


def models_no_dim_method( x, y , models,parameters,method_details,a):
        
    ''' Classification with multiple classifiers,grid search,feature selection.The classfiers with best params and recall score is then used. Returns tables with predictions'''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True,stratify=y)
    #το number του iteration βασει του οποιου θα ταιραξει classifier & parameters
    i=0
    #results table
        
    results=pd.DataFrame(columns=["Methodology","Score",])
        
    selector=SelectFromModel(RandomForestClassifier(),max_features=a)
    #for loop
    for classifier in models:
    
            
            pipe = Pipeline([('scaller',MinMaxScaler()),('featselect',selector ),
                             ('classifier', classifier) ])
            pipe.fit(X_train, y_train)
            
            mask = selector.get_support()
            new_features = x.columns[mask]
            #grid search      
            grid = GridSearchCV(pipe, parameters[i] ,scoring='balanced_accuracy').fit(X_train, y_train)
            #βbest params of each pipe
            best_params = grid.best_params_
            optimised_pipe = grid.best_estimator_
            #run optimised pipe
            optimised_pipe.fit(X_train, y_train)
            #prediction
            y_true1, y_pred1 = y_test , optimised_pipe.predict(X_test)
            #cross validate        
            scores = cross_validate(optimised_pipe, x, y, cv=20,
                    scoring=('average_precision', 'balanced_accuracy','roc_auc','recall'),
                    return_train_score=True)
            #results  table updated 
            results = results.append({
                'Methodology': 'Grind'+method_details+str(a)+str(classifier).split('(')[0],
                'Optimised pipe':str(optimised_pipe),
                'Picle':'Grind'+method_details+str(a)+str(classifier).split('(')[0],
                'test_balanced_accuracy':format(scores['test_balanced_accuracy'].mean(),'.2f'),
                'train_balanced_accuracy':format(scores['train_balanced_accuracy'].mean(),'.2f'),
                'test_average_precision':format(scores['test_average_precision'].mean(),'.2f'),
                'train_average_precision':format(scores['train_average_precision'].mean(),'.2f'),
                'test_recall':format(scores['test_recall'].mean(),'.2f'),
                'train_recall':format(scores['train_recall'].mean(),'.2f'),
                'test_roc_auc':format(scores['test_roc_auc'].mean(),'.2f'),
                'train_roc_auc':format(scores['train_roc_auc'].mean(),'.2f'),
                'scores':scores,
                'best params': best_params,
                                             
                'Score': format(optimised_pipe.score(X_test, y_test),'.2f'),
                'Precision 0':format(precision_recall_fscore_support(y_true1, y_pred1, average=None)[0][0],'.2f'),
                'Precision 1':format(precision_recall_fscore_support(y_true1, y_pred1, average=None)[0][1],'.2f'),
                'F1 0':format(precision_recall_fscore_support(y_true1, y_pred1, average=None)[2][0],'.2f'),
                'F1 1':format(precision_recall_fscore_support(y_true1, y_pred1, average=None)[2][1],'.2f'),
                'Recal 0':format(precision_recall_fscore_support(y_true1, y_pred1, average=None)[1][0],'.2f'),
                'Recal 1':format(precision_recall_fscore_support(y_true1, y_pred1, average=None)[1][1],'.2f'),
                #'select_from_model_params':SelectFromModel.get_params(),
                'select_from_model':new_features
                },ignore_index=True)
            #save each model
            filename = 'Grind'+method_details+str(a)+str(classifier).split('(')[0]
            pickle.dump(optimised_pipe, open(filename, 'wb'))
        

                
            i+=1
    results.sort_values(['test_recall', 'test_roc_auc'], ascending=[False, False], inplace=True)
    optimised=results['Picle'][0]
    loaded_model = pickle.load(open(optimised, 'rb'))
    Y_pred = loaded_model.predict(x)
        

    return results,Y_pred,y 

  
def give_input():
    ''''Uses input as variables for importing data
    today: curent date in format YYYY-MM-DD
    n_month: number of months that will be refered to for early warning
    path: the directory of the data file
    sheet1: the name of the sheet with the customer info'''

    today=str(input("Enter current date in the following format: YYYY-MM-DD "))
    #today='2022-02-14'
    n_months=int(input("Enter how many months prior you wish to examine, it should be a number.e.g. 1 = last month only, 2=last two months"))
    #n_months=4
    #για windows
    #b='\\'
    #f='/'
    #path=str(input("Enter path of excell file (xlsx)")).replace(b,f)
    path=str(input("Enter path of excell file (xlsx)"))
    #path=r'/Users/chrestoslogaras/Downloads/BigBlueAcademy __ Bryq (1).xlsx'
    sheet1=str(input("Enter name of sheet with customers"))
    #sheet1=r'Customer Info'
    sheet2=str(input("Enter name of sheet with montly data"))
    #sheet2=r"Data per Month"
    
    return today,n_months,path,sheet1,sheet2
    
    





par1={ 
    
    0:{'classifier__n_estimators':[5,20, 50, 100, 300, 500],
       'classifier__criterion':['gini','entropy'],},
    
    1:{'classifier__loss':['deviance', 'exponential'],
               'classifier__learning_rate':[0.1,0.5,1],},
    2:{'classifier__learning_rate':[0.1,0.5,1],
               'classifier__algorithm':['SAMME', 'SAMME.R'],},
    3:{'classifier__criterion':['gini', 'entropy'],
               'classifier__splitter':['best', 'random'],},
    4:{'classifier__base_estimator':[DecisionTreeClassifier(),RandomForestClassifier()]},
    }

clas=[RandomForestClassifier(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    DecisionTreeClassifier(),
    BaggingClassifier()]

#############################
today,n_months,path,sheet1,sheet2=give_input()
data,df=import_excel(path,sheet1,sheet2)
pattern=pre_process(df,data,today,n_months)
total=pd.DataFrame()
x=pattern.drop(columns=['Status', 'Customer','Since_Start',"Since_End"])
y=pattern['Status'].replace({'active':0,'paused': 0,'cancelled':1,'non_renewing':1 })

results,Y_pred,y = models_no_dim_method( x, y , clas, par1," max feat select (without strt,end date ) for number of periods",n_months)

early_warning = pattern
early_warning['class'] = Y_pred

early_warning.to_csv('early_warning.csv',index=False)
                     
results.to_csv('classification_results.csv',index=False)

#Δυνητικά μπορεί να προστεθεί το κάτωθι confusion matrix
#from sklearn.metrics import confusion_matrix
#import numpy as np
#import seaborn as sns

#cf_matrix = confusion_matrix(y , Y_pred)
#group_names = ['True Neg','False Pos','False Neg','True Pos']
#group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
#group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

#labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
#labels = np.asarray(labels).reshape(2,2)
#sns.color_palette('pastel', as_cmap=True)
#ax = sns.heatmap(cf_matrix, annot=labels, fmt='',cmap='Pastel1',annot_kws={'fontsize':'small'})

#ax.set_title('Confusion Matrix\n\n');
#ax.set_xlabel('\nPredicted Values')
#ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])
    ## Display the visualization of the Confusion Matrix.
#plt.show()

#early_warning[]

