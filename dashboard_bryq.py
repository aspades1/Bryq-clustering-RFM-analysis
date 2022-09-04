##### libraries
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

####### functions
def data_preprocess(dataframe,df,date):
    newdf = df.drop_duplicates(
        subset = ['Customer', 'Month'],keep = 'last').reset_index(drop = True)
    newdf1 = newdf.groupby(['Customer'])['# of Assessments This Month'].sum()
    dataframe=dataframe.merge(newdf1.to_frame(),how='left',left_on="Customer",right_on="Customer")
    dataframe=dataframe.merge(newdf.groupby(['Customer'])['Month'].count(),how='left',left_on="Customer",right_on="Customer",)
   
    today=date
    end=pd.to_datetime(today)
    dataframe['End Date'] = dataframe['End Date'].fillna(end)
    for i in dataframe['End Date']:
        if i>end:
            i=end
        else:
            pass
    dataframe['Since last'] = end-dataframe["End Date"]
    dataframe['Since last'] = dataframe['Since last'].dt.days.astype('int16')
    
    return dataframe, df

def rfm_labes(dataframe):
    # top=4 
    # Create labels for Recency and Frequency
    r_labels = [4,3,2,1]; f_labels = [1,2,3,4];m_labels = [1,2,3,4]
    # Assign these labels to 4 equal percentile groups 
    r_groups = pd.cut(dataframe['Since last'].rank(pct=True), bins=4, labels=r_labels)
    # Assign these labels to 4 equal percentile groups 
    f_groups = pd.cut(dataframe['Month'].rank(pct=True), bins=4, labels=f_labels)
    # Assign these labels to 4 equal percentile groups 
    m_groups = pd.cut(dataframe['MRR'].rank(pct=True), bins=4, labels=m_labels)
    # Create new columns R and F 
    dataframe = dataframe.assign(R = r_groups.values, F = f_groups.values,M=m_groups.values)
    return dataframe

    
def join_rfm(x): 
    return str(x['R']) + str(x['F']) + str(x['M'])


def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Can\'t Loose Them'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 9)):
        return 'Champions'
    elif ((df['RFM_Score'] >= 7) and (df['RFM_Score'] < 8)):
        return 'Loyal'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7)):
        return 'Potential'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6)):
        return 'Promising'
    elif ((df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'
    
def create_rfm_table(dataframe):
    rfm = dataframe[['RFM_Segment_Concat','Customer','Status','R','F','M','Since last','Month','MRR']]
    rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)   
    #rfm_count_unique = rfm.groupby('RFM_Segment_Concat')['RFM_Segment_Concat'].nunique()
    rfm=rfm[rfm['Status']!='non_renewing']    
    # Create a new variable RFM_Level
    rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)
    # Calculate average values for each RFM_Level, and return a size of each segment 
    return rfm

def cust_segm(rfm):

    rfm.replace({'Status': {'active': 2, 'paused': 1,'cancelled':0}},inplace=True)
    rfm.replace({'Since last': {0:30}},inplace=True)
    rfm.rename(columns={"Since last": "Recency", "Month": "Frequency",'MRR':"MonetaryValue"},inplace=True)
    rfm.reset_index(drop=True, inplace=True)
    customers_fix = pd.DataFrame()
    customers_fix["Status"] = rfm['Status']
    customers_fix["Recency"] = rfm["Recency"]
    
    customers_fix["Frequency"] = stats.boxcox(rfm["Frequency"])[0]
    customers_fix["MonetaryValue"] = stats.boxcox(rfm["MonetaryValue"])[0]
    scaler = StandardScaler()
    
    scaler.fit(customers_fix)
    customers_normalized = scaler.transform(customers_fix)
    # Assert that it has mean 0 and variance 1
    #print(customers_normalized.mean(axis = 0).round(3)) # [0. -0. 0.]
    #print(customers_normalized.std(axis = 0).round(3)) # [1. 1. 1.]
    return customers_normalized


def clustering(customers_normalized,n):
    model_kmeans = KMeans(n_clusters=n, random_state=0)
    model_kmeans.fit(customers_normalized )
    return model_kmeans,model_kmeans.labels_


def convert_df(df):
    return df.to_csv().encode('utf-8')

###### app##############################
st.set_page_config(layout="wide")

st.title('Bryq customer app')
st.text('\n\n\n')

page = st.selectbox("Choose your page", ["RFM", "Clustering", "Early warning"]) 
if page == "RFM":
    entry=st.container()
    rfm=st.container()
    
    with entry:
        
        st.subheader('In this app you must upload an excel file. \n The first sheet must be named "Customer Info" and the second "Data per Month". \n Then you must choose date the data correspond too etc.\n\n\n\n\n')
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader('Please upload an excel file')
        with col2:
            date1=st.date_input('Please choose date the data was downloaded')
        st.text('\n\n\n\n')


        if uploaded_file and date1 is not None:
            dataframe = pd.read_excel(uploaded_file,sheet_name='Customer Info')
            df = pd.read_excel(uploaded_file,sheet_name='Data per Month')
        
            with rfm:
                st.header('RFM marketing analysis')
                
                col20,col21=st.columns((2,3))
                with col20:
                    st.text('\n\n\n\n\n\n\n\n\n\n\n\n\nThis is a basic transormation of data based on \nRFM marketing analysis\n\nThe individual Recency-Frequency-Monetery metrics \nare summarised with an index\n\nBased on the value of the index the following \nsegments occur:\n\n\n\n  >= 9  -> Cant Loose Them\n  >= 8  -> Champions\n  >= 7  -> Loyal\n  >= 6  -> Potential\n  >= 5  -> Promising\n  else  -> Require Activation')
            
                dataframe, df = data_preprocess(dataframe,df,date1)
                marketing=rfm_labes(dataframe)
                marketing['RFM_Segment_Concat'] = marketing.apply(join_rfm, axis=1)
                rfm=create_rfm_table(marketing)
                
                with col21:
                    fig1 = px.treemap(rfm, path=['Status', 'RFM_Level','Customer'], values='RFM_Score',color='RFM_Score',color_continuous_scale='thermal')
                    st.plotly_chart(fig1)
                    
                col24,col25,col26=st.columns((2,8,2))
                with col25:
                    rfm_level_agg = rfm.groupby('RFM_Level').agg({'Since last': 'mean','Month': 'mean','MRR': ['mean', 'count']}).round(1)
                    rfm_level_agg.columns = rfm_level_agg.columns.droplevel()
                    rfm_level_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
                    st.text('Below you can find the summary of the customer segments created by RFM marketing analysis')
                    #st.write(rfm_level_agg)
                    
                    gg=rfm_level_agg.index
                    fig09=px.bar(rfm_level_agg, x=gg, y='RecencyMean',log_y=True)
                    st.plotly_chart(fig09)
                    
                    fig010=px.bar(rfm_level_agg, x=gg, y='FrequencyMean',log_y=True)
                    st.plotly_chart(fig010)
                    
                    fig011=px.bar(rfm_level_agg, x=gg, y='MonetaryMean',log_y=True)
                    st.plotly_chart(fig011)
                
                csv = convert_df(rfm)
                st.download_button(label="You can download the output of RFM analalysis here",data=csv,file_name='rfm.csv',mime='text/csv',)
                
                
                

        else:
            st.write('No data uploaded')

    
elif page == "Clustering":
    clusterin=st.container()
    with clusterin:
        
        col3, col4= st.columns(2)
        with col3:
            uploaded_file = st.file_uploader('Please upload an excel file')
        with col4:
            date1=st.date_input('Please choose date the data was downloaded')
        st.text('\n\n\n\n')
        st.title('Unsupervised machine learning- Clustering analysis')


        if uploaded_file and date1 is not None:
            dataframe = pd.read_excel(uploaded_file,sheet_name='Customer Info')
            df = pd.read_excel(uploaded_file,sheet_name='Data per Month')

            dataframe, df = data_preprocess(dataframe,df,date1)
            marketing=rfm_labes(dataframe)
            marketing['RFM_Segment_Concat'] = marketing.apply(join_rfm, axis=1)
            rfm=create_rfm_table(marketing)
            customers_normalized=cust_segm(rfm)
            
            visualizer = KElbowVisualizer(KMeans(),k=(2,15)).fit(customers_normalized)
        
            col5, col7,col6= st.columns((6,2,1))
            with col5:                             
                st.text("Based on the data you uploaded the best number of segments is "+str(visualizer.elbow_value_))
                st.text('You have the ability to modify the number of customer segments \nbased on business inteligence.\nYou can choose the number of clusters here------------------>')
        
            with col7:
                b=st.number_input('Choose the number of clusters', min_value=2, max_value=8, value=visualizer.elbow_value_, step=1)
            
            model,a=clustering(customers_normalized,b) 
            rfm["Cluster"] = a
            df_normalized = pd.DataFrame(customers_normalized, columns=['Status','Recency', 'Frequency', 'MonetaryValue',])
            df_normalized['Customer'] = rfm.Customer
            df_normalized['Cluster'] = a
            
            df_nor_melt = pd.melt(df_normalized.reset_index(),id_vars=['Customer', 'Cluster'],value_vars=['Recency','Frequency','MonetaryValue','Status'],var_name='Attribute',value_name='Value')
            
            col8,col10= st.columns(2)
            with col8:    
                
                st.text('The complete data')
                st.write(rfm)
                csv = convert_df(rfm)
                st.download_button(label="You can download this table as CSV",data=csv,file_name='rfm_n_cluster.csv',mime='text/csv',)
                
                c=df_normalized.groupby('Cluster').agg({
                    "Status":'mean',
                    'Recency':'mean',
                    'Frequency':'mean',
                    'MonetaryValue':'mean'}).round(1)
                c_nor_melt = pd.melt(c.reset_index(),id_vars=['Cluster'],value_vars=['Recency','Frequency','MonetaryValue','Status'],var_name='Attribute',value_name='Value')
            
            with col10:
                st.text('Bellow you can find a graphical linear representation of the variables for each graph')
                fig = plt.figure(figsize=(10, 4))
                sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt,palette="pastel")
                st.pyplot(fig)
            
            col12,col13,col14= st.columns((2,8,1))
            with col13:
                st.text('Bellow you can find a graphical radial representation of the variables for each cluster')
                fig2 = px.line_polar(c_nor_melt, r='Value', theta='Attribute',color='Cluster' ,line_close=True)
                fig2.update_traces(fill='toself')
                st.plotly_chart(fig2)
                
                df=rfm[['Customer',"Recency",'Frequency','MonetaryValue','Status','Cluster']]
                st.text('A more detailed analysis that visualizes the range of the variables for each cluster')
                fig10 = go.Figure(data=
                    go.Parcoords(
                        line = dict(color = df_normalized['Cluster'],
                                   colorscale = 'HSV'),
                        dimensions = list([
                            dict(
                                label = 'Cluster', values = df_normalized['Cluster']),
                            dict(
                                label = 'Recency', values = df_normalized['Recency']),
                            dict(
                                label = "Frequency", values = df_normalized["Frequency"]),
                            dict(
                                label = 'MonetaryValue', values = df_normalized['MonetaryValue']),
                            dict(
                                label = 'Status', values = df_normalized['Status']),
                        ])
                    )
                )

                
                st.plotly_chart(fig10)
            
        else:
            st.write('No data uploaded')

    
elif page == "Early warning":
    warning=st.container()
    
    with warning:
        st.header('Early Warning System')
        st.subheader('In this module you must first run the script py_process.py and then upload the results')
        uploaded_file2 = st.file_uploader('Please upload the csv file with the results of the classification ("early waring.csv)')
        uploaded_file3 = st.file_uploader('Please upload the csv file the performance of the different classifiers( "classification_results.csv")')
        col44,col45= st.columns((5,2))
        if uploaded_file2 and uploaded_file3 is not None:
            with col44:
            
                data1 = pd.read_csv(uploaded_file2)
                data2 = pd.read_csv(uploaded_file3)
                st.text('The customer that were identified as dangerous are shown below')
                st.write(data1[data1['class']==1])
                
            with col45:
                st.text('The variables identified \nas statistically significant:')
                my_list=(data2.select_from_model[0].split('[')[1].split(']')[0].replace("'","").split(', '))
                st.write(my_list)
             
            st.write('Based on the varibles identified as significant for churned customers below you can find graphical representation of those')
            col54,col55,col56= st.columns(3)  
            with col54:
                option1 = st.selectbox('Choose the first variable to be examined',my_list)
        
            with col55:
                option2 = st.selectbox('Choose the second variable to be examined',my_list)
            with col56:  
                option3 = st.selectbox('Choose the third variable to be examined',my_list)
            
                
            col64,col65,col66= st.columns((2,5,1))  
            with col65:
                fig28 = px.scatter(data1, x=option1, y=option2, color='class',symbol="Status",hover_name='Customer',log_x=True, log_y=True,color_continuous_scale='Picnic')
                fig28.update_layout(
                    plot_bgcolor = 'grey',
                    paper_bgcolor = 'grey'                    )
                #fig28.update_traces(marker_showscale=False)
                #fig28.update_traces(showscale=False)
                fig28.update_coloraxes(showscale=False)
                st.plotly_chart(fig28)
            
            col74,col75,col76= st.columns((2,5,1))  
            with col75:
                fig29 = px.scatter_3d(data1, x=option1, y=option2,z=option3 ,color='class',symbol="Status",hover_name='Customer',log_x=True, log_y=True,log_z=True,color_continuous_scale='Picnic')
                fig29.update_layout(
                    plot_bgcolor = 'grey',
                    paper_bgcolor = 'grey'                    )
                
                fig29.update_coloraxes(showscale=False)
                st.plotly_chart(fig29)
            
            
            
            
        else:
            st.write('No data uploaded')
        
            







        
        






