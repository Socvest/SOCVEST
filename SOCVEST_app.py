import os
import pandas as pd 
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import seaborn as sns
import plotly.express as px
# Prevent error showing up
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objects as go
from textwrap import wrap
from datetime import datetime, date
import matplotlib as mpl
import urllib
# for min_max scaling
from sklearn.preprocessing import MinMaxScaler
from streamlit.report_thread import get_report_ctx
from PIL import Image

st.set_page_config(page_title='SOCVEST', page_icon =Image.open("Data/Front page/page icon.png"))
#st.set_page_config(page_title='SOCVEST') #,layout="wide")
import datetime
import calendar

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

analytics_js = """
    <!-- Default Statcounter code for SOCVEST
            https://share.streamlit.io/socvest/socvest/main/SOCVEST_app.py -->
            <script type="text/javascript">
            var sc_project=12548058; 
            var sc_invisible=1; 
            var sc_security="baba5af9"; 
            </script>
            <script type="text/javascript"
            src="https://www.statcounter.com/counter/counter.js" async></script>
            <noscript><div class="statcounter"><a title="Web Analytics"
            href="https://statcounter.com/" target="_blank"><img class="statcounter"
            src="https://c.statcounter.com/12548058/0/baba5af9/1/" alt="Web
            Analytics"></a></div></noscript>
            <!-- End of Statcounter Code -->
    """


# DATA MANIPULATION - UNIVERSAL
def reshape_data(data: pd.DataFrame):
    date_cols_regex = "^(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])[- /.]((19|20)\\d\\d)$"
    value_vars = list(data.filter(regex=(date_cols_regex)).columns)
    
    data_unpivoted = pd.melt(data, id_vars=['Country', 'Category', 'Series', 'Data'], 
        value_vars=value_vars, var_name='Date', value_name='Value')
    data_unpivoted['Date']=pd.to_datetime(data_unpivoted['Date'], infer_datetime_format=True)
    data_unpivoted['Value']=data_unpivoted['Value'].astype(float)
    #data_unpivoted = data_unpivoted.sort_values(['Country', 'Category', 'Series', 'Data','Date'])
    #data_unpivoted.loc[:,'Value'].fillna(0, inplace = True)
    return data_unpivoted

@st.cache(persist=True, allow_output_mutation=True)
# COVID_19 DATA
def COVID_19_data():
    data = pd.read_csv('Data/COVID-19/COVID data13.csv', index_col='Date',  parse_dates=True,infer_datetime_format=True) 
    return data

#@st.cache(persist=True)
# Choose data for datasets
def Filter_COVID_Timeseries_Data(Data_filtering):
    # GET ALL COVID-19 DATA
    data = COVID_19_data()
    #data = reshape_data(data)   
    
    # date column sorting out
    ##, format = '%Y-%m-%d') #.dt.date

    # GET COUNTRY DATA FOR THE COVID-19 THEME DATABASE
    # COUNTRY - get all values from the countries column
    countries = data.Geography.unique()
    #countries = countries.unique() #drop_duplicates(False)
    # first data filtering choice by country
    Country_choice = Data_filtering[0].selectbox("Geography", countries)
    # CATEGORY - get all row values in the category column that are in the country column
    category = data['Category'].loc[data['Geography'] == Country_choice].unique()  
    Category_choice = Data_filtering[1].selectbox("Category", category)
    # SERIES - get all series row values that are in the category column
    series = data.Series.loc[data['Category']==Category_choice].unique()
    Series_choice = Data_filtering[2].radio('Sequential Data type', series)
    
    if Country_choice == 'United Kingdom':
    
        if Category_choice == 'Social':

            data = data[['Geography', 'Category', 'Series', 'Data Type', 'Data', 'Values']]

             # Data type     
            data_type = data['Data Type'].loc[data['Category']==(Category_choice)].unique()
            Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)

            data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Data Type']==Data_type_choice)].unique())

            Trans_data=data.pivot_table(index='Date', columns='Data', values='Values').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)  

        # Category Choices
        if Category_choice == 'Spending':

            data = data[['Geography', 'Category', 'Series', 'Data Type', 'Data', 'Values']]

            # Data type     
            data_type = data['Data Type'].loc[data['Category']==(Category_choice)].unique()
            Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)

            data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Data Type']==Data_type_choice)].unique())

            Trans_data=data.pivot_table(index='Date', columns='Data', values='Values').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)  

        # Category Choices
        elif Category_choice == 'Transport':

            data = data[['Geography', 'Category', 'Series', 'Data Type', 'Data', 'Values']]

            # Data type     
            data_type = data['Data Type'].loc[data['Category']==(Category_choice)].unique()
            Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)

            data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Data Type']==Data_type_choice)].unique())

            Trans_data=data.pivot_table(index='Date', columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)      

        elif Category_choice == 'Mobility':

            data = data[['Country', 'Category', 'Series', 'Data Type', 'Regional Data', 'Data', 'Values']]

            # Data type     
            data_type = data['Data Type'].loc[data['Category']==(Category_choice)].unique()
            Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)

            #data = data.set_index('Date')

            regional_data = data['Regional Data'].loc[data['Category']==Category_choice].unique()
            Regional_data_choice = Data_filtering[1].selectbox("Regional Data Option", options=regional_data)

            data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Data Type']==Data_type_choice) & (data['Regional Data']==Regional_data_choice)].unique())

            Trans_data=data.pivot_table(index='Date', columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)

        elif Category_choice == 'Business impact and conditions survey':

            data = data[['Geography', 'Category', 'Series', 'Data Type', 'Regional Data',  'Survey Topic', 'Data', 'Values']]

            #regional_data = data['Regional Data'].loc[data['Category']==Category_choice].unique()
            #Regional_data_choice = Data_filtering[1].selectbox("Regional Data Option", options=regional_data)

            survey_type = data['Survey Topic'].loc[data['Category']==Category_choice].unique()
            Survey_choice = Data_filtering[0].selectbox("Survey Topic Option", options=survey_type)

            data_type = data['Data Type'].loc[data['Survey Topic']==(Survey_choice)].unique()
            Data_type_choice = Data_filtering[1].selectbox("Data Type", data_type)

            data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Data Type']==Data_type_choice) & (data['Survey Topic']==Survey_choice)].unique())

            Trans_data=data.pivot_table(index='Date', columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)

        elif Category_choice == 'Job Search Adverts':

            data = data[['Geography', 'Category', 'Series', 'Data Type', 'Regional Data', 'Data', 'Values']]

            regional_data = data['Regional Data'].loc[data['Category']==Category_choice].unique() #loc[data['Data Type']==Data_type_choice].unique()
            Regional_data_choice = Data_filtering[0].selectbox("Regional Data Option", options=regional_data)

            data_type = data['Data Type'].loc[data['Category']==Category_choice].unique()
            Data_type_choice = Data_filtering[1].selectbox("Data Type", data_type)


            data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Regional Data']==Regional_data_choice) & (data['Data Type'].isin(data_type))].unique())

            Trans_data=data.pivot_table(index='Date', columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)

        elif Category_choice == 'Economy':

            data = data[['Geography', 'Category', 'Series', 'Economic segment', 'Inflation Type', 'Inflation sub segment', 'Data Type', 'Data', 'Values']] 

            # type of economic data
            economic_segment = data['Economic segment'].loc[data['Category']==Category_choice].unique()
            economic_segment_choice = Data_filtering[0].selectbox("Economic Segment", economic_segment)

            # If inflation is selected
            if economic_segment_choice == 'Inflation':

                # inflation type
                inflation_type = data['Inflation Type'].loc[data['Economic segment']==economic_segment_choice].unique()
                inflation_type_choice = Data_filtering[1].selectbox("Inflation Type", inflation_type)   

                # data type
                data_type = data['Data Type'].loc[data['Inflation Type']==inflation_type_choice].unique()
                Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)           

                if inflation_type_choice == 'CPIH segments':

                    inflation_sub_segment = data['Inflation sub segment'].loc[data['Inflation Type']==inflation_type_choice].unique()
                    inflation_sub_segment_choice = Data_filtering[1].selectbox("Inflation Sub-Segment", inflation_sub_segment)



                data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Economic segment']==economic_segment_choice) & (data['Inflation Type']==(inflation_type_choice)) & (data['Data Type']==(Data_type_choice))].unique())

                Trans_data=data.pivot_table(index='Date', columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)

            elif economic_segment_choice == 'Online weekly price changes for food & drink items':

                     # inflation type                
                inflation_type = data['Inflation Type'].loc[data['Economic segment']==economic_segment_choice].unique()
                inflation_type_choice = Data_filtering[1].selectbox("Inflation Type", inflation_type)   

                # data type
                data_type = data['Data Type'].loc[data['Inflation Type']==inflation_type_choice].unique()
                Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)      

                data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Economic segment']==economic_segment_choice) & (data['Inflation Type']==(inflation_type_choice)) & (data['Data Type']==(Data_type_choice))].unique())

                Trans_data=data.pivot_table(index='Date', columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)  
                






            #elif Category_choice == 'Job Search Adverts':


           # elif Category_choice == 'Social':

            #    data = data[['Country', 'Category', 'Series', 'Data Type', 'Data', 'Values', 'Date']]

            else:


                # Data type     
                #data_type = data['Data Type'].loc[data['Series']==(Series_choice)].unique()
                #Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)

                # Prepare the dataframe that will be used for other types of analysis
                # Data filteration function - pass data into the function. Filter the data column according to the above choices (First set of choices)
                data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice)].unique()) # & (data['Data Type']==Data_type_choice)].unique()




                # the data to select from the dataframe - we want to select the values in the data column based on what we selected in the select data 
                # Create a new table making columns from the data columns. Use pivot table because if we specify the value, it won't aggregate by mean or some other statistic method. 
                Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0) 
    
    
    
    elif Country_choice == 'European Union':
        
        data = data[['Geography', 'Category', 'Series', 'Economic segment', 'Inflation Type', 'Inflation sub segment', 'Government finance segment','Household segment', 'European Union segment', 'Data Type', 'Data', 'Values']]
            
        if Category_choice == 'The Economy':
                        
            # type of economic data
            economic_segment = data['Economic segment'].loc[data['Category'] == Category_choice].unique()
            economic_segment_choice = Data_filtering[0].selectbox("Economic Segment", economic_segment)
            
            if economic_segment_choice == 'Government Finance':
                
                data = data[['Geography', 'Category', 'Series', 'Economic segment', 'Government finance segment', 'Data Type', 'Data', 'Values']]
                
                # display Government finance segment 
                gov_finance_segment = data['Government finance segment'].loc[data['Economic segment'] == economic_segment_choice].unique()
                Gov_fin_choice = Data_filtering[1].selectbox("Government finance Segment", options=gov_finance_segment)
                
                # data type
                data_type = data['Data Type'].loc[data['Government finance segment']==Gov_fin_choice].unique()
                Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)  
                
                
                # Data filteration function - pass data into the function. Filter the data column according to the above choices (First set of choices)
                data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Economic segment']==economic_segment_choice) & (data['Government finance segment']==Gov_fin_choice) & (data['Data Type']==Data_type_choice)].unique())
                
                Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)
                
            if economic_segment_choice == 'Household accounts':
                
                data = data[['Geography', 'Category', 'Series', 'Economic segment', 'Household segment', 'European Union segment', 'Data Type', 'Data', 'Values']]
                
                # display Housing segment 
                Housing_segment = data['Household segment'].loc[data['Economic segment'] == economic_segment_choice].unique()
                housing_choice = Data_filtering[0].selectbox("Housing Accounts Segment", options=Housing_segment)
                
                # data type
                data_type = data['Data Type'].loc[data['Household segment']==housing_choice].unique()
                Data_type_choice = Data_filtering[1].selectbox("Data Type", data_type)  
                
                
                # Data filteration function - pass data into the function. Filter the data column according to the above choices (First set of choices)
                data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Economic segment']==economic_segment_choice) & (data['Household segment']==housing_choice) & (data['Data Type']==Data_type_choice)].unique())
                
                Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)
                
                if housing_choice == 'Household real income and consumption':
                    
                    data = data[['Geography', 'Category', 'Series', 'Economic segment', 'Household segment', 'European Union segment', 'Data Type', 'Data', 'Values']]
                    
                    # display Housing segment 
                    EU_segment = data['European Union segment'].loc[data['Household segment']==housing_choice].unique()
                    EU_segment_choice = Data_filtering[1].selectbox("European Union Segment", options=EU_segment)
                    
                    # data type
                    data_type = data['Data Type'].loc[data['European Union segment']==EU_segment_choice].unique()
                    Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type, key=1)  
                    
                    # Data filteration function - pass data into the function. Filter the data column according to the above choices (First set of choices)
                    data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Economic segment']==economic_segment_choice) & (data['Household segment']==housing_choice) & (data['European Union segment']==EU_segment_choice) & (data['Data Type']==Data_type_choice)].unique())

                    Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0)
                    
            if economic_segment_choice == 'Inflation data':

                data = data[['Geography', 'Category', 'Series', 'Economic segment', 'Inflation Type', 'Data Type', 'Data', 'Values']] 

                inflation_type_seg = data['Inflation Type'].loc[data['Economic segment']==economic_segment_choice].unique()
                inflation_type_choice = Data_filtering[1].selectbox("Inflation Type", options=inflation_type_seg)

                 # data type
                data_type = data['Data Type'].loc[data['Inflation Type']==inflation_type_choice].unique()
                Data_type_choice = Data_filtering[0].selectbox("Data Type", data_type)  
                
                # Data filteration function - pass data into the function. Filter the data column according to the above choices (First set of choices)
                data_col = list(data['Data'][(data['Geography']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice) & (data['Economic segment']==economic_segment_choice) & (data['Inflation Type']==inflation_type_choice) & (data['Data Type']==Data_type_choice)].unique())

                Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1)#.reindex(data['Date'].unique(), axis=0) 
    
    
    # data 2
    #data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    # Create a new table making columns from the data columns. Use pivot table because if we specify the value, it won't aggregate by mean or some other statistic method. 
   # Trans_data2=data.pivot_table(index=['Date'], columns='Data', values='Values', aggfunc='first').rename_axis(None, axis=1) #.reindex(data['Date'].unique(), axis=0) 
    # return the whole function
    return list(data_col), Trans_data, data_type #, Trans_data2


# Visualisation functions
def Line_Area_chart(Dataframe_to_display, Columns_to_show):
    plotly_fig_area = px.area(data_frame=Dataframe_to_display,x=Dataframe_to_display.index,y=Columns_to_show)

    # Legend settings
    plotly_fig_area.update_layout(showlegend=False)        
    plotly_fig_area.update_layout(margin_autoexpand=True) # prevent size from changing because of legend or anything
    plotly_fig_area.update_traces(mode="lines", hovertemplate=None)
    plotly_fig_area.update_layout(hovermode="x unified")
    plotly_fig_area.update_traces(connectgaps=True)

    plotly_fig_area.update_layout(legend=dict(
                      orientation = "h",
                      yanchor="bottom",
                      y=-.85,
                      xanchor="right",
                      x=1.0
                    )) 


    height = 690
    plotly_fig_area.update_layout(
    autosize=False,
    width=970,
    height=height)
    
     # Date range
    plotly_fig_area.update_xaxes(rangeselector=dict(
                                buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                                ])), rangeslider_visible=True)

    # Background colour
    #plotly_fig.update_layout(paper_bgcolor="white") # Change background of the 'page' not the graph
    plotly_fig_area.layout.plot_bgcolor="white"
    # area colour
    
    
    return st.plotly_chart(plotly_fig_area,use_container_width=True)


# Line chart using the filtered data
def Line_Chart(Dataframe_to_display, Columns_to_show):    
    
    plotly_fig = px.line(data_frame=Dataframe_to_display,x=Dataframe_to_display.index,y=Columns_to_show)
                                            #width=780, height=830) # Get data from the dataframe with selected columns, choose the x axis as the index of the dataframe, y axis is the                                             data that will be multiselected
    # Date graph


    # Legend settings
    plotly_fig.update_layout(showlegend=False)        
    plotly_fig.update_layout(margin_autoexpand=True) # prevent size from changing because of legend or anything
    #plotly_fig.update_traces(mode="lines", hovertemplate=None)
    plotly_fig.update_layout(hovermode="x unified")
    plotly_fig.update_traces(connectgaps=True)

    plotly_fig.update_layout(legend=dict(
                      orientation = "h",
                      yanchor="bottom",
                      y=-.75,
                      xanchor="right",
                      x=1.0
                    )) 


    height = 690
    plotly_fig.update_layout(
    autosize=False,
    width=910,
    height=height)

    # Date range
    plotly_fig.update_xaxes(rangeselector=dict(
                                buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                                ])), rangeslider_visible=True)



    #grids
    plotly_fig.update_xaxes(showgrid=False)  # Removes X-axis grid lines
    plotly_fig.update_yaxes(showgrid=False)  # Removes Y-axis grid lines

    # Background colour
    # plotly_fig.update_layout(paper_bgcolor="white") # Change background of the 'page' not the graph
    plotly_fig.layout.plot_bgcolor="white"
    
    
    return st.plotly_chart(plotly_fig,use_container_width=True)

# Visualisation functions
def Line_Area_chart_single_view(Dataframe_to_display, Columns_to_show):
    
    titles = '<br>'.join(wrap(Columns_to_show, 90))
    
    plotly_fig_area = px.area(data_frame=Dataframe_to_display,x=Dataframe_to_display.index,y=Dataframe_to_display, title=titles) #Columns_to_show)

    # Legend settings
    plotly_fig_area.update_layout(showlegend=False)        
    plotly_fig_area.update_layout(margin_autoexpand=True) # prevent size from changing because of legend or anything
    plotly_fig_area.update_traces(mode="lines", hovertemplate=None)
    plotly_fig_area.update_layout(hovermode="x unified")
    plotly_fig_area.update_traces(connectgaps=True)
    plotly_fig_area.update_layout(title_font_family="Times New Roman")
    plotly_fig_area.update_layout(title_font_size=17)
    plotly_fig_area.update_layout(
    title={
        'text': titles,
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    plotly_fig_area.update_layout(legend=dict(
                      orientation = "h",
                      yanchor="bottom",
                      y=-.85,
                      xanchor="right",
                      x=1.0
                    )) 


    height = 690
    plotly_fig_area.update_layout(
    autosize=False,
    width=970,
    height=height)
    
     # Date range
    plotly_fig_area.update_xaxes(rangeselector=dict(
                                buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                                ])), rangeslider_visible=True)

    # Background colour
    #plotly_fig.update_layout(paper_bgcolor="white") # Change background of the 'page' not the graph
    plotly_fig_area.layout.plot_bgcolor="white"
    # area colour
    
    
    return st.plotly_chart(plotly_fig_area,use_container_width=True)


# Line chart using the filtered data
def Line_Chart_single_view(Dataframe_to_display, Columns_to_show):  
    ### - https://community.plotly.com/t/how-to-break-time-series-lines-at-data-gaps/645 (fix gaps plots for line charts)
    
    
    titles = '<br>'.join(wrap(Columns_to_show, 90)) #for l in Columns_to_show]
    
    #wrapped_labels = [ Columns_to_show.replace(' ', '\n')] # for label in Columns_to_show ]
    
    #st.write(wrapped_labels)
    
    #st.write(Columns_to_show)
    
    #st.markdown("<h1 style='text-align: center; color: red;'>"Columns_to_show</h1>", unsafe_allow_html=True)
    
    plotly_fig = px.line(data_frame=Dataframe_to_display,x=Dataframe_to_display.index,y=Dataframe_to_display, title=titles)
                                            #width=780, height=830) # Get data from the dataframe with selected columns, choose the x axis as the index of the dataframe, y axis is the                                             data that will be multiselected
    # Date graph


    # Legend settings
    plotly_fig.update_layout(showlegend=False)        
    plotly_fig.update_layout(margin_autoexpand=True) # prevent size from changing because of legend or anything
    #plotly_fig.update_traces(mode="lines", hovertemplate=None)
    plotly_fig.update_layout(hovermode="x unified")
    plotly_fig.update_traces(connectgaps=True)
    plotly_fig.update_layout(title_font_family="Times New Roman")
    plotly_fig.update_layout(title_font_size=17)
    plotly_fig.update_layout(
    title={
        'text': titles,
        'y':0.94,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

    ##plotly_fig.update_layout(legend=dict(
      #                orientation = "h",
       #               yanchor="bottom",
        #              y=-.75,
         #             xanchor="right",
          #            x=1.0
           #         )) 


    height = 690
    plotly_fig.update_layout(
    autosize=False,
    width=910,
    height=height)

    # Date range
    plotly_fig.update_xaxes(rangeselector=dict(
                                buttons=list([
                                dict(count=7, label="7d", step="day", stepmode="backward"),
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                                ])), rangeslider_visible=True)
    

    #grids
    plotly_fig.update_xaxes(showgrid=False)  # Removes X-axis grid lines
    plotly_fig.update_yaxes(showgrid=False)  # Removes Y-axis grid lines

    # Background colour
    # plotly_fig.update_layout(paper_bgcolor="white") # Change background of the 'page' not the graph
    plotly_fig.layout.plot_bgcolor="white"
    
    
    #fig3 = go.Figure(data=plotly_fig.data, connectgaps=True)
    
    #connectgaps=True
    
    return st.plotly_chart(plotly_fig,use_container_width=True)


@st.cache(persist=True)
def Heatmap_Timeseries_data_prep(data_to_analyse, Heatmap_dataframe_timeseries):
    # Extract day, month and year from dataframe table. First reset index
    Data_to_select_no_index = Heatmap_dataframe_timeseries.reset_index()

    
    # Create new dataframes from the data column
    Data_to_select_no_index['Year'] = Data_to_select_no_index.Date.dt.year
    Data_to_select_no_index['Month'] = Data_to_select_no_index.Date.dt.month
    Data_to_select_no_index['Month'] = Data_to_select_no_index['Month'].apply(lambda x: calendar.month_abbr[x])
    Data_to_select_no_index['Day'] = Data_to_select_no_index.Date.dt.day
    # set new indexes
    Data_to_select_no_index.set_index(['Day','Month', 'Year'], inplace=True)

    # Indexed data label
    Data_to_select_indexed = Data_to_select_no_index

    #Data_to_select_indexed = Data_to_select_indexed.unstack(level=0)
    return Data_to_select_indexed

# Heatmap graph
def Heatmap_timeseries_index(data_to_analyse, Heatmap_dataframe_timeseries): 
    Data_to_select_indexed = Heatmap_Timeseries_data_prep(data_to_analyse, Heatmap_dataframe_timeseries)
    axis_data = Data_to_select_indexed.index.names
    return axis_data

def Heatmap_chart(Data_to_select_indexed, data_to_analyse, y_axis, x_axis, colour, annot):
    title = st.columns([2,2])
    # Pivot table
    Pivot_data = Data_to_select_indexed.pivot_table(index=y_axis ,columns=x_axis, values=data_to_analyse).fillna(0)
    f, ax2 = plt.subplots(figsize=(16, 11))
    # title 
    ax2.set_title(data_to_analyse, fontsize=12)
    #plt.suptitle(data_to_analyse)
    # show data
    ax2 = sns.heatmap(Pivot_data, cmap=colour, annot=annot, fmt='.2f', vmin= 0.0) #, vmax=1.9) #.set_title(data_to_analyse, fontsize=15)
    
    return st.pyplot(use_container_width=True)


# Sidebar titles

#pages available
st.sidebar.title("SOCVEST")

pages = ["About us", "Data and Analysis", "Feedback"]

# About us page
choice_1 = st.sidebar.selectbox("Menu", pages)

if choice_1 =="About us":
   
    components.html(analytics_js, width=200, height=200) 
            
    title1,title2,title3 = st.columns(3)
    title2.title("SOCVEST")
    banner_picture = st.columns(1)
    with banner_picture[0]:
        st.image("Data/Front page/Socvest.jpg", caption='Enabling the Data Driven Capabilities of the Retail Trader')
    
    # st.title("SOCVEST")
    themes1 = st.columns([1,3,1])
    
    #themes1[1].text("Enabling the Data Driven Capabilities of the Retail Trader")
    #themes2.write("A partner to the retail investor")
    #themes3.write("Supporter of the retail investor")
    
    st.header("Who are we?")
    st.write("We are an alternative data provider for the retail investor.")
    st.header("Why do we exist?")
    st.write("Retail investors have always relied on alternative means of deriving investment thesis and insights. Limited industry transparency and costly service providers tailored to professional investment companies meant retail traders were always left behind. Things are changing. There are masses of data and the technology that can make deriving data driven insights into the world, markets and to construct investment opportunities are available at relatively low cost. We aim to further enhance the retail trader's data driven decision making capabilities.")
    
    st.write("This platform is here to serve you, the new age investor.")
    
    st.header("What do we do?")
    st.write("We provide alternative data for world and investment themes to support the retail trader in their investment decision making. We also provide the capacity to run machine learning models for research and predictive probability purposes.")
      
    # labels to indicate we do not provide investment adviceh
    st.success("We do not provide investment and financial advice. Data and capabilities provided on this platform is not financial advice")    
    
elif choice_1 == "Data and Analysis":   
    Data_choices = st.sidebar.radio("Choose dataset", ['COVID-19', 'Economy', 'Geopolitics','Country','Financial Markets'])
    st.title("Data Exploration")
    st.write("Select a dataset and the range of data visualisation tools available to build your analysis. Once done, uncheck the 'Hide Data' below to view the data.")
    if Data_choices == 'COVID-19':
        components.html(analytics_js, width=200, height=200) 
        # create title based on choice            
        title = 'COVID-19'
        st.subheader(title)
        # Create expander to view/choose data
        with st.expander("Choose Data"):
            st.write("Choose data unique to a country for this theme.")
            # Create labels to slot filtering sections
            Data_filtering = st.columns(3)
            # define data to be used for this section
            # Get the select boxes that will be used for filtering the data. Load the filtered data and the pivoted datatable
            data_col, Trans_data, data_type = Filter_COVID_Timeseries_Data(Data_filtering)
            
            # create new labels for hide data
            data_mix_buttons = st.columns([3,1,1])
            # We use this to hide or show the data
            Hide_data = data_mix_buttons[2].checkbox("Hide Data", value=True) # default
            #read_state('size',engine,session_id)

            if not Hide_data:
                # title to show they can select data
                DF_Display = st.subheader("View Data Table")
                # Create a multiselect to choose the columns based on the filtered data
                Select_cols_dataframe = st.multiselect("Choose data to Display", options=data_col)# distinct selectbox
                #read_state('Column selection', engine, session_id)
                #write_state('Select_cols_dataframe',Select_cols_dataframe, engine, session_id)
                #Select_cols_dataframe = read_state('Select_cols_dataframe',engine,session_id)
                
                Data_to_select= Trans_data[Select_cols_dataframe].fillna("") # dataframe to select from
                #write_state_df(Data_to_select,engine,session_id + '_df')
                        
                Data_to_select.index = Data_to_select.index.date
                
                if not Select_cols_dataframe:
                    st.error("Please select at least one column.")
                else:
                    st.write(Data_to_select) #.style.set_precision(2))                    
        # CHARTS
        st.sidebar.title("Visualisation")
        # Show charts for selected data
        Visualisation_segment = st.sidebar.expander("Choose Visualisation Method")
        # The opportunity to select visualisation type
        # Label timeseries
        Visualisation_segment.subheader("Timeseries")
        
        # LINE CHART
        line_chart = Visualisation_segment.checkbox("Line Chart")
        if line_chart:
            # subtitle
            st.subheader("Line Chart Visualisation")
            show_line_chart = st.expander("Show Chart")
            # if the show line chart shows
            with show_line_chart:               
                                
                # new label for line chart options
                line_chart_options = st.columns(3)
                
                # Choose to show one data point per time or multiple at once to compare
                type_of_chart_to_view = line_chart_options[2].selectbox("Chart Type", options=['Compare Variables', 'Single Variable View'])
                
                if type_of_chart_to_view is not 'Single Variable View':
                    
                
                    # Create a multiselect to choose the columns based on the filtered data
                    Data_to_show_line = st.multiselect("Choose Data to Display", options=data_col) #, key='linechart')
                    Dataframe_to_display_line_chart = Trans_data[Data_to_show_line] # dataframe to select from
                    # button to switch between line and area charts
                    line_chart = line_chart_options[0].radio("Chart Options", ["Line Chart", "Area Chart"], index=0)
                    #area_chart = line_chart_options[1].radio("", ['Area Chart'])

                    # if we select line chart
                    if line_chart== "Line Chart":
                        try:
                            # if no data is showing, 
                            if not Data_to_show_line:
                                st.error("Please select at least one Data Point.")
                            else:                           
                                Line_Chart(Dataframe_to_display_line_chart, Data_to_show_line) 

                        except urllib.error.URLError as e:
                            st.error(
                                """
                                **This demo requires internet access.**

                                Connection error: %s
                            """
                                % e.reason
                            )

                    elif line_chart == "Area Chart":
                        try:
                            # if no data is showing, 
                            if not Data_to_show_line:
                                st.error("Please select at least one Data Point.")

                            else:                           
                                Line_Area_chart(Dataframe_to_display_line_chart, Data_to_show_line) 

                        except urllib.error.URLError as e:
                            st.error(
                                """
                                **This demo requires internet access.**

                                Connection error: %s
                            """
                                % e.reason

                            )
                else: 
                    Data_to_show_line = st.selectbox("Choose Data to Display", options=data_col, key='linechart')
                    
                    Dataframe_to_display_line_chart = Trans_data[Data_to_show_line] # dataframe to select from
                    # button to switch between line and area charts
                    line_chart = line_chart_options[0].radio("Chart Options", ["Line Chart", "Area Chart"], index=0)
                    #area_chart = line_chart_options[1].radio("", ['Area Chart'])

                    # if we select line chart
                    if line_chart== "Line Chart":
                        try:
                            # if no data is showing, 
                            if not Data_to_show_line:
                                st.error("Please select at least one Data Point.")
                            else:                           
                                Line_Chart_single_view(Dataframe_to_display_line_chart, Data_to_show_line) 

                        except urllib.error.URLError as e:
                            st.error(
                                """
                                **This demo requires internet access.**

                                Connection error: %s
                            """
                                % e.reason
                            )

                    elif line_chart == "Area Chart":
                        try:
                            # if no data is showing, 
                            if not Data_to_show_line:
                                st.error("Please select at least one Data Point.")

                            else:                           
                                Line_Area_chart_single_view(Dataframe_to_display_line_chart, Data_to_show_line) 

                        except urllib.error.URLError as e:
                            st.error(
                                """
                                **This demo requires internet access.**

                                Connection error: %s
                            """
                                % e.reason

                            )
                    

        heatmap_timeseries = Visualisation_segment.checkbox("Heatmap") 
        
        if heatmap_timeseries:
        
            st.subheader("Timeseries Heatmap")

            Show_heatmap_times = st.expander("Show Chart")
            with Show_heatmap_times:  
            
                # Data to select
                data_to_analyse = st.selectbox("Choose data", options=data_col)
                # Dataframe to choose data from
                Heatmap_dataframe_timeseries = Trans_data[data_to_analyse]
                
                # new label for line chart options
                Axis_data = st.columns([3,3,3,3])   
                
                Data_to_select_indexed = Heatmap_Timeseries_data_prep(data_to_analyse, Heatmap_dataframe_timeseries) 
                
                #st.write(Data_to_select_indexed.Date.dt.day)
                
                axis_data = Heatmap_timeseries_index(data_to_analyse, Heatmap_dataframe_timeseries)
                
                colours = ['Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
                annotations = [True,False]
                
                # Indexed datasets
                x_axis = Axis_data[0].selectbox("Choose x axis", options=axis_data, index=2)
                y_axis = Axis_data[1].selectbox("Choose y axis", options=axis_data, index=1)
                colour = Axis_data[2].selectbox("Choose colour", options=colours, index=2)
                annot = Axis_data[3].radio("Show figures", options=annotations, index=1) #annotations[0]) #False)
                
                # if columns not selected...
                if not data_to_analyse:
                    st.error("Please select at least one column.")
                else:
                    Heatmap_chart(Data_to_select_indexed, data_to_analyse, y_axis, x_axis, colour, annot)
        
        bar_chart = Visualisation_segment.checkbox("Bar Chart") 
        if bar_chart == True:
            st.subheader("Bar Chart")
            
            bar_chart_section = st.expander("Show Chart")
            
            with bar_chart_section:
                # Data to select
                data_to_analyse = st.multiselect("Choose data", options=data_col)
                # Dataframe to choose data from
                Heatmap_dataframe_timeseries = Trans_data[data_to_analyse]

                if not data_to_analyse:
                    st.error("Please insert at least one Data Point")
                else:
                    st.bar_chart(data=Heatmap_dataframe_timeseries, width=480, height=450, use_container_width=True)
        
        # Data Relationship
        Subheader_data_relationship = Visualisation_segment.subheader("Data Relationships")
        
        # Categorical data (Scatter graph, correlation heatmap)             
        charts_relationship = Visualisation_segment.checkbox("Scatter Chart")
        # if the button is selected
        if charts_relationship==True:  
            # Title indicating this is a timeseries
            st.subheader("Scatter Chart")
            
            Relationship_charts = st.expander("Show Charts")
            with Relationship_charts: 
                 
                st.write("*Note that data has been scaled to allow for better ploting outcomes")
                # define data that will be used for analysis (the whole dataset)
                scaling = MinMaxScaler()
                Scatter_data_ = Trans_data.fillna(0)
                
                # Scale data
                scaled_dataframe= scaling.fit_transform(Scatter_data_)
                # create new dataframe with scaled data
                scaled_dataframe = pd.DataFrame(scaled_dataframe, index=Trans_data.index, columns=Trans_data.columns)
                
                Scatter_graph_data = scaled_dataframe[data_col]
                
                type_chart = st.radio("Type of chart", options= ['Advanced Chart', 'Simple Chart'], index=1)
                
                if type_chart == 'Simple Chart':
                    
                    axes_options = st.columns(2)
                    chart_d = st.columns(1)
                    
                    # select x axis
                    select_box_X = axes_options[0].selectbox('Select x-axis', data_col)
                    #select y-axis
                    select_box_Y = axes_options[1].selectbox('Select y-axis', data_col)      
                    
                    
                    X_boundary = Scatter_graph_data[select_box_X]
                    Y_boundary = Scatter_graph_data[select_box_Y]
                    
                    #test_x = test1.min()
                    
                    X_boundary_min = X_boundary[X_boundary > 0.00001].min() - 0.005 
                    Y_boundary_min = Y_boundary[Y_boundary > 0.00001].min() - 0.005
                    X_boundary_max = X_boundary.max() + 0.004
                    Y_boundary_max = Y_boundary.max() + 0.004
                    
                    # Y_boundary_min, X_boundary_min
                    
                    X_boundary_min = round(X_boundary_min,4)
                    Y_boundary_min = round(Y_boundary_min,4)
                    X_boundary_max = round(X_boundary_max,3)
                    Y_boundary_max = round(Y_boundary_max,3)
                    
                    test = px.scatter(Scatter_graph_data, x=select_box_X, y=select_box_Y, height=650, width=900)
                    test.update_layout(yaxis_range=[Y_boundary_min, Y_boundary_max],
                                       xaxis_range=[X_boundary_min, X_boundary_max])
                    st.plotly_chart(test, use_container_width=True)
                    
                else:
                    
                    
                    chart_options = st.columns(3)
                    axes_options = st.columns(2)
                    chart_d = st.columns(1)


                    # Create an empty space to replace contents into
                    #select x-axis
                    select_box_X = axes_options[0].selectbox('Select x-axis', data_col, key="x-axis")
                    #select y-axis
                    select_box_Y = axes_options[1].selectbox('Select y-axis', data_col, key="y-axis") 

                    # Control dimensions of chart
                    color = ['navy', 'blue', 'green', 'brown', 'skyblue', 'grey', 'black', 'cornsilk'] 
                    color = chart_options[0].selectbox('Choose chart color', options=color, key="colour")
                    # Edge colors
                    edge_col = ['navy','red', 'blue']
                    edge_col_opt = chart_options[1].selectbox('Choose edge plots colour', options=edge_col)
                    # Height
                    height = chart_options[0].slider('Size of chart', min_value=0.0, max_value=30.0, step=.1, value=11.5, key="height")
                    # Ratio
                    #ratio_chart = chart_options[1].slider('Ratio', min_value=0, max_value=30, step=1, value=7, key=2)
                    # Space
                    space = chart_options[1].slider('Distance main/edge plot', min_value=0.0, max_value=1.0, step=.1, value=.4, key="space")
                    
                    bkc = ['darkgrid', 'whitegrid', 'dark', 'white']
                     # graph background colour
                    Background_colr = chart_options[2].selectbox('Choose Background Colour', options=bkc)
                    # Show cuts/shapes of plots
                    alpha = chart_options[2].slider('Plot density', min_value=0.0, max_value=1.0,  step=.1, value=.4, key="alpha") 
                    # Default settings
                    #default_options = chart_options[3].button('Default', key=1)
                    
                    X_boundary = Scatter_graph_data[select_box_X]
                    Y_boundary = Scatter_graph_data[select_box_Y]
                    
                    #test_x = test1.min()
                    
                    X_boundary_min = X_boundary[X_boundary > 0.00001].min() - 0.005 
                    Y_boundary_min = Y_boundary[Y_boundary > 0.00001].min() - 0.005
                    X_boundary_max = X_boundary.max() + 0.004
                    Y_boundary_max = Y_boundary.max() + 0.004
                    
                    # Y_boundary_min, X_boundary_min
                    
                    X_boundary_min = round(X_boundary_min,4)
                    Y_boundary_min = round(Y_boundary_min,4)
                    X_boundary_max = round(X_boundary_max,3)
                    Y_boundary_max = round(Y_boundary_max,3)


                    #ticks for margin plots
                    #margin_grid = chart_options[2].radio('Show grid for edge plots', ['True', 'False'])

                    sns.set_style(Background_colr)
                    
                    chart = sns.jointplot(x=select_box_X, 
                                             y=select_box_Y, 
                                              data = Scatter_graph_data,
                                              alpha=alpha,
                                              height=height,  
                                              space=space, 
                                              xlim=(X_boundary_min,X_boundary_max), 
                                              ylim=(Y_boundary_min,Y_boundary_max),
                                              color=color,
                                              marginal_kws={'color':edge_col_opt}) #, color=color) # hue=select_box_hue, s=select_box_size,  marginal_ticks=True)

                    st.pyplot(chart, use_container_width=True)     
                
                #Advanced_plots = st.columns([4,4,3,3,5])

                # Correlation
        Correlation = Visualisation_segment.checkbox("Correlation")
        if Correlation:
            # subtitle
            st.subheader("Correlation Chart ")
            show_scatter_chart = st.expander("Show Chart")
            #Show_corr = st.checkbox("Show Correlation Diagram")
            with show_scatter_chart:
                
                # labels
                Chart_options_heatmap = st.columns(4) 
    
                # Colours
                colours = ['Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
                Heatmap_corr_colours = Chart_options_heatmap[1].selectbox("Choose Colours", options = colours, index=1) #, help='Colour density is based on values. The higher the values, the darker the colours')
                # Font size
                Heatmap_fontsize = Chart_options_heatmap[2].number_input("Select Font Size of figures", min_value=5, max_value=100, value=12, step=1) 
                # Font weight
                f_weight = ['normal','bold','heavy','light','ultrabold','ultralight']
                Font_weight = Chart_options_heatmap[3].selectbox("Font format", options=f_weight)
                
                # Line width between plots
                line_width = Chart_options_heatmap[1].slider("Choose width between cells", min_value=0.0, max_value=5.0, value=0.3, step=0.1)
                 # Option to annotate
                choice = [True, False]
                Annotate = Chart_options_heatmap[0].selectbox("Show Correlation figures", options=choice)
                
                # Line colour
                line_colours_heatM = ['black', 'pink', 'white', 'grey', 'orange']
                line_colour = Chart_options_heatmap[0].selectbox("Choose line colour", options = line_colours_heatM)
                
                # size of chart
                Chart_height_Heatm_corr = Chart_options_heatmap[2].slider("Chart height", min_value=10, max_value=30, value=12, step=1)
                Chart_width_Heatm_corr = Chart_options_heatmap[3].slider("Chart width", min_value=10, max_value=30, value=12, step=1)
                
                # Choose data to view
                Data_to_select_for_view = st.multiselect("Choose Data", options=data_col)
                # Data to show on graph
                COVID_19_data_Corr = Trans_data[Data_to_select_for_view]

                #Show_all_button = st.button("Show all data") # use button at the right using column

                # if columns not selected...
                if not Data_to_select_for_view:
                    st.error("Please select at least one column.")

                else:

                    labels1 = COVID_19_data_Corr.columns

                    wrapped_labels_1 = ['\n'.join(wrap(l, 20)) for l in labels1]

                    mask = np.zeros_like(COVID_19_data_Corr.corr())
                    triangle_indices = np.triu_indices_from(mask)
                    mask[triangle_indices]= True


                    f, ax = plt.subplots(figsize=(Chart_height_Heatm_corr,Chart_width_Heatm_corr))    
                    ax = sns.heatmap(COVID_19_data_Corr.corr(), annot=Annotate, fmt='.2f', annot_kws={'fontsize':Heatmap_fontsize, 'fontweight':Font_weight}, linewidths=line_width,
                                     linecolor=line_colour, xticklabels=wrapped_labels_1, yticklabels = wrapped_labels_1, square=True, mask=mask, cmap=Heatmap_corr_colours, vmin=-1, vmax=1, center=0);

                    # set labels
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')

                    # customize labels
                    for label in ax.get_yticklabels():
                        label.set_size(9)
                        label.set_weight("bold")
                        label.set_color("black")

                    for label in ax.get_xticklabels():
                        label.set_size(9)
                        label.set_weight("bold")
                        label.set_color("black")

                        ## need to create feature to show just y labels or x, need to include option to show all data, 
                        ## show option to change colour

                    ax1 = st.write(ax)

                    st.pyplot()#use_container_width=True);
                    
        # Select Categorical data visualisation
        Subheader_data_relationship = Visualisation_segment.subheader("Frequency Distribution")
        
        # Histogram
        # Box plot
       
        Histogram_chart = Visualisation_segment.checkbox("Histogram")
        if Histogram_chart:
            # Histogram subtitle
            st.subheader("Histogram Chart Visualisation")
            show_Histogram_chart = st.expander("Show Chart")
            # if you choose Scatter Plot
            with show_Histogram_chart:
                
                # columns for flexible options
                Hist_options = st.columns(3)
                Hist_options1 = st.columns(4)
                data_choice = st.columns(1)
                
                bkc = ['darkgrid', 'whitegrid', 'dark', 'white']
                # graph background colour
                Background_col = Hist_options[1].selectbox('Choose Background Colour', options=bkc, key="hist")
                sns.set_style(Background_col)
                
                # KDE argument
                KDE_plot = Hist_options[0].selectbox("Show KDE Plot", options=(True, False))
                # Horizontal/vertical
                horizontal_vertical = Hist_options[2].radio("Choose Orientation", ['Horizontal','Vertical'], index=1)
                # Colour
                colour_hist = ['indianred', 'indigo']
                colour = Hist_options1[0].selectbox("Colour", options=colour_hist)
                # Statistics
                stats = ["count", "frequency", "density", "probability"]
                count = Hist_options1[1].selectbox('Choose stats to view', options=stats, index=2)
                # styling
                fill = Hist_options1[2].selectbox('Fill bar plots', options=(True, False))
                visual_stat = ['bars', 'step', 'poly']
                element = Hist_options1[3].selectbox('Visual of stat', options=visual_stat)
                cumulative = Hist_options1[0].selectbox('Show Cumulation data', options=(True,False))
                                
                # Size control
                height = Hist_options1[1].slider("Chart height", min_value=5, max_value=30, value=5, step=1)
                width = Hist_options1[2].slider("Chart width", min_value=5, max_value=30, value=8, step=1)
                histogram_slider = Hist_options1[3].slider(label="Number of bins to display", min_value=5, max_value= 30, value = 15)
            
                # select feature from select box
                histogram_data_selection = data_choice[0].selectbox("Choose feature to observe", options=data_col)

                # Data to view
                COVID_19_data_Hist = Trans_data[data_col]
            
                min_boundary_hist = Trans_data[histogram_data_selection]
                max_boundary_hist = Trans_data[histogram_data_selection]
                                       
                min_boundary_hist = min_boundary_hist[min_boundary_hist > 0.00001].min() - 0.005
                max_boundary_hist = max_boundary_hist.max() + 0.005
                
                min_boundary_hist = round(min_boundary_hist,4)
                max_boundary_hist = round(max_boundary_hist,4)
                
                if not horizontal_vertical == 'Vertical':
                    
                    f, ax = plt.subplots(figsize=(width,height)) 
                    sns.histplot(y=histogram_data_selection, data=COVID_19_data_Hist, kde=KDE_plot, stat=count, fill=fill, element=element, cumulative=cumulative, color=colour) #, bins = histogram_slider, binwidth=bin_width, binrange=(bin_range_1,bin_range_2))
                    ax.set_xlim(min_boundary_hist, max_boundary_hist)
                    st.pyplot()
                else:
                    f, ax = plt.subplots(figsize=(width,height)) 
                    sns.histplot(x=histogram_data_selection, data=COVID_19_data_Hist, kde=KDE_plot, stat=count, fill=fill, element=element, cumulative=cumulative, color=colour) #, bins = histogram_slider, binwidth=bin_width, binrange=(bin_range_1,bin_range_2))
                    ax.set_xlim(min_boundary_hist, max_boundary_hist)
                    st.pyplot()
                       

    components.html(analytics_js, width=200, height=200)   

 
                
            
    if Data_choices == 'Economy':
        components.html(analytics_js, width=200, height=200)  
            
        title = Data_choices = 'Economy'
        st.subheader(title)
        st.markdown("## Coming soon...")

    if Data_choices == 'Geopolitics':
        components.html(analytics_js, width=200, height=200)                
        
        title = Data_choices = 'Geopolitics'
        st.subheader(title)
        st.markdown("## Coming soon...")

    if Data_choices == 'Country':
        components.html(analytics_js, width=200, height=200)
        title = Data_choices = 'Country'
        st.subheader(title)
        st.markdown("## Coming soon...")
        
        
    if Data_choices == 'Financial Markets':
        components.html(analytics_js, width=200, height=200)
            
        title = Data_choices = 'Financial Markets'
        st.subheader(title)
        st.markdown("## Coming soon...")

elif choice_1 =="Feedback":
    components.html(analytics_js, width=200, height=200) 
    
    st.subheader("Feedback")
    
    st.write("We value your input and are always striving to improve the app to tailor to your needs. Help us help you!")
    
    survey = """
    <iframe src="https://docs.google.com/forms/d/e/1FAIpQLSex4XONiQinqRHPlt1Ak01--8BCAgVVTxfoUQw1tTvyJZ49SA/viewform?embedded=true" width="640" height="536" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
    
    """
    
    components.html(survey, width=1200, height=1200)

