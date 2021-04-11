import os
import pandas as pd 
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import st_state_patch
import SessionState
import seaborn as sns
import plotly.express as px
# Prevent error showing up
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objects as go
from textwrap import wrap
from datetime import datetime, date
import matplotlib as mpl
import urllib
# for min_max scaling
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import psycopg2
from sqlalchemy import create_engine
from streamlit.report_thread import get_report_ctx
import config
st.set_page_config(page_title='SOCVEST') #,layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



def get_session_id():
    session_id = get_report_ctx().session_id
    session_id = session_id.replace('-','_')
    session_id = '_id_' + session_id
    return session_id

# Where the '%s', we be replaced with the values after '%'. 
# for this, we want to save the state of the columns and their respective values
def write_state(column,value,engine,session_id):
    engine.execute("UPDATE %s SET %s='%s'" % (session_id,column,value))

# we want to save the state of the dataframe    
def write_state_df(df,engine,session_id):
    df.to_sql('%s' % (session_id),engine,index=False,if_exists='replace',chunksize=1000)

# We want to read the state of the column values
def read_state(column,engine,session_id):
    state_var = engine.execute("SELECT %s FROM %s" % (column,session_id))
    state_var = state_var.first()[0]
    return state_var

# we want to read the state of the dataframe 
def read_state_df(engine,session_id):
    try:
        df = pd.read_sql_table(session_id,con=engine)
    except:
        df = pd.DataFrame([])
    return df


# Chart
#sns.set_style("whitegrid")

# Get the Data
# Streamlit knows not to always reload data unless its new data.
#@st.cache(persist=True)
# data table for data contents page
#def contents_page():
    #table = {
     #   "COUNTRY": ['United Kingdom']
      #  "DATA TYPE":
       # "SOURCE":
    
    #table = pd

# import data from csv
#def COVID_19_Data():
    #Get headline data
 #   Headline_df = pd.read_csv('./Data/Timeseries/Trends on headline indicators.csv',index_col='Date',parse_dates=True,
  #                      infer_datetime_format=True)

    # Want to get the numeric data
   # Headline_numeric_data = Headline_df.select_dtypes(include=[np.float64,np.int64]) 
    # Want the columns attribute
    #Headline_numeric_cols = Headline_numeric_data.columns
    # Get the text dataframe
    #Headline_text_data = Headline_df.select_dtypes(include='object')
    #Headline_text_cols = Headline_text_data.columns
    
    # Getfuture perception data
    #Future_expect_df = pd.read_csv('./Data/Timeseries/Future expectations COVID.csv',index_col='Date',parse_dates=True,
     #                   infer_datetime_format=True)
    #
    # Want to get the numeric data
    #Future_expectation_numeric_data = Future_expect_df.select_dtypes(include=[np.float64,np.int64]) 
    # Want the columns attribute
    #Future_expectation_numeric_cols = Future_expectation_numeric_data.columns
    # Get the text dataframe
    #Future_expectation_text_data = Headline_df.select_dtypes(include='object')
    #Future_expectation_text_cols = Future_expectation_text_data.columns
    

    #return Headline_df, Headline_numeric_cols, Headline_text_cols, Future_expect_df, Future_expectation_numeric_cols, Future_expectation_text_cols

# All data

#Headline_df, Headline_numeric_cols, Headline_text_cols, Future_expect_df, Future_expectation_numeric_cols, Future_expectation_text_cols = COVID_19_Data()

# Columns
#COVID_19_cols = [*Headline_numeric_cols, *Future_expectation_numeric_cols] # unpack the lists and add them together 
#Data
#COVID_19_data = pd.concat([Headline_df,Future_expect_df], axis=1, ignore_index=False)
# copy of data
#COVID_19_data_2 = pd.concat([Headline_df,Future_expect_df], axis=1, ignore_index=False)


#Creating PostgreSQL client
#engine = create_engine('postgresql://<username>:<password>@localhost:5432/<database_name>')
                       
                        #Creating PostgreSQL client
#engine = create_engine('postgresql://<username>:<password>@localhost:5432/<database name>')


# DATA MANIPULATION - UNIVERSAL
def reshape_data(data: pd.DataFrame):
    date_cols_regex = "^(0[1-9]|[12][0-9]|3[01])[- /.](0[1-9]|1[012])[- /.]((19|20)\\d\\d)$"
    value_vars = list(data.filter(regex=(date_cols_regex)).columns)
    
    data_unpivoted = pd.melt(data, id_vars=['Country', 'Category', 'Series', 'Data'], 
        value_vars=value_vars, var_name='Date', value_name='Value')
    data_unpivoted['Date']=pd.to_datetime(data_unpivoted['Date'], infer_datetime_format=True)
    data_unpivoted['Value']=data_unpivoted['Value'].astype(float)
    #data_unpivoted = data_unpivoted.sort_values(['Country', 'Category', 'Series', 'Data','Date'])
    data_unpivoted.loc[:,'Value'].fillna(method='bfill', inplace = True)
    return data_unpivoted

@st.cache(persist=True)
# COVID_19 DATA
def COVID_19_data():
    data = pd.read_csv('./Data/Timeseries/Social impact.csv', infer_datetime_format = True)
    return data

#@st.cache(persist=True)
# Choose data for datasets
def Filter_COVID_Timeseries_Data(Data_filtering):
    # GET ALL COVID-19 DATA
    data = COVID_19_data()
    data = reshape_data(data)
        

    # GET COUNTRY DATA FOR THE COVID-19 THEME DATABASE
    # COUNTRY - get all values from the countries column
    countries = data.Country.unique()
    #countries = countries.unique() #drop_duplicates(False)
    # first data filtering choice by country
    Country_choice = Data_filtering[0].selectbox("Country", countries)
    # CATEGORY - get all row values in the category column that are in the country column
    category = data['Category'].loc[data['Country'] == Country_choice].unique() 
    Category_choice = Data_filtering[1].selectbox("Category", category)
    # SERIES - get all series row values that are in the category column
    series = data.Series.loc[data['Category']==Category_choice].unique()
    Series_choice = Data_filtering[2].radio('Data Type', series)
    
    # Prepare the dataframe that will be used for other types of analysis
    # Data filteration function - pass data into the function. Filter the data column according to the above choices (First set of choices)
    data_col = data['Data'][(data['Country']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice)].unique()

    # the data to select from the dataframe - we want to select the values in the data column based on what we selected in the select data 
    # Create a new table making columns from the data columns. Use pivot table because if we specify the value, it won't aggregate by mean or some other statistic method. 
    Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Value', fill_value='').rename_axis(None, axis=1) #.reindex(data['Date'].unique(), axis=0)
    
    # Second set of choices
    #question = data['Question'].loc[data['Series'].isin(series)].unique()
    #Question_choice = Data_filtering[0].selectbox("Question", question)
    #Survey_Response = data['Survey Response'].loc[data['Question'].isin(question)].unique()
    #Survey_Response_choice = Data_filtering[1].selectbox("Survey Response", Survey_Response)
    #People_categories = data['People categories'].loc[data['Survey Response'].isin(Survey_Response)].unique()
    #People_categories_choice = Data_filtering[1].selectbox("People categories", People_categories)
    #Response = data['Reponse'].loc[data['People categories'].isin(People_categories)].unique()
    #Response_choice = Data_filtering[1].selectbox("Reponse", Response)
    
    #data col for these choices
    ##data_col2 = data['Data'][(data['Question']==(Question_choice)) & (data['Survey Response']==(Survey_Response_choice)) & (data['People categories']==(People_categories_choice)) &
                             #(data['Reponse']==(Response_choice))].unique()
    
    
    
    # return the whole function
    return data_col, Trans_data



# Choose COVID-19 scaled data
def Filter_COVID_Non_timeseries_Data():
    # GET ALL COVID-19 DATA
    data = COVID_19_data()
    data = reshape_data(data)
    Trans_data = Filter_COVID_Timeseries_Data()
    
    #create selectbox for date filtering
    data_filter = st.beta_columns(3)
        
    # Create labels to slot filtering sections
    #Data_filtering = st.beta_columns(3)

    # GET COUNTRY DATA FOR THE COVID-19 THEME DATABASE
    # COUNTRY - get all values from the countries column
    date = Trans_data.index #drop_duplicates(False)
    # first data filtering choice by country
    Country_choice = data_filter[0].selectbox("Country", countries)
    # CATEGORY - get all row values in the category column that are in the country column
    category = data['Category'].loc[data['Country'] == Country_choice].unique() #Country_choice]
    Category_choice = Data_filtering[1].selectbox("Category", category)
    # SERIES - get all series row values that are in the category column
    series = data.Series.loc[data['Category']==Category_choice].unique()
    Series_choice = Data_filtering[2].radio('Data Type', series)
    
    # Prepare the dataframe that will be used for other types of analysis
    # Data filteration function - pass data into the function. Filter the data column according to the above choices
    data_col = data['Data'][(data['Country']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice)].unique()

    # the data to select from the dataframe - we want to select the values in the data column based on what we selected in the select data 
    # Create a new table making columns from the data columns. Use pivot table because if we specify the value, it won't aggregate by mean or some other statistic method. 
    Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Value', fill_value='').rename_axis(None, axis=1).reindex(data['Date'].unique(), axis=0)
    
    # return the whole function
    return data_col, Trans_data


# View filtered data as dataframe
#def View_Filtered_Data_DataFrame(data):
     
       
                    

            
    #Country_choice, Category_choice, Series_choice = Filter_COVID_Data(data)
    # Users get to choose data from the data column
    
       # if choose_theme_category == 'Other':
    #    st.write("Choose data common to all countries for this theme.")
     #   Countries = st.beta_columns(3)
       # Country = Countries[0].multiselect("Country", countries)            
        #Data_category = Countries[1].multiselect("Data Category", empty_category)
        #Data_type = data_segments[2].radio('Data Type', ['Timeseries','Non-Timeseries'])
        #data_items = st.success("Items")
        #data_items.multiselect("Items", lol)

        # when comparing these data, turn them into the time frame standards - weekly, monthly, annual. Provide the option to 
        # select or blank out time frame (Have an option to view data on a standardised mode - daily, weekly, montlhy etc or Unique - data that have the same time frames or shape)

      #  data_mix_buttons = st.beta_columns([3,1,1])
        #confirmation_data = data_mix_buttons[0].button("Show Data")
       # Hide_data = data_mix_buttons[2].checkbox("Hide Data")
    
    #return data_col, Trans_data


# Visualisation functions
def Line_Area_chart(Dataframe_to_display, Columns_to_show):
    plotly_fig_area = px.area(data_frame=Dataframe_to_display,x=Dataframe_to_display.index,y=Columns_to_show)

    # Legend settings
    plotly_fig_area.update_layout(showlegend=False)        
    plotly_fig_area.update_layout(margin_autoexpand=True) # prevent size from changing because of legend or anything
    plotly_fig_area.update_traces(mode="lines", hovertemplate=None)
    plotly_fig_area.update_layout(hovermode="x unified")

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

@st.cache(persist=True)
def Heatmap_Timeseries_data_prep(data_to_analyse, Heatmap_dataframe_timeseries):
    # Extract day, month and year from dataframe table. First reset index
    Data_to_select_no_index = Heatmap_dataframe_timeseries.reset_index()

    # Create new dataframes from the data column
    Data_to_select_no_index['Year'] = Data_to_select_no_index.Date.dt.year
    Data_to_select_no_index['Month'] = Data_to_select_no_index.Date.dt.month
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
    title = st.beta_columns([2,2])
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
#st.sidebar.markdown("#### A partner to the retail investor")

#username, password, database_name
username = config.username
password = config.password
database = config.database_name

#Creating PostgreSQL client
engine = create_engine('postgresql://' + username + ':' + password+ '@localhost:5432/' + database) 
#engine = create_engine('postgresql://username:password@localhost:5432/database name')

#Getting session ID
session_id = get_session_id()

# Create table using the current session
engine.execute("CREATE TABLE IF NOT EXISTS %s (Variable text)" % (session_id)) 
len_table =  engine.execute("SELECT COUNT(*) FROM %s" % (session_id));
len_table = len_table.first()[0]
if len_table == 0:
    engine.execute("INSERT INTO %s (Variable) VALUES ('1')" % (session_id));

# st.sidebar.title("Get Started")
pages = ["About us", "Data and Analysis"] #, "Feedback"]

# About us page
choice_1 = st.sidebar.selectbox("Menu", pages)

if choice_1 =="About us":
    title1,title2,title3 = st.beta_columns(3)
    title2.title("SOCVEST")
    banner_picture = st.beta_columns(1)
    with banner_picture[0]:
        st.image("img/Socvest.jpg")
    
    # st.title("SOCVEST")
    themes1 = st.beta_columns([1,3,1])
    
    themes1[1].text("Bringing clarity to a world we used to gaze into")
    #themes2.write("A partner to the retail investor")
    #themes3.write("Supporter of the retail investor")
    
    st.header("Who are we?")
    st.write("We are an alternative data provider for the retail investor.")
    st.header("Why do we exist?")
    st.write("Retail investors have always relied on alternative means of deriving investment thesis and insights. Limited industry transparency and costly service providers tailored to professional investment companies meant retail traders were always left behind. Things are changing. There are masses of data and the technology that can make deriving data driven insights into the world, markets and to construct investment opportunities are available at relatively low cost. We aim to further enhance the retail trader's data driven decision making capabilities.")
    
    st.write("This platform is here to serve you, the new age investor.")
    
    st.header("What do we do?")
    st.write("We provide alternative data for world and investment themes to support the retail trader in their investment decision making. We also provide the capacity to run machine learning models for research and predictive probability purposes.")
    
    # Map
    #countries = ['United Kingdom']
    #df = pd.DataFrame(countries, columns=['Country'])
    #Map = st.map(df)
    
    # labels to indicate we do not provide investment adviceh
    st.success("We do not provide investment and financial advice. Data and capabilities provided on this platform is not financial advice")    
    
elif choice_1 == "Data and Analysis":  
    st.title("Data Exploration")
    st.write("Select a dataset and the range of data visualisation tools available to build your analysis. Once done, press the 'Show Data' below to view the data.")
    #st.write("For visualisations, choose an option from the Sidebar to start your data analysis.")

    # Data exploration
    #st.sidebar.title("Data Mix")
    #st.sidebar.markdown("Choose your dataset")
    # The option to upload your own data
    #user_data = st.sidebar.checkbox("Upload your own data")
    
    #if user_data == True:
         # Title detailing what this data set is (User data)
     #   st.subheader("Your Uploaded data")
        # want to ensure the error that comes up when we use file uploader does not show up
       # st.set_option("deprecation.showfileUploaderEncoding", False)
      #  file_upload = st.sidebar.file_uploader(label="Upload your csv or Excel file",
       #                                        type = ["csv", "xlsx"], accept_multiple_files=False)
        # initially file will be empty. So inform them to upload file to view data
        #show_file = st.empty()
       # if not file_upload:
        #    show_file.info("Please upload a file : {} ".format(' '.join(["csv", "xlsx"])))

        # Need to come up with an alternative - if not a datetime series
        #st.sidebar.markdown("When uploading data, please note that the first column will be automatically transformed into the index column.                               If your data is a timeseries, please ensure your first column is the Date column, formating it as 'yyyy-mm-dd'.")

        # 
        
       # global all_columns 
        #if file_upload is not None:
 #           try:
  #              df = pd.read_csv(file_upload,index_col=0,parse_dates=True,infer_datetime_format=True)       
   #         except Exception as e: 
    #            df = pd.read_excel(file_upload.getvalue(),engine='openpyxl',index_col=0)  
             # Want to get the numeric data
     #       all_columns = df.columns.to_list()
            
            # display data only when file is uploaded                      
      #      checkbox_userdata = st.checkbox("Display Data")
       #     if checkbox_userdata:                    
        #        user_data_selection = st.multiselect("Choose the columns from your dataset", options=all_columns, key='dataframe')
                # The dataframe to show the data. Only show dataframe with these columns (Only show data that has the same column length/number of rows)
         #       Final_user_data = df[user_data_selection]

                # if columns not selected...
          #      if not user_data_selection:
           #         st.error("Please select at least one column.")
            #    else:
                    # if data is timeseries data, column 1 will be turned into an index label the column as 'Date' and format as follows:
                    # yyyy-mm-dd or 
             #       st.dataframe(Final_user_data)

    #st.sidebar.title('Datasets')    
    #app_data = st.sidebar.radio("Select from available data.", ['Datasets']) #, 'Data mix'])
    # Datasets
    #if app_data == 'Datasets':   
    # data choices
    Data_choices = st.sidebar.radio("Choose dataset", ['COVID-19', 'Economy', 'Geopolitics','Country','Financial Markets'])

    if Data_choices == 'COVID-19':
        # create title based on choice            
        title = Data_choices = 'COVID-19'
        st.subheader(title)
        # Create expander to view/choose data
        with st.beta_expander("Choose Data"):

    #country_specific = st.success("Data Segments")
    #choose_theme_category = country_specific.radio('Theme Category', ['Country specific', 'Other'])

            #if choose_theme_category == 'Country specific':
            st.write("Choose data unique to a country for this theme.")
            
            # Create labels to slot filtering sections
            Data_filtering = st.beta_columns(3)

            # define data to be used for this section
            # Get the select boxes that will be used for filtering the data. Load the filtered data and the pivoted datatable
            data_col, Trans_data = Filter_COVID_Timeseries_Data(Data_filtering)

            # create new labels for hide data
            data_mix_buttons = st.beta_columns([3,1,1])
            # We use this to hide or show the data
            Hide_data = data_mix_buttons[2].checkbox("Hide Data", value=True) # default
            #read_state('size',engine,session_id)

            if not Hide_data:
                st.markdown("##### Social data is from surveys conducted by the UK government")
                # title to show they can select data
                DF_Display = st.subheader("View Data Table")
                # Create a multiselect to choose the columns based on the filtered data
                Select_cols_dataframe = st.multiselect("Choose columns to Display", options=data_col, key="dataframe choice")# distinct selectbox
                #read_state('Column selection', engine, session_id)
                #write_state('Select_cols_dataframe',Select_cols_dataframe, engine, session_id)
                #Select_cols_dataframe = read_state('Select_cols_dataframe',engine,session_id)
                
                Data_to_select= Trans_data[Select_cols_dataframe] # dataframe to select from
                #write_state_df(Data_to_select,engine,session_id + '_df')
                
                if not Select_cols_dataframe:
                    st.error("Please select at least one column.")
                else:
                    st.write(Data_to_select.style.set_precision(2)) 
                        
                    #THIS SET UP IS FOR THE NON-TIMESERIES (Trend to the left, stats to the right)
                    #Additional data in its own container 
                   # additional_data_title = dataframe_selection[1].markdown("##### Additional Data")
                    ##extra_filter_one = dataframe_selection[1].selectbox("Question", options = ("Hiya"))
      #              extra_filter_one = dataframe_container.extra_filter_one
       #             extra_filter_two = dataframe_selection[1].selectbox("Survey Response", options = ("Hiya"))
        #            extra_filter_two = dataframe_container.extra_filter_two
         #           extra_filter_three = dataframe_selection[1].selectbox("People categories", options = ("Hiya"))
          ##          extra_filter_three = dataframe_container.extra_filter_four
            #        extra_filter_four = dataframe_selection[1].selectbox("Reponse", options = ("Hiya"))
             #       extra_filter_four = dataframe_container.extra_filter_four
                    
              #      Select_cols_dataframe = dataframe_selection[2].multiselect("Choose columns to Display", options=data_col, key="dataframe 2 choice")
               #     Data_to_select1= Trans_data[Select_cols_dataframe] # dataframe to select from
                #    if not Select_cols_dataframe:
                 #       dataframe_selection[2].error("Please select at least one column.")
                  #  else:
                   #     dataframe_selection[2].write(Data_to_select1) #.style.set_precision(2))
                    
               # else:
                    # subheader
                   # Non_timeseries_DF_Disp = st.subheader("View Data Table")
                    
                    # Create further filters for showing other data
                    #non_timeseries_data = st.beta_columns([2,5])
                    #non_timeseries_data_2 = st.beta_columns(3)

                    #Data 
                    #data = COVID_19_data()
                    #data = reshape_data(data) #unpivoted data
                    
                    #Date filter                   
                  #  nT_date = data.Date.loc[data.Series==Series_choice]
                   # nT_date_filter = non_timeseries_data[0].selectbox("Choose date", options = nT_date)
                    #Question filter
                    #Question_NT = data.Question.loc[data.Date==nT_date]
                    #Question_NT_1 = Question_NT.unique() #.fillna("") #sorts#.unique()
                    #Question_nT_Filter = non_timeseries_data[1].selectbox("Choose Question", options=Question_NT) 
                    # Survey response
                    #Survey_NT = data['Survey Response'].loc[data.Question==Question_NT]
                    #Survey_nT_Filter = non_timeseries_data_2[0].multiselect("Choose Survey Response", options=Survey_NT) #	Survey Response	People categories	Reponse
                    #.loc[data['Question'] == nT_date]#.unique()
                    
                    
                    
                    # NON TIMESERIES DATA
                    # After you remove the unhide button what will show, some few criterias. If the timeseries in not ticked, then show something else else, show the dataframe.
               # if not Series_choice == "Non-timeseries":
                #    # create container
                 #   dataframe_container = st.beta_container()
                  #  dataframe_selection = st.beta_columns([8,2,8]) 
                    
                    # title to show they can select data
                   # DF_Display = st.subheader("View Data Table")
            #        # Create a multiselect to choose the columns based on the filtered data
             ##       Select_cols_dataframe = dataframe_selection[0].multiselect("Choose columns to Display", options=data_col, key="dataframe choice") # distinct selectbox
            #        Data_to_select= Trans_data[Select_cols_dataframe] # dataframe to select from
            #        if not Select_cols_dataframe:
             #           dataframe_selection[0].error("Please select at least one column.")
             #       else:
              #          dataframe_selection[0].write(Data_to_select.style.set_precision(2)) 
                        
                    #THIS SET UP IS FOR THE NON-TIMESERIES (Trend to the left, stats to the right)
                    #Additional data in its own container 
           #         additional_data_title = dataframe_selection[1].markdown("##### Additional Data")
            #        extra_filter_one = dataframe_selection[1].selectbox("Question", options = ("Hiya"))
             #       extra_filter_one = dataframe_container.extra_filter_one
              #      extra_filter_two = dataframe_selection[1].selectbox("Survey Response", options = ("Hiya"))
               ##     extra_filter_two = dataframe_container.extra_filter_two
           #         extra_filter_three = dataframe_selection[1].selectbox("People categories", options = ("Hiya"))
            #        extra_filter_three = dataframe_container.extra_filter_four
            #        extra_filter_four = dataframe_selection[1].selectbox("Reponse", options = ("Hiya"))
            #        extra_filter_four = dataframe_container.extra_filter_four
                    
             #       Select_cols_dataframe = dataframe_selection[2].multiselect("Choose columns to Display", options=data_col, key="dataframe 2 choice")
              #      Data_to_select1= Trans_data[Select_cols_dataframe] # dataframe to select from
               #     if not Select_cols_dataframe:
                ##        dataframe_selection[2].error("Please select at least one column.")
                 #   else:
                #        dataframe_selection[2].write(Data_to_select1) #.style.set_precision(2))
                    
                    
                    
                    
                    
                    
                    
                    # GET COUNTRY DATA FOR THE COVID-19 THEME DATABASE
    # COUNTRY - get all values from the countries column
    #date = Trans_data.index #drop_duplicates(False)
    # first data filtering choice by country
    #Country_choice = data_filter[0].selectbox("Country", countries)
    # CATEGORY - get all row values in the category column that are in the country column
    #category = data['Category'].loc[data['Country'] == Country_choice].unique() #Country_choice]
    #Category_choice = Data_filtering[1].selectbox("Category", category)
    # SERIES - get all series row values that are in the category column
    #series = data.Series.loc[data['Category']==Category_choice].unique()
    #Series_choice = Data_filtering[2].radio('Data Type', series)
    
    # Prepare the dataframe that will be used for other types of analysis
    # Data filteration function - pass data into the function. Filter the data column according to the above choices
    #data_col = data['Data'][(data['Country']==Country_choice) & (data['Category']==Category_choice) & (data['Series']==Series_choice)].unique()

    # the data to select from the dataframe - we want to select the values in the data column based on what we selected in the select data 
    # Create a new table making columns from the data columns. Use pivot table because if we specify the value, it won't aggregate by mean or some other statistic method. 
    #Trans_data=data.pivot_table(index=['Date'], columns='Data', values='Value', fill_value='').rename_axis(None, axis=1).reindex(data['Date'].unique(), axis=0)
                    
        # CHARTS
        st.sidebar.title("Visualisation")
        # Show charts for selected data
        Visualisation_segment = st.sidebar.beta_expander("Choose Visualisation Method")
        # The opportunity to select visualisation type
        # Label timeseries
        Visualisation_segment.subheader("Timeseries")
        
        # LINE CHART
        line_chart = Visualisation_segment.checkbox("Line Chart")
        if line_chart:
            # subtitle
            st.subheader("Line Chart Visualisation")
            show_line_chart = st.beta_expander("Show Chart")
            # if the show line chart shows
            with show_line_chart:
                # Create a multiselect to choose the columns based on the filtered data
                Data_to_show_line = st.multiselect("Choose Data to Display", options=data_col, key='linechart')
                Dataframe_to_display_line_chart = Trans_data[Data_to_show_line] # dataframe to select from
                # new label for line chart options
                line_chart_options = st.beta_columns(3)  
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
                    

        heatmap_timeseries = Visualisation_segment.checkbox("Heatmap") 
        
        if heatmap_timeseries:
        
            st.subheader("Timeseries Heatmap")

            Show_heatmap_times = st.beta_expander("Show Chart")
            with Show_heatmap_times:           
            
                # Data to select
                data_to_analyse = st.selectbox("Choose data", options=data_col)
                # Dataframe to choose data from
                Heatmap_dataframe_timeseries = Trans_data[data_to_analyse]
                
                # new label for line chart options
                Axis_data = st.beta_columns([3,3,3,3])   
                
                Data_to_select_indexed = Heatmap_Timeseries_data_prep(data_to_analyse, Heatmap_dataframe_timeseries) 
                
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
                    
                #across_time = st.checkbox("Variables across time")
                #if across_time:
                    # load data to view in charts
                    # date/time
                    # Data_to_select_indexed = Heatmap_Timeseries_data_prep(data_to_analyse, Heatmap_dataframe_timeseries) 
                
                    #axis_data = Heatmap_timeseries_index(data_to_analyse, Heatmap_dataframe_timeseries)
                    
                    
                    # Data to show on graph
                    # Data to select
                 #   variables_to_analyse = st.multiselect("Choose data", options=data_col)
                  #  dates_to_analyse = st.selectbox("Choose date range", options=axis_data)
                    # Dataframe to choose data from
                   # Heatmap_dataframe_timeseries_variables = Trans_data[variables_to_analyse]
                    
                    
                    # turn date time into just date
                    # Dates = pd.to_datetime(Heatmap_dataframe_timeseries_variables.index).strftime('%d-%m-%Y')

                    #labels
                    #labels = Heatmap_dataframe_timeseries_variables.columns
                    #wrapped_labels = ['\n'.join(wrap(l, 20)) for l in labels] 

                    # Data to select
                    # data_to_analyse = st.multiselect("Choose data", options=COVID_19_cols)
                    # pivot data/turn it into sns format
                   # Pivot_data_heatmap = pd.pivot_table(Heatmap_dataframe_timeseries_variables, columns=variables_to_analyse, values=axis_data)

                    #f, ax2 = plt.subplots(figsize=(20, 10))  
                    # show data
                    #ax2 = sns.heatmap(Pivot_data_heatmap) #, linewidths=5, linecolor="pink", yticklabels = wrapped_labels, square=True)
                    
                    #st.pyplot()

                    # set labels
                    #ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
                    #ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, horizontalalignment='right')
                    
                    
        
        # Data Relationship
        Subheader_data_relationship = Visualisation_segment.subheader("Data Relationships")
        
        # Categorical data (Scatter graph, correlation heatmap)             
        charts_relationship = Visualisation_segment.checkbox("Scatter Chart")
        # if the button is selected
        if charts_relationship==True:  
            # Title indicating this is a timeseries
            st.subheader("Scatter Chart")
            
            Relationship_charts = st.beta_expander("Show Charts")
            with Relationship_charts: 
                 
                st.write("*Note that data has been scaled to allow for better ploting outcomes")
                # define data that will be used for analysis (the whole dataset)
                scaling = MinMaxScaler()
                # Scale data
                scaled_dataframe= scaling.fit_transform(Trans_data)
                # create new dataframe with scaled data
                scaled_dataframe = pd.DataFrame(scaled_dataframe, index=Trans_data.index, columns=data_col)
                
                Scatter_graph_data = scaled_dataframe[data_col]
                
                axes_placeholder = st.empty()
                scatter_choices_placeholder = st.empty()
                chart_placeholder = st.empty()
                
               # @st.cache(allow_output_mutation=True, suppress_st_warning=True)
            
                #session_state = SessionState.get(axes_options = axes_placeholder.beta_columns(3))
                
                # session_state = SessionState.get(
                
                # select data for default graph
                axes_options = st.beta_columns(4)

                # Options for default chart
                scatter_choices = st.beta_columns([4,4,4,4,1])

                # User_choices_placeholder = st.empty()

                #select x-axis
                select_box_X = axes_options[0].selectbox('Select x-axis', data_col, key=1)
                #select y-axis
                select_box_Y = axes_options[1].selectbox('Select y-axis', data_col, key=2) 
                # Control dimensions of chart
                color = ['navy', 'blue', 'green', 'brown', 'skyblue', 'grey', 'black', 'cornsilk'] 
                color = axes_options[2].selectbox('Choose chart color', options=color, key=1)
                # Edge colors
                edge_col = ['navy','red', 'blue']
                edge_col_opt = axes_options[3].selectbox('Choose edge plots colour', options=edge_col)
                # Height
                height = scatter_choices[1].slider('Size of chart', min_value=0.0, max_value=30.0, step=.1, value=9.0, key=1)
                # Ratio
                #ratio_chart = scatter_choices[1].slider('Ratio', min_value=0, max_value=30, step=1, value=7, key=2)
                # Space
                space = scatter_choices[2].slider('Distance main/edge plot', min_value=0.0, max_value=1.0, step=.1, value=.4, key=3)
                # Show cuts/shapes of plots
                alpha = scatter_choices[3].slider('Plot density', min_value=0.0, max_value=1.0,  step=.1, value=.4, key=4) 
                # Default settings
                #default_options = scatter_choices[4].button('Default', key=1)
                
                bkc = ['darkgrid', 'whitegrid', 'dark', 'white']
                # graph background colour
                Background_colr = scatter_choices[0].selectbox('Choose Background Colour', options=bkc)
                
                
                #ticks for margin plots
                #margin_grid = scatter_choices[4].radio('Show grid for edge plots', ['True', 'False'])
                
                sns.set_style(Background_colr)
                
                chart = sns.jointplot(x=select_box_X, 
                                          y=select_box_Y, 
                                          data = Scatter_graph_data,
                                          alpha=alpha,
                                          height=height,  
                                          space=space, 
                                          xlim=(-0.01,1.01), 
                                          ylim=(-0.01,1.01),
                                          color=color,
                                          marginal_kws={'color':edge_col_opt}) #, color=color) # hue=select_box_hue, s=select_box_size,  marginal_ticks=True)

                st.pyplot(chart, use_container_width=True)
                
                Advanced_plots = st.beta_columns([4,4,3,3,5])
                
               # Advanced_plots_opt = Advanced_plots[4].checkbox("Advanced Chart")
                
              #  if Advanced_plots_opt:
                    
               #     advanced_plot_axes =  st.beta_columns(4)
                #    advanced_plot_scatter_choices = st.beta_columns([4,4,4,4,1])
                 #   chart_plot = st.beta_columns([30,2])
                    
                    #select x-axis
               #     Adv_select_box_X = advanced_plot_axes[0].selectbox('Select x-axis', data_col)
                    #select y-axis
             #       Adv_select_box_Y = advanced_plot_axes[1].selectbox('Select y-axis', data_col)
                    # Control dimensions of chart
              #      Adv_color = ['navy', 'blue', 'green', 'brown', 'skyblue', 'winterred', 'grey']

               #     Adv_colour = advanced_plot_axes[2].selectbox('Choose color', options=Adv_color)

                #    Adv_height = advanced_plot_scatter_choices[0].slider('Height of chart', min_value=0.0, max_value=30.0, step=.1, value=8.0)
                        # Rati0
                 #   Adv_ratio_chart = advanced_plot_scatter_choices[1].slider('Ratio', min_value=0, max_value=30, step=1, value=7)
                        # Space
                  #  Adv_space = advanced_plot_scatter_choices[2].slider('Space', min_value=0.0, max_value=1.0, step=.1, value=.4)
                        # Show cuts/shapes of plots
                 #   Adv_alpha = advanced_plot_scatter_choices[3].slider('Plot density', min_value=0.0, max_value=1.0,  step=.1, value=.4) 
                        # Default settings
                  #  Adv_default_options = advanced_plot_scatter_choices[4].button('Default')
                        # Additional options
                    #Adv_options = Advanced_plots_opt[4].checkbox('Additional Options')
                    
                    # view both charts
                    #edge_colour = advanced_plot_scatter_choices[3].selectbox('Edge Color', options=Adv_color)
                    
                    
                    
                    #chart = sns.JointGrid(x=Adv_select_box_X, y=Adv_select_box_Y, data=Scatter_graph_data,  ratio=Adv_ratio_chart, space=Adv_space, height=Adv_height) #, , size=select_box_size) 
                    #chart.plot(sns.scatterplot, sns.histplot,alpha=Adv_alpha, edgecolor=edge_colour) #, edgecolor=".2", linewidth=.5), alpha=alpha, ratio=8, marginal_ticks=True)
                   # if st.radio("Chart_options", ['scatter', 'blue']) =='blue':
                     #   chart.plot_joint(sns.scatterplot, s=100, alpha=.5)
                    #chart.plot_marginals(sns.histplot, kde=True)
                    #chart.sns.kdeplot(y=Adv_select_box_Y, linewidth=2, ax=chart.ax_marg_y)
                    #chart.plot_joint(sns.regplot) #, s=100, alpha=.5) plot(sns.regplot,
                    
                    #plt.colorbar(mappable=Adv_select_box_Y)
                
                    #sns.scatterplot(color='blue')
                    #jointplot(x=select_box_X, y=select_box_Y, data=Scatter_graph_data,  hue=select_box_hue, s=select_box_size, alpha=alpha, ratio=8, marginal_ticks=True) #, xlim=(-15,15), ylim=(-15,15)) #.set_axis_labels(wrapped_labels)#, yticklabels = wrapped_labels) #, marker="o", color = colour); #, s=select_box_size, alpha=.6, marker="o", color = 'green');
                    #plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
                    #cbar_ax = chart.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
                    #cb1  = mpl.colorbar.ColorbarBase(cbar_ax,orientation='vertical')

                    #chart.ax_joint.set_ylabel(chart.wrapped_labels, fontweight='bold')
                    #chart.ax_marg_x.set_axis_off()
                    #chart.ax_marg_y.set_axis_off()
                    # Define axes limit
                    #chart.ax_marg_x.set_xlim(0, 3000)
                    #chart.ax_marg_y.set_ylim(0, 1200)

                    #plt.figure(figsize=(16, 6))
                    #chart_plot[0].pyplot(chart, use_container_width=True)
                    
                
                # CREATE A MORE ADVANCED CHART USING JOINT GRID
                
               # if options:
                    
                    
                #    kind_opt = ['kde', 'reg']
                 #   kind = scatter_choices[0].selectbox('Additional Options', options=kind_opt)

                #labels
                #labels = Scatter_graph_data.columns
                #wrapped_labels = ['\n'.join(wrap(l, 20)) for l in labels] 

                # Option to turn off and on additional axis
                # additional_axis = scatter_choices[0].radio('Add additional axis plot', ['Yes', 'No'])


                # what category influences the other dataset
                #select_box_hue = scatter_choices[2].selectbox('Category', data_col)
                # what category influences the other dataset
                #select_box_size = scatter_choices[3].slider('Size of plots', min_value=1, max_value=100, value=35) #range(1,1000))
                
                #chart_placeholder = st.empty()

                # Chart
                #@st.cache(suppress_st_warning=True, allow_output_mutation=True)
               # def chart(select_box_X, select_box_Y, Scatter_graph_data, alpha, height, ratio_chart, space, color):
                  #  chart = sns.jointplot(x=select_box_X, 
                   #                       y=select_box_Y, 
                    #                      data = Scatter_graph_data,
                     #                     kind=kind,
                      #                    height=height, 
                       #                   ratio=ratio_chart, 
                        #                  space=space, 
                         #                 xlim=(-0.01,1.01), 
                          #                ylim=(-0.01,1.01),
                           #               color=color)
                                         # marginal_kws={'color':'red'}) #, color=color) # hue=select_box_hue, s=select_box_size,  marginal_ticks=True)

                    # chart_placeholder.pyplot(chart, use_container_width=True)
                
                # chart(select_box_X, select_box_Y, Scatter_graph_data, alpha, height, ratio_chart, space, color)
                
                #session_state.checkboxed = False
                
                # the above is the initial state
                #s0 = st.SessionState()
                #if not s:
                 #   s.pressed_first_button = False
                    
                #if st.button("Default"): #or s.pressed_first_button:
                 #   s.pressed_first_button = True # preserve the info that you hit a button between runs
           # try: 
                
                #s = st.State() 
                
                # session_state = SessionState.get(axes_options = axes_placeholder.beta_columns(3))
               # session_state = SessionState.get(axes_options = axes_placeholder.beta_columns(3)) 
                # once button is pushed, load new format data
                
                
                #if default_options: # or session_state.default_options:
                    #session_state.default_options=False
                    
                 #   try:
                    
                        # create a session for the below choices so that it won't have to reinitialise
                        #session_state = SessionState.get(axes_options = axes_placeholder.beta_columns(3)) 
                                                         #(scatter_choices = scatter_choices_placeholder.beta_columns([4,4,4,4,1]))
                        #session_state.axes_options

                        # select data for default graph
                  #      axes_options = axes_placeholder.beta_columns(3)
                        #Options for default chart
                   #     scatter_choices = scatter_choices_placeholder.beta_columns([4,4,4,4,1])

                        #session_state.checkboxed = True

                        #select x-axis
                    #    select_box_X = axes_options[0].selectbox('Select x-axis', data_col)
                        #default_placeholder.select_box_X
                        #select y-axis
                     #   select_box_Y = axes_options[1].selectbox('Select y-axis', data_col)
                        #session_state.checkboxed = True

                        # Control dimensions of chart
                      #  color = ['navy', 'blue', 'green', 'brown', 'skyblue', 'winterred', 'grey']

                       # color = axes_options[2].selectbox('Choose color', options=color)

                        #height = scatter_choices[0].slider('Height of chart', min_value=0.0, max_value=30.0, step=.1, value=8.0)
                        # Rati
                    #    ratio_chart = scatter_choices[1].slider('Ratio', min_value=0, max_value=30, step=1, value=7)
                        # Space
                   #     space = scatter_choices[2].slider('Space', min_value=0.0, max_value=1.0, step=.1, value=.4)
                        # Show cuts/shapes of plots
                #        alpha = scatter_choices[3].slider('Plot density', min_value=0.0, max_value=1.0,  step=.1, value=.4) 
                        # Default settings
                 #       default_options = scatter_choices[4].button('Default')
                        # Additional options
                  #      options = scatter_choices[4].checkbox('Additional Options')
                        
                    
                   #     chart = sns.jointplot(x=select_box_X, 
                    #                        y=select_box_Y, 
                     #                       data=Scatter_graph_data,
                      #                      alpha=.4, 
                       #                     height=8.0, 
                        #                    ratio=7, 
                         #                   space=.4, 
                          #                  xlim=(-0.01,1.01), 
                           #                 ylim=(-0.01,1.01), 
                            #                color=color)                   
                       # chart_placeholder.pyplot(chart, use_container_width=True)
                    #except:
                     #   pass
                    
                  
              


                   
                    
                    


                    
               
                
                
                
                
                #Default chart
                #g = sns.JointGrid()
                #x, y = select_box_X, select_box_Y
                #sns.scatterplot(x=select_box_X, y=select_box_Y, data=Scatter_graph_data, ec="b", fc="none", s=100, linewidth=1.5, ax=g.ax_joint)
                #sns.histplot(x=select_box_X, data=Scatter_graph_data, fill=False, linewidth=2, ax=g.ax_marg_x)
                #sns.kdeplot(y=select_box_Y, data=Scatter_graph_data, linewidth=2, ax=g.ax_marg_y)
                
                #if additional_axis == 'Yes':
                 #   chart = sns.JointGrid(x=select_box_X, y=select_box_Y, data=Scatter_graph_data,  hue=select_box_hue, s=select_box_size, alpha=alpha, height=8, ratio=6, marginal_ticks=True)
                    #chart.plot_joint(sns.regplot)
                  #  chart.ax_marg_x.set_axis_off()
                   # chart.ax_marg_y.set_axis_off()
                #elif additional_axis == 'No':
                  #  chart = sns.jointplot(x=select_box_X, y=select_box_Y, data=Scatter_graph_data,  hue=select_box_hue, s=select_box_size, alpha=alpha, height=8, ratio=6, marginal_ticks=True)
          

                        

                # Correlation
        Correlation = Visualisation_segment.checkbox("Correlation")
        if Correlation:
            st.subheader("Correlation") 
            # subtitle
            st.subheader("Correlation Chart Visualisation")
            show_scatter_chart = st.beta_expander("Show Chart")
            #Show_corr = st.checkbox("Show Correlation Diagram")
            with show_scatter_chart:
                
                # labels
                Chart_options_heatmap = st.beta_columns(4) 
    
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
            show_Histogram_chart = st.beta_expander("Show Chart")
            # if you choose Scatter Plot
            with show_Histogram_chart:
                
                # columns for flexible options
                Hist_options = st.beta_columns(4)
                
                bkc = ['darkgrid', 'whitegrid', 'dark', 'white']
                # graph background colour
                Background_col = Hist_options[2].selectbox('Choose Background Colour', options=bkc, key=1)
                sns.set_style(Background_col)
                
                # KDE argument
                KDE_plot = Hist_options[1].selectbox("Show KDE Plot", options=(True, False))
                # Horizontal/vertical
                horizontal_vertical = Hist_options[3].radio("Choose Orientation", ['Horizontal','Vertical'], index=1)
                # select feature from select box
                histogram_data_selection = Hist_options[0].selectbox("Choose feature to observe", options=data_col)
                histogram_slider = Hist_options[0].slider(label="Number of bins to display", min_value=5, max_value= 30, value = 15)
                # Statistics
                stats = ["count", "frequency", "density", "probability"]
                count = Hist_options[1].selectbox('Choose stats to view', options=stats, index=2)
                # styling
                fill = Hist_options[2].selectbox('Fill bar plots', options=(True, False))
                visual_stat = ['bars', 'step', 'poly']
                element = Hist_options[3].selectbox('Visual of stat', options=visual_stat)
                cumulative = Hist_options[0].selectbox('Show Cumulation data', options=(True,False))
                colour_hist = ['indianred', 'indigo']
                colour = Hist_options[1].selectbox("Colour", options=colour_hist)
                # Size control
                height = Hist_options[2].slider("Chart height", min_value=5, max_value=30, value=5, step=1)
                width = Hist_options[3].slider("Chart width", min_value=5, max_value=30, value=8, step=1)

                
                 # Data to view
                COVID_19_data_Hist = Trans_data[data_col]
                
                if not horizontal_vertical == 'Vertical':
                    
                    f, ax = plt.subplots(figsize=(width,height)) 
                    sns.histplot(y=histogram_data_selection, data=COVID_19_data_Hist, kde=KDE_plot, stat=count, fill=fill, element=element, cumulative=cumulative, color=colour) #, bins = histogram_slider, binwidth=bin_width, binrange=(bin_range_1,bin_range_2))
                    st.pyplot()
                else:
                    
                    f, ax = plt.subplots(figsize=(width,height)) 
                    sns.histplot(x=histogram_data_selection, data=COVID_19_data_Hist, kde=KDE_plot, stat=count, fill=fill, element=element, cumulative=cumulative, color=colour) #, bins = histogram_slider, binwidth=bin_width, binrange=(bin_range_1,bin_range_2))
                    st.pyplot()
                    
                # ADVANCED CHARTS
                 # Bin width
                #bin_width = Hist_options[1].number_input("Select custom bin width", min_value=None, max_value=100, value=5, step=1)
                # Bin range
                #bin_range_1 = Hist_options[2].number_input("Select custom bin range begin", min_value=None, max_value=100, value=5, step=1)
                #bin_range_2 = Hist_options[3].number_input("Select custom bin range end", min_value=None, max_value=100, value=5, step=1)

        # Bar Chart
       # if Visualisation_segment.checkbox("Bar Charts") == True:
        #     # subtitle for Bar chart
         #   st.subheader("Bar Chart Visualisation")
          #  show_Bar_chart = st.beta_expander("Show Chart")
           # with show_Bar_chart:
                
#                #Chart options (X,Y)
 #               Bar_Chart_choices = st.beta_columns(3)
  #                              
   #             X_Chart_choices = Bar_Chart_choices[0].selectbox(label="Choose X-axis", options=data_col)
    #            
     #           Y_Chart_Choices = Bar_Chart_choices[1].selectbox(label="Choose Y-axis", options=data_col)
      #          
       #         Colour_options = Bar_Chart_choices[2].selectbox(label='Choose colour', options=data_col)
        #        
         #       size_chart_bar = Bar_Chart_choices[0].slider('Choose Height of Chart', min_value=0, max_value=1000, step=10, value=400)
          #      
           #     width_char_bar = Bar_Chart_choices[1].slider('Choose Width of Chart', min_value=0, max_value=1000, step=10, value=400)
                
                # Only show dataframe with these columns
            #    dataframe_cols = Trans_data[data_col] #reset_index()
                
                #Columns for chart options
                # Shape_of_bar_chart = st.beta_columns(5)
                
#                Bar_chart = Bar_Chart_choices[2].radio("Bar Chart Options",('Vertical', 'horizontal'))    
 #               if Bar_chart == 'Vertical':    
                    
                    #st.write(dataframe_cols2)

                    #print(dataframe_cols2)

  #                  fig = px.bar(dataframe_cols, x=X_Chart_choices, y=Y_Chart_Choices, color=Y_Chart_Choices)
#
 #                   fig.update_layout(showlegend=False)        
  #                  fig.update_layout(margin_autoexpand=False)
   #                 fig.update_traces(hovertemplate=None)
    #                fig.update_layout(hovermode="x unified")

     #               height = size_chart_bar
      #              width = width_char_bar
       #             fig.update_layout(
        #            width=width,
         #           height=height)

 #                   st.plotly_chart(fig, use_container_width=True)
#
  #              if Bar_chart == 'horizontal':
                    #Bar_chart_selection = st.selectbox(label = "Choose feature", options=data_col)
                    # Only show dataframe with these columns
                    # Only show dataframe with these columns
   #                 dataframe_cols = Trans_data[Bar_chart_selection] #reset_index()

    #                fig = px.bar(dataframe_cols, x=Bar_chart_selection, y=dataframe_cols.index, color=Bar_chart_selection,orientation='h')

     #               fig.update_layout(margin_autoexpand=False)
      #              fig.update_traces(hovertemplate=None)
       #             fig.update_layout(hovermode="x unified")

        #            height = 590
         #           fig.update_layout(
          #          width=350,
           #         height=height)

            #        st.plotly_chart(fig, use_container_width=True)

        
        # MACHINE LEARNING
       # st.sidebar.title("Machine Learning")
        #Machine_Learning = st.sidebar.beta_expander("Choose Machine Learning Model")
        #Linear_regression = Machine_Learning.checkbox("Linear Regression")
        
        #if Linear_regression:
         #   title_linear_regression = "Linear Regression"
          #  Linear_reg_title = st.title(title_linear_regression)
           # st.subheader("Test the Normality of the data")
            #st.write("Linear Regression tends to work best normally distributed data. Below are three methods to help you determine the normality of the data.")
            #Trans_data.hist(bins=3)
            




            
                
                
                           
                
            
    if Data_choices == 'Economy':
        title = Data_choices = 'Economy'
        st.subheader(title)
        st.markdown("## Coming soon...")
        #Hide_data = Choose_Data()

            
            
            #if country_choice == 'United Kingdom':
                
    
    if Data_choices == 'Geopolitics':
        title = Data_choices = 'Geopolitics'
        st.subheader(title)
        st.markdown("## Coming soon...")
        #Hide_data = Choose_Data()
    
    
   # if app_data == 'Data mix':
    #    # Data available on the platform
     #   st.subheader("Available Datasets")
        
      #  Hide_data = Choose_Data()
        
        
        
            
        # Display dataframe data when you tick the box
        #checkbox = st.button("Show Data")
       # if confirmation_data:
            # Can select what columns to look at. 
        #    Columns_select = st.multiselect(label="Select Columns to view",
         #                                   options=COVID_19_cols)      
            # Only show dataframe with these columns (Only show data that has the same column length/number of rows)
          #  dataframe_cols = COVID_19_data[Columns_select]
            
             # columns to be transformed to 2 decimal places
            #Two_decimal_places = dataframe_cols.style.format({'Life satisfaction': '{:.02f}', 'Worthwhile': '{:2f}', 'Happiness yesterday':  '{:2f}', 'Anxious yesterday':  '{:2f}',
                                                            #  '% with high levels of anxiety': '{:.2f}'})

            # if columns not selected...
           # if not Columns_select:
            #    st.error("Please select at least one column.")

            #else:
             #   st.dataframe(dataframe_cols.style.set_precision(2))   
    
    
    # Data graphs
    
   
   #     charts_display = st.sidebar.checkbox("Timeseries")

    #    if charts_display==True:
            # Title indicating this is a timeseries
      #      st.subheader("Time Series Analysis")
     #       
    
    if Data_choices == 'Country':
        title = Data_choices = 'Country'
        st.subheader(title)
        st.markdown("## Coming soon...")
        #Hide_data = Choose_Data()
        
        
    if Data_choices == 'Financial Markets':
        title = Data_choices = 'Financial Markets'
        st.subheader(title)
        st.markdown("## Coming soon...")
        #Hide_data = Choose_Data()
        
    #if choice_1 =="Feedback":
        
    
        
        
    
    
                
                
                    
                        
            
    
