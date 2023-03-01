import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt # Visualizations

from sklearn.linear_model import LassoCV # Target variable prediction
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Import warnings + watermark
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


#------------------------------------------------------------------#

# Set page configurations - ALWAYS at the top
st.set_page_config(page_title="Real estate price predictor",
                   page_icon="🏠",
                   layout="centered",
                   initial_sidebar_state="auto")

@st.cache_data # Add cache data decorator

# Load and Use local style.css file
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")


# Load data
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except:
        logging.error(f"Cannot find {filename}")
        st.error(f"Failed to load {filename}")
        
        
# Create variables to load datafiles as dataframes
mainland = load_data("./Resources/mainland_data.csv")
puerto_rico = load_data("./Resources/puerto_rico_data.csv")
virgin_islands = load_data("./Resources/virgin_islands_data.csv")


#------------------------------------------------------------------#


# Create dataframes with aggregated Zip Code Values
mainland_by_zip = mainland.groupby('zip_code').agg({'price': 'mean', 'house_size': 'mean'}).round(2)
puerto_rico_by_zip = puerto_rico.groupby('zip_code').agg({'price': 'mean', 'house_size': 'mean'}).round(2)
virgin_islands_by_zip = virgin_islands.groupby('zip_code').agg({'price': 'mean', 'house_size': 'mean'}).round(2)


#------------------------------------------------------------------#


# Title/ header
st.header("Real estate price predictor")
st.write(f"Select from different Machine Learning models to view the best housing predictor for your budget - <b>4,030</b> from U.S. Mainland, <b>136</b> from Puerto Rico and <b>6</b> from U.S. Virgin Islands",unsafe_allow_html=True)
st.info("Download Kaggle `csv` data > Cleanup and group by regions with the following dimensions - `price`, `bed`, `bath`, `acre_lot`, `house_size`, `zip_code` > Display dataframe(s)/visualization(s) > Features Accuracy coefficient using LassoCV > Implement BaggingRegressor with R-squared score & Root Mean Squared Error metrics > Next steps ??")
st.write("---")


#------------------------------------------------------------------#


tab1, tab2, tab3 = st.tabs(["Intro", "Accuracy & ML", "Config"])

with tab1:
    
    # Show dataset grouped by regions
    
    # U.S. Mainland
    if st.checkbox("View U.S. Mainland data"):
        st.write(f"<b>Clean Data</b>",mainland,unsafe_allow_html=True)
        st.write(f"<b>Cleaned data visualization - add desc</b>",unsafe_allow_html=True)
        st.bar_chart(mainland,x="house_size",y="price",width=450, height=450,use_container_width=True)
        
        st.write(f"<b>Data agg by zip</b>",mainland_by_zip,unsafe_allow_html=True)
        st.write(f"<b>Visualization by zip - add desc</b>",unsafe_allow_html=True)
        st.bar_chart(mainland_by_zip, x="house_size", y="price", height=450, use_container_width=True)
        
        st.write(mainland_by_zip.shape)
        st.write(mainland_by_zip.keys())
        st.write(f"<b>Data descriptive statistics</b>",unsafe_allow_html=True)
        st.write(mainland_by_zip.describe())
        
        
    
    # Puerto Rico
    if st.checkbox("View Puerto Rico data"):        
        st.write(f"<b>Clean Data</b>",puerto_rico,unsafe_allow_html=True)
        st.write(f"<b>Cleaned data visualization - add desc</b>",unsafe_allow_html=True)
        st.bar_chart(puerto_rico,x="house_size",y="price",width=450, height=450,use_container_width=True)
        
        st.write(f"<b>Data agg by zip</b>",puerto_rico_by_zip,unsafe_allow_html=True)
        st.write(f"<b>Visualization by zip - add desc</b>",unsafe_allow_html=True)
        st.bar_chart(puerto_rico_by_zip, x="house_size", y="price", height=450, use_container_width=True)
        
        st.write(puerto_rico_by_zip.shape)
        st.write(puerto_rico_by_zip.keys())
        st.write(f"<b>Data descriptive statistics</b>",unsafe_allow_html=True)
        st.write(puerto_rico_by_zip.describe())
        
        
    
    # U.S. Virgin Islands
    if st.checkbox("View U.S. Virgin Islands data"):        
        st.write(f"<b>Clean Data</b>",virgin_islands,unsafe_allow_html=True)
        st.write(f"<b>Cleaned data visualization - add desc</b>",unsafe_allow_html=True)
        st.bar_chart(virgin_islands,x="house_size",y="price",width=450, height=450,use_container_width=True)
        
        st.write(f"<b>Data agg by zip</b>",virgin_islands_by_zip,unsafe_allow_html=True)
        st.write(f"<b>Visualization by zip - add desc</b>",unsafe_allow_html=True)
        st.bar_chart(virgin_islands_by_zip, x="house_size", y="price", height=450, use_container_width=True)
        
        st.write(virgin_islands_by_zip.shape)
        st.write(virgin_islands_by_zip.keys())
        st.write(f"<b>Data descriptive statistics</b>",unsafe_allow_html=True)
        st.write(virgin_islands_by_zip.describe())
        
        
             
#------------------------------------------------------------------#


with tab2:
    if st.checkbox("To Dos"):
        st.write(f"<b>WIP...⏳</b>",unsafe_allow_html=True)
        st.write(f"<b>Pick 1 or 2 Accuracy score metrics</b>",unsafe_allow_html=True)
        st.write("<b>'We used these ML models' - Ridge + LassoCV + ElasticNetCV >> feed into BaggingReggresor</b>",unsafe_allow_html=True)
        
        st.write(f"<b>Line graph for prediction 2025</b>",unsafe_allow_html=True)
        
    
    if st.checkbox("U.S. Mainland LassoCV Accuracy & Bagging Regressor ML"):
        # Split data into input (X) and output (y) variables
        mainland_by_zip_data=mainland_by_zip.values
        
        # Split data into input (X) and output (y) variables
        X, y = mainland_by_zip_data[:, :-1], mainland_by_zip_data[:, -1]
        
        # Define LassoCV model
        model = LassoCV()

        # Fit the model on the whole dataset
        model.fit(X, y)

        # Make predictions on the whole dataset
        y_pred = model.predict(X)

        # Calculate R-squared score
        lasso_score = r2_score(y, y_pred)
        st.write(f"LassoCV R-squared score: <b>{lasso_score:.3f}</b>",unsafe_allow_html=True)
        
        # Define BaggingRegressor model
        bagging_model = BaggingRegressor(base_estimator=model, n_estimators=50, random_state=1)

        # Fit the BaggingRegressor model on the whole dataset
        bagging_model.fit(X, y)

        # Make predictions on the whole dataset
        y_pred = bagging_model.predict(X)

        # Calculate R-squared score and RMSE
        bagging_score = r2_score(y, y_pred)
        bagging_rmse = mean_squared_error(y, y_pred, squared=False)

        # Print BaggingRegressor model performance metrics
        st.write("BaggingRegressor Model Performance Metrics:")
        st.write(f"R-squared score: <b>{bagging_score:.3f}</b>",unsafe_allow_html=True)
        st.write(f"Root Mean Squared Error: <b>{bagging_rmse:.3f}</b>",unsafe_allow_html=True)

    
    
    
    if st.checkbox("Puerto Rico LassoCV Accuracy & Bagging Regressor ML"):
        # Split data into input (X) and output (y) variables
        puerto_rico_by_zip_data=puerto_rico_by_zip.values
        
        # Split data into input (X) and output (y) variables
        X, y = puerto_rico_by_zip_data[:, :-1], puerto_rico_by_zip_data[:, -1]
        
        # Define LassoCV model
        model = LassoCV()

        # Fit the model on the whole dataset
        model.fit(X, y)

        # Make predictions on the whole dataset
        y_pred = model.predict(X)

        # Calculate R-squared score
        lasso_score = r2_score(y, y_pred)
        st.write(f"LassoCV R-squared score: <b>{lasso_score:.3f}</b>",unsafe_allow_html=True)
        
        
        # Define BaggingRegressor model
        bagging_model = BaggingRegressor(base_estimator=model, n_estimators=50, random_state=1)

        # Fit the BaggingRegressor model on the whole dataset
        bagging_model.fit(X, y)

        # Make predictions on the whole dataset
        y_pred = bagging_model.predict(X)

        # Calculate R-squared score and RMSE
        bagging_score = r2_score(y, y_pred)
        bagging_rmse = mean_squared_error(y, y_pred, squared=False)

        # Print BaggingRegressor model performance metrics
        st.write("BaggingRegressor Model Performance Metrics:")
        st.write(f"R-squared score: <b>{bagging_score:.3f}</b>",unsafe_allow_html=True)
        st.write(f"Root Mean Squared Error: <b>{bagging_rmse:.3f}</b>",unsafe_allow_html=True)
        
        
    
    
    if st.checkbox("Virgin Islands LassoCV Accuracy & Bagging Regressor ML"):
        # Split data into input (X) and output (y) variables
        virgin_islands_by_zip_data=virgin_islands_by_zip.values
        
        # Split data into input (X) and output (y) variables
        X, y = virgin_islands_by_zip_data[:, :-1], virgin_islands_by_zip_data[:, -1]
        
        # Define LassoCV model
        model = LassoCV()

        # Fit the model on the whole dataset
        model.fit(X, y)

        # Make predictions on the whole dataset
        y_pred = model.predict(X)

        # Calculate R-squared score
        lasso_score = r2_score(y, y_pred)
        st.write(f"LassoCV R-squared score: <b>{lasso_score:.3f}</b>",unsafe_allow_html=True)
        
        # Define BaggingRegressor model
        bagging_model = BaggingRegressor(base_estimator=model, n_estimators=50, random_state=1, bootstrap_features=True)

        # Fit the BaggingRegressor model on the whole dataset
        bagging_model.fit(X, y)

        # Make predictions on the whole dataset
        y_pred = bagging_model.predict(X)

        # Calculate R-squared score and RMSE
        bagging_score = r2_score(y, y_pred)
        bagging_rmse = mean_squared_error(y, y_pred, squared=False)

        # Print BaggingRegressor model performance metrics
        st.write("BaggingRegressor Model Performance Metrics:")
        st.write(f"R-squared score: <b>{bagging_score:.3f}</b>",unsafe_allow_html=True)
        st.write(f"Root Mean Squared Error: <b>{bagging_rmse:.3f}</b>",unsafe_allow_html=True)
        
        
        

#------------------------------------------------------------------#



with tab3:
    if st.checkbox("Config"):
        # Define the location dropdown
        location_options = ["U.S. Mainland", "Puerto Rico", "U.S. Virgin Islands"]
        location = st.selectbox("Select a location", location_options)

        # Define the state dropdown (only visible if U.S. Mainland is selected)
        if location == "U.S. Mainland":
            state_options = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
                             "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
                             "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
                             "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
                             "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
                             "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
                             "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
            state = st.selectbox("Select a state", state_options)
        else:
            state = None

        # Print the location and state
        st.write(f"Selected region: <b>{location}</b>",unsafe_allow_html=True)
        if state:
            st.write(f"Selected state: <b>{state}</b>",unsafe_allow_html=True)
        
        st.write("---")
        
        # Define dropdown number of bedrooms
        dropdown_bedroom = ['1','2','3','4','5','6']

        # Create dropdown
        selected_option = st.selectbox('Select bedroom count:', dropdown_bedroom)
        
        
        # Define dropdown number of bathrooms
        dropdown_bathroom = ['1','2','3','4']

        # Create dropdown
        selected_option = st.selectbox('Select bathroom count:', dropdown_bathroom)
        
        
