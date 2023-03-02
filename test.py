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


#------------------------------------------------------------------#


# Create dataframes with aggregated by State
mainland_by_state = mainland.groupby('state').agg({'price': 'mean', 'bed': 'mean'}).round(2)

# Create dataframes with aggregated by Zip Code Values
mainland_by_zip = mainland.groupby('zip_code').agg({'price': 'mean', 'bed': 'mean'}).round(2)


#------------------------------------------------------------------#


# Title/ header
st.header("Real estate price predictor")
st.write(f"Select from different Machine Learning models to view the best housing predictor for your budget, grouped by Zip Code - <b>4,051</b> from U.S. Mainland, <b>136</b> from Puerto Rico and <b>6</b> from U.S. Virgin Islands",unsafe_allow_html=True)
st.info("Download Kaggle `csv` data >> Cleanup and group by regions with the following dimensions - `price`, `bed`, `bath`, `acre_lot`, `house_size`, `zip_code` >> Funnel down to `price`, `bed`, `bath`, `zip_code` >> Remove outliers >> Focus on U.S. Mainland data only >> Display dataframe(s)/visualization(s) >> Run `lazypredict` analysis >> Features Accuracy coefficient using LassoCV >> Implement BaggingRegressor with R-squared score & Root Mean Squared Error metrics >> Next steps ??")
st.write("---")


#------------------------------------------------------------------#


tab1, tab2 = st.tabs(["Intro", "Accuracy & ML"])

with tab1:
        
    # Define the location dropdown
    state_options = ["Connecticut", "Delaware",
                     "Massachusetts", "Maine", "New Hampshire", "New Jersey",
                     "New York", "Pennsylvania", "Rhode Island", "Vermont", "West Virginia", "Wyoming"]
    state = st.selectbox("Select a U.S. mainland state from the dropdown menu below", state_options)


    # Print the location and state
    st.write(f"State selected: <b>{state}</b>",unsafe_allow_html=True)

    st.write("---")
    
    st.write(f"Display cleaned <b>{state}</b> Data Visualizations",unsafe_allow_html=True)
    st.write(f"<b>Data desc {state}</b>",mainland.describe().round(2),unsafe_allow_html=True)
    # st.write(f"<b>Data types</b>",mainland.dtypes,unsafe_allow_html=True)
    st.write(f"<b>Number of bedrooms vs price bar chart</b>",unsafe_allow_html=True)
    st.bar_chart(mainland,x="bed",y="price",width=450, height=450,use_container_width=True)

    st.write(f"<b>Visualization by zip - add desc</b>",unsafe_allow_html=True)
    st.bar_chart(mainland_by_zip, x="bed", y="price", height=450, use_container_width=True)

    st.write(mainland_by_zip.shape)
    st.write(mainland_by_zip.keys())
    st.write(f"<b>Data descriptive statistics</b>",unsafe_allow_html=True)
    st.write(mainland_by_zip.describe().round(2))

    st.write(f"Data grouped by `state`",mainland_by_state)
    st.write(f"Data grouped by `zip_code`",mainland_by_zip)
    
    
    st.write("---")

    # Define dropdown number of bedrooms
    dropdown_bedroom = ['1','2','3','4','5','6','7','8']

    # Create dropdown
    selected_option = st.selectbox('Select bedroom count:', dropdown_bedroom)


    # Define dropdown number of bathrooms
    dropdown_bathroom = ['1','2','3','4','5','6','7','8']

    # Create dropdown
    selected_option = st.selectbox('Select bathroom count:', dropdown_bathroom)
    
    
    st.write("---")
    
    st.write(f"Display list of zipcodes ranked with the highest count of user bed/bath preferences for <b>{state}</b>",unsafe_allow_html=True)
    
    st.write("---")
    
    st.write(f"User can then pick a zipcode from the ranked list for scoring + ML<b>{state}</b>",unsafe_allow_html=True)
    
    st.write("---")
    
    
#------------------------------------------------------------------#


with tab2:

    st.write(f"<b>To Dos...⏳</b>",unsafe_allow_html=True)
    st.write("<b>'We will use these scoring metrics' - Ridge + LassoCV + ElasticNetCV >> feed into ML model - BaggingReggresor</b>",unsafe_allow_html=True)

    st.write(f"<b>Line graph for prediction 2025</b>",unsafe_allow_html=True)
        
    
    if st.checkbox("U.S. Mainland LassoCV Accuracy & Bagging Regressor ML"):
        # Split data into input (X) and output (y) variables
        mainland_by_zip_data=mainland_by_zip.values
        st.write(mainland_by_zip_data)
        
        # Split data into input (X) and output (y) variables
        X, y = mainland_by_zip_data[:, :-1], mainland_by_zip_data[:, -1]
        st.write(X)
        st.write(y)
        
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
        
        

#------------------------------------------------------------------#