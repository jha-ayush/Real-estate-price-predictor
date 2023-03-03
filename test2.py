import streamlit as st
import pandas as pd

import plotly.express as px # Visualizations

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
        
    # Define state and number of bedrooms and bathrooms dropdowns
    state_options = mainland["state"].unique()
    state_selected = st.selectbox("Select a state", state_options)

    bedrooms_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    bedrooms_selected = st.selectbox("Select number of bedrooms", bedrooms_options)

    bathrooms_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    bathrooms_selected = st.selectbox("Select number of bathrooms", bathrooms_options)

    # Filter data based on user selections
    filtered_df = mainland[(mainland["state"] == state_selected) & 
                     (mainland["bed"] == bedrooms_selected) & 
                     (mainland["bath"] == bathrooms_selected)]
    
    
    if st.checkbox(f"Display data for the above criteria for {state_selected}"):
        # Show table
        st.write(f"<b>Dataframe for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        st.write(filtered_df)
        st.write(f"<b>Summary statistics for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        st.write(filtered_df.describe().round(2))

        # Show bar chart
        st.write(f"<b>Zip code vs Price for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        fig = px.bar(filtered_df, x=filtered_df["zip_code"].apply(lambda x: '{0:0>5}'.format(x)), y="price")
        fig.update_xaxes(title_text="Zip Code")
        fig.update_yaxes(title_text="Price (USD)")
        st.plotly_chart(fig)
        
        
    if st.checkbox(f"Display top 10 zip codes per price for the above criteria for {state_selected}"):
        # Show list of top 10 zip codes based on overall price
        top_zipcodes = filtered_df.groupby("zip_code")["price"].mean().reset_index().sort_values(by="price", ascending=False).head(10)
        top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
        top_zipcodes["price"] = top_zipcodes["price"].round(2)

        st.write("Top 10 zip codes:")
        st.write(top_zipcodes.set_index("zip_code"))
    
    
    
    if st.checkbox(f"U.S. Mainland LassoCV Accuracy & Bagging Regressor ML for {state_selected}"):
        zip_selected = st.selectbox("Select a zip code for ML analysis", top_zipcodes)
        st.write(f"Selected zipcode: {zip_selected}")
        
        
        # Filter the dataframe by the selected zip code
        data = filtered_df[filtered_df['zip_code'] == zip_selected]

        # Use zip_selected to generate a new dataframe
        zip_df = zip_selected(zip_selected, data)

        # Display the new dataframe in Streamlit
        st.write(zip_df)
        
        
        
        
        # Split data into input (X) and output (y) variables
        if type(zip_selected) == dict:
            zip_selected_data = zip_selected.values()
        else:
            # Handle the case where zip_selected is not a dictionary
            st.write(zip_selected_data)

            # Split data into input (X) and output (y) variables
            X, y = zip_selected_data[:, :-1], zip_selected_data[:, -1]
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


with tab2:

    st.write(f"<b>To Dos...⏳</b>",unsafe_allow_html=True)
    st.write("<b>'We will use these scoring metrics' - Ridge + LassoCV + ElasticNetCV >> feed into ML model - BaggingReggresor</b>",unsafe_allow_html=True)

    st.write(f"<b>Line graph for prediction 2025</b>",unsafe_allow_html=True)
        
        

#------------------------------------------------------------------#