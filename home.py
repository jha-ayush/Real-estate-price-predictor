import streamlit as st
import pandas as pd

import plotly.express as px # Visualizations

from sklearn.linear_model import LassoCV, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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


tab1, tab2 = st.tabs(["Intro", "Next Steps"])

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

        
    st.write("---")
    
        
    if st.checkbox(f"Display top 10 zip codes per price for the selected state of {state_selected}"):
        # Show list of top 10 zip codes based on overall price
        top_zipcodes = filtered_df.groupby("zip_code")["price"].mean().reset_index().sort_values(by="price", ascending=False).head(10)
        top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
        top_zipcodes["price"] = top_zipcodes["price"].round(2)

        st.write("Top 10 zip codes:")
        st.write(top_zipcodes.set_index("zip_code"))
    
    
    st.write("---")
    
    
    # Select a single zipcode from the Top 10 list for further analysis
    if st.checkbox(f"Scoring metrics & ML models for {state_selected}"):
        # Check if top_zipcodes is empty
        if top_zipcodes.empty:
            st.write("Please select the Display data checkbox above to populate top 10 zip codes.")
        else:
            zip_selected = st.selectbox("Select a zip code for ML analysis", top_zipcodes["zip_code"])

            # Filter the dataframe by the selected zip code
            data = filtered_df[filtered_df['zip_code'] == zip_selected]
            data = data.drop_duplicates()
            st.write(f"Number of available properties in <b>{zip_selected}</b> zip code: <b>{len(data)}</b>",unsafe_allow_html=True)
            st.write(data)
            
    
            st.write("---")
        
        
        
            # Split data into input (X) and output (y) variables
            X = filtered_df.drop(["price","state"], axis=1)
            y = filtered_df["price"]

            # Define regression models and scoring metrics
            models = {
                "LassoCV": LassoCV(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet()
            }
            scoring_metrics = {
                "R^2 Score": r2_score,
                "Mean Absolute Error": mean_absolute_error,
                "Root Mean Squared Error": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
            }
            
            
            
            # Select a scoring metric
            scoring_options = ["r2_score", "mean_squared_error", "mean_absolute_error"]
            scoring_selected = st.selectbox("Select a scoring metric", scoring_options)
            
            
            # Select a model
            model_options = ["LassoCV", "Ridge", "ElasticNet"]
            model_selected = st.selectbox("Select a model for regression analysis", model_options)


            if st.button("Train Model"):
                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

                # Initialize the model
                if model_selected == "LassoCV":
                    model = LassoCV()
                elif model_selected == "Ridge":
                    model = Ridge()
                else:
                    model = ElasticNet()

                # Train the model
                model.fit(X_train, y_train)

                # Test the model
                y_pred = model.predict(X_test)

                # Calculate scoring metric
                if scoring_selected == "r2_score":
                    score = r2_score(y_test, y_pred)
                elif scoring_selected == "mean_squared_error":
                    score = mean_squared_error(y_test, y_pred)
                else:
                    score = mean_absolute_error(y_test, y_pred)

                # Display results
                st.write(f"Model: <b>{model_selected}</b>",unsafe_allow_html=True)
                st.write(f"Scoring metric: <b>{scoring_selected}</b>",unsafe_allow_html=True)
                st.write(f"Score: <b>{score:.2f}</b>",unsafe_allow_html=True)

            
            

#------------------------------------------------------------------#


with tab2:

    st.write(f"<b>To Dos...⏳</b>",unsafe_allow_html=True)
    st.write(f"<b>Line graph for prediction 2025</b>",unsafe_allow_html=True)
              

#------------------------------------------------------------------#