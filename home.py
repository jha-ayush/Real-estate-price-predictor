############### Real-estate predictor data app using Machine Learning for regions like - U.S. Mainland state, Puerto Rico & U.S. Virgin Islands ###############

############### Import Librarires ###############
import streamlit as st
import pandas as pd

import plotly.express as px # Visualizations

from sklearn.model_selection import train_test_split # Train/ Test package
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # Scoring metrics
from sklearn.linear_model import LassoCV, Ridge, ElasticNet # Machine Learning Models
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor # Regression ML Models

############### Import warnings + watermark ###############
from watermark import watermark
from warnings import filterwarnings
filterwarnings("ignore")
print(watermark())
print(watermark(iversions=True, globals_=globals()))


#######################################################################################################################


############### Set page configurations - ALWAYS at the top ###############
st.set_page_config(page_title="Real estate price predictor",
                   page_icon="🏠",
                   layout="centered",
                   initial_sidebar_state="auto")

@st.cache_data # Add cache data decorator



############### Load and Use local style.css file ###############
def local_css(file_name):
    """
    Use a local style.css file.
    """
    with open(file_name) as f:
        st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")



############### Load data ###############
def load_data(filename):
    try:
        return pd.read_csv(filename)
    except:
        logging.error(f"Cannot find {filename}")
        st.error(f"Failed to load {filename}")
        
        

############### Create variables to load datafiles as dataframes ###############

# Import csv data files for - U.S. Mainland, Puerto Rico, U.S. Virgin Islands
mainland = load_data("./Resources/mainland_data.csv")
puerto_rico = load_data("./Resources/puerto_rico_data.csv")
virgin_islands = load_data("./Resources/virgin_islands_data.csv")

# st.write(mainland["state"].unique())


#######################################################################################################################


############### U.S. Mainland by state - Create dataframes with aggregated by State ###############
mainland_by_state = mainland.groupby('state').agg({'price': 'mean', 'bed': 'mean'}).round(2)

############### U.S. Mainland by zipcode - Create dataframes with aggregated by Zip Code Values ###############
mainland_by_zip = mainland.groupby('zip_code').agg({'price': 'mean', 'bed': 'mean'}).round(2)



############### Puerto Rico - Create dataframes with aggregated by Zip Code Values ###############
puerto_rico_by_zip = puerto_rico.groupby('zip_code').agg({'price': 'mean', 'bed': 'mean'}).round(2)


############### U.S. Virgin Islands - Create dataframes with aggregated by Zip Code Values ###############
virgin_islands_by_zip = virgin_islands.groupby('zip_code').agg({'price': 'mean', 'bed': 'mean'}).round(2)


#######################################################################################################################


############### Title/ header ###############
st.header("Real estate price predictor")
st.write(f"Select from different Machine Learning models to view the best housing predictor for your budget",unsafe_allow_html=True)
st.info("Download Kaggle `csv` data >> Cleanup and group by regions with the following dimensions - `price`, `bed`, `bath`, `acre_lot`, `house_size`, `state`, `zip_code` >> Focus on U.S. Mainland data only >> Display dataframe(s)/visualization(s) >> Run `lazypredict` analysis on the back-end for PR & VI >> Scoring metrics & Regression Model >> Next steps ??")
st.write("---")


#######################################################################################################################


############### Initial app in Streamlit tab format ###############
tab1, tab2, tab3 = st.tabs(["U.S. Mainland", "Puerto Rico", "U.S. Virgin Islands"])


#######################################################################################################################


############### U.S. Mainland ###############
with tab1:
      
    st.subheader("U.S. Mainland")
    
    ############### Define a dictionary to map states to U.S. mainland ###############
    state = {"", "Connecticut", "Delaware", "Maine", "Massachusetts", "New Hampshire", "New Jersey", "New York", "Pennsylvania", "Rhode Island", "Vermont", "West Virginia", "Wyoming"}


    ############### Create a selectbox for the region ###############
    state_selected = st.selectbox("Select a U.S. mainland state", state)
    st.write(f"You have selected the following state: <b>{state_selected}</b>",unsafe_allow_html=True)
    
    
    ############### Display number of bedrooms dropdown menu ###############
    bedrooms_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    bedrooms_selected = st.selectbox(f"Select number of bedroom(s) in the state of {state_selected}", bedrooms_options)
    st.write(f"You have selected the following count for bedroom(s): <b>{bedrooms_selected}</b>",unsafe_allow_html=True)

    ############### Display number of bedrooms dropdown menu ###############
    bathrooms_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    bathrooms_selected = st.selectbox(f"Select number of bathroom(s) in the state of {state_selected}", bathrooms_options)
    st.write(f"You have selected the following count for bathroom(s): <b>{bathrooms_selected}</b>",unsafe_allow_html=True)

    ############### Create new datafram - Filter data based on user selections ###############
    filtered_mainland_df = mainland[(mainland["state"] == state_selected) & 
                     (mainland["bed"] == bedrooms_selected) & 
                     (mainland["bath"] == bathrooms_selected)]
    
    
    
    if st.checkbox(f"Display data for the above criteria for {state_selected}"):
        ############### Show table ###############
        st.write(f"<b>Dataframe for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        st.write(f"We found <b>{filtered_mainland_df.count().price}</b> properties 🏠 matching your criteria! <br>Here's a little preview of the data ⬇️",unsafe_allow_html=True)
        st.write(filtered_mainland_df.head(50))
        # st.write(f"<b>Summary statistics for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        # st.write(filtered_mainland_df.describe().round(2))
        # st.write(filtered_mainland_df.dtypes)

        ############### Show bar chart ###############
        st.write(f"<b>Zip code vs Price for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        fig = px.bar(filtered_mainland_df, x=filtered_mainland_df["zip_code"].apply(lambda x: '{0:0>5}'.format(x)), y="price")
        fig.update_xaxes(title_text="Zip Code")
        fig.update_yaxes(title_text="Price (USD)")
        st.plotly_chart(fig)

        
    st.write("---")
    
    ############### Show list of top 15 zip codes based on overall price ###############  
    if st.checkbox(f"Display top 15 zip codes by median price for the selected state of {state_selected}"):
        top_zipcodes = filtered_mainland_df.groupby("zip_code")["price"].mean().reset_index().sort_values(by="price", ascending=False).head(15)
        top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
        top_zipcodes["price"] = top_zipcodes["price"].round(2)

        st.write("Top 15 zip codes:")
        st.write(top_zipcodes.set_index("zip_code"))
    
    
    st.write("---")
    
    
    ############### Select a single zipcode from the Top 10 list for further analysis ###############
    if st.checkbox(f"Machine Learning model run for {bedrooms_selected} bedroom(s) & {bedrooms_selected} bathroom(s) in the state of {state_selected}"):
        ############### Check if top_zipcodes is empty ###############
        if top_zipcodes.empty:
            st.write("Please select the Display data checkbox above to populate top 10 zip codes.")
        else:
            zip_selected = st.selectbox("Select a zip code for ML analysis", top_zipcodes["zip_code"])

            ############### Filter the dataframe by the selected zip code ###############
            data = filtered_mainland_df[filtered_mainland_df['zip_code'] == zip_selected]
            data = data.drop_duplicates()
            st.write(f"Number of available properties in <b>{zip_selected}</b> zip code: <b>{len(data)}</b>",unsafe_allow_html=True)
            st.write(data)
            
    
            st.write("---")
        
        
        
            ############### Training & Testing - Split data into input (X) and output (y) variables ###############
            X = filtered_mainland_df.drop(["price","state"], axis=1)
            y = filtered_mainland_df["price"]

            ############### Define regression models and scoring metrics ###############
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
            
            
            
            ############### Select a Scoring metric ###############
            scoring_options = ["r2_score", "mean_squared_error", "mean_absolute_error"]
            scoring_selected = st.selectbox("Select a scoring metric", scoring_options)
            
            
            ############### Select a Regression model ###############
            model_options = ["LassoCV", "Ridge", "ElasticNet"]
            model_selected = st.selectbox("Select a model for regression analysis", model_options)

            ############### Add Title for Model Training ###############
            st.subheader("Press the button 🔘 below to train the model")
            
            if st.button("Train Model"):
                ############### Split data into training and testing sets ###############
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

                ############### Initialize the model ###############
                if model_selected == "LassoCV":
                    model = LassoCV()
                elif model_selected == "Ridge":
                    model = Ridge()
                else:
                    model = ElasticNet()

                ############### Train the model ###############
                model.fit(X_train, y_train)

                ############### Test the model ###############
                y_pred = model.predict(X_test)

                ############### Calculate scoring metric ###############
                if scoring_selected == "r2_score":
                    score = r2_score(y_test, y_pred)
                elif scoring_selected == "mean_squared_error":
                    score = mean_squared_error(y_test, y_pred)
                else:
                    score = mean_absolute_error(y_test, y_pred)

                ############### Display results ###############
                st.write(f"Model: <b>{model_selected}</b>",unsafe_allow_html=True)
                st.write(f"Scoring metric: <b>{scoring_selected}</b>",unsafe_allow_html=True)
                st.write(f"Score: <b>{score:.2f}</b>",unsafe_allow_html=True)


                
                
                
                
#######################################################################################################################                
                
############### Puerto Rico ###############
with tab2:

    st.subheader("Puerto Rico")
    
    ############### Display number of bedrooms dropdown menu ###############
    bedrooms_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    bedrooms_selected = st.selectbox(f"Select number of bedroom(s) in Puerto Rico", bedrooms_options)
    st.write(f"You have selected the following count for bedroom(s): <b>{bedrooms_selected}</b>",unsafe_allow_html=True)

    ############### Display number of bedrooms dropdown menu ###############
    bathrooms_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    bathrooms_selected = st.selectbox(f"Select number of bathroom(s) in Puerto Rico", bathrooms_options)
    st.write(f"You have selected the following count for bathroom(s): <b>{bathrooms_selected}</b>",unsafe_allow_html=True)

    ############### Create new datafram - Filter data based on user selections ###############
    filtered_puerto_rico_df = mainland[(mainland["bed"] == bedrooms_selected) & (mainland["bath"] == bathrooms_selected)]
    
    
    
    if st.checkbox(f"Display data for the above criteria for Puerto Rico"):
        ############### Show table ###############
        st.write(f"<b>Dataframe for the above selected criteria for Puerto Rico</b>",unsafe_allow_html=True)
        st.write(f"We found <b>{filtered_puerto_rico_df.count().price}</b> properties 🏠 matching your criteria! <br>Here's a little preview of the data ⬇️",unsafe_allow_html=True)
        st.write(filtered_puerto_rico_df.head(50))
        # st.write(f"<b>Summary statistics for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        # st.write(filtered_puerto_rico_df.describe().round(2))
        # st.write(filtered_puerto_rico_df.dtypes)

        ############### Show bar chart ###############
        st.write(f"<b>Zip code vs Price for the above selected criteria for Puerto Rico</b>",unsafe_allow_html=True)
        fig = px.bar(filtered_puerto_rico_df, x=filtered_puerto_rico_df["zip_code"].apply(lambda x: '{0:0>5}'.format(x)), y="price")
        fig.update_xaxes(title_text="Zip Code")
        fig.update_yaxes(title_text="Price (USD)")
        st.plotly_chart(fig)

        
    st.write("---")
    
    ############### Show list of top 15 zip codes based on overall price ###############  
    if st.checkbox(f"Display top 15 zip codes by median price for Puerto Rico"):
        top_zipcodes = filtered_puerto_rico_df.groupby("zip_code")["price"].mean().reset_index().sort_values(by="price", ascending=False).head(15)
        top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
        top_zipcodes["price"] = top_zipcodes["price"].round(2)

        st.write("Top 15 zip codes:")
        st.write(top_zipcodes.set_index("zip_code"))
    
    
    st.write("---")
    
    
    ############### Select a single zipcode from the Top 10 list for further analysis ###############
    if st.checkbox(f"Machine Learning model run for {bedrooms_selected} bedroom(s) & {bedrooms_selected} bathroom(s) in  Puerto Rico"):
        ############### Check if top_zipcodes is empty ###############
        if top_zipcodes.empty:
            st.write("Please select the Display data checkbox above to populate top 10 zip codes.")
        else:
            zip_selected = st.selectbox("Select a zip code for ML analysis", top_zipcodes["zip_code"])

            ############### Filter the dataframe by the selected zip code ###############
            data = filtered_puerto_rico_df[filtered_puerto_rico_df['zip_code'] == zip_selected]
            data = data.drop_duplicates()
            st.write(f"Number of available properties in <b>{zip_selected}</b> zip code: <b>{len(data)}</b>",unsafe_allow_html=True)
            st.write(data)
            
    
            st.write("---")
        
        
        
            ############### Training & Testing - Split data into input (X) and output (y) variables ###############
            X = filtered_puerto_rico_df.drop(["price"], axis=1)
            y = filtered_puerto_rico_df["price"]

            ############### Define regression models and scoring metrics ###############
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
            
            
            
            ############### Select a Scoring metric ###############
            scoring_options = ["r2_score", "mean_squared_error", "mean_absolute_error"]
            scoring_selected = st.selectbox("Select a scoring metric", scoring_options)
            
            
            ############### Select a Regression model ###############
            model_options = ["LassoCV", "Ridge", "ElasticNet"]
            model_selected = st.selectbox("Select a model for regression analysis", model_options)

            ############### Add Title for Model Training ###############
            st.subheader("Press the button 🔘 below to train the model")
            
            if st.button("Train Model"):
                ############### Split data into training and testing sets ###############
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

                ############### Initialize the model ###############
                if model_selected == "LassoCV":
                    model = LassoCV()
                elif model_selected == "Ridge":
                    model = Ridge()
                else:
                    model = ElasticNet()

                ############### Train the model ###############
                model.fit(X_train, y_train)

                ############### Test the model ###############
                y_pred = model.predict(X_test)

                ############### Calculate scoring metric ###############
                if scoring_selected == "r2_score":
                    score = r2_score(y_test, y_pred)
                elif scoring_selected == "mean_squared_error":
                    score = mean_squared_error(y_test, y_pred)
                else:
                    score = mean_absolute_error(y_test, y_pred)

                ############### Display results ###############
                st.write(f"Model: <b>{model_selected}</b>",unsafe_allow_html=True)
                st.write(f"Scoring metric: <b>{scoring_selected}</b>",unsafe_allow_html=True)
                st.write(f"Score: <b>{score:.2f}</b>",unsafe_allow_html=True)
    
    
                


#######################################################################################################################           
            

############### U.S. Virgin Islands ###############
with tab3:
    
    st.subheader("U.S. Virgin Islands")
    
    ############### Display number of bedrooms dropdown menu ###############
    bedrooms_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
    bedrooms_selected = st.selectbox(f"Select number of bedroom(s) in U.S. Virgin Islands", bedrooms_options)
    st.write(f"You have selected the following count for bedroom(s): <b>{bedrooms_selected}</b>",unsafe_allow_html=True)

    ############### Display number of bedrooms dropdown menu ###############
    bathrooms_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    bathrooms_selected = st.selectbox(f"Select number of bathroom(s) in U.S. Virgin Islands", bathrooms_options)
    st.write(f"You have selected the following count for bathroom(s): <b>{bathrooms_selected}</b>",unsafe_allow_html=True)

    ############### Create new datafram - Filter data based on user selections ###############
    filtered_virgin_islands_df = mainland[(mainland["bed"] == bedrooms_selected) & (mainland["bath"] == bathrooms_selected)]
    
    
    
    if st.checkbox(f"Display data for the above criteria for U.S. Virgin Islands"):
        ############### Show table ###############
        st.write(f"<b>Dataframe for the above selected criteria for U.S. Virgin Islands</b>",unsafe_allow_html=True)
        st.write(f"We found <b>{filtered_virgin_islands_df.count().price}</b> properties 🏠 matching your criteria! <br>Here's a little preview of the data ⬇️",unsafe_allow_html=True)
        st.write(filtered_virgin_islands_df.head(50))
        # st.write(f"<b>Summary statistics for the above selected criteria for {state_selected}</b>",unsafe_allow_html=True)
        # st.write(filtered_virgin_islands_df.describe().round(2))
        # st.write(filtered_virgin_islands_df.dtypes)

        ############### Show bar chart ###############
        st.write(f"<b>Zip code vs Price for the above selected criteria for U.S. Virgin Islands</b>",unsafe_allow_html=True)
        fig = px.bar(filtered_virgin_islands_df, x=filtered_virgin_islands_df["zip_code"].apply(lambda x: '{0:0>5}'.format(x)), y="price")
        fig.update_xaxes(title_text="Zip Code")
        fig.update_yaxes(title_text="Price (USD)")
        st.plotly_chart(fig)

        
    st.write("---")
    
    ############### Show list of top 15 zip codes based on overall price ###############  
    if st.checkbox(f"Display top 15 zip codes by median price for U.S. Virgin Islands"):
        top_zipcodes = filtered_virgin_islands_df.groupby("zip_code")["price"].mean().reset_index().sort_values(by="price", ascending=False).head(15)
        top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
        top_zipcodes["price"] = top_zipcodes["price"].round(2)

        st.write("Top 15 zip codes:")
        st.write(top_zipcodes.set_index("zip_code"))
    
    
    st.write("---")
    
    
    ############### Select a single zipcode from the Top 10 list for further analysis ###############
    if st.checkbox(f"Machine Learning model run for {bedrooms_selected} bedroom(s) & {bedrooms_selected} bathroom(s) in  U.S. Virgin Islands"):
        ############### Check if top_zipcodes is empty ###############
        if top_zipcodes.empty:
            st.write("Please select the Display data checkbox above to populate top 10 zip codes.")
        else:
            zip_selected = st.selectbox("Select a zip code for ML analysis", top_zipcodes["zip_code"])

            ############### Filter the dataframe by the selected zip code ###############
            data = filtered_virgin_islands_df[filtered_virgin_islands_df['zip_code'] == zip_selected]
            data = data.drop_duplicates()
            st.write(f"Number of available properties in <b>{zip_selected}</b> zip code: <b>{len(data)}</b>",unsafe_allow_html=True)
            st.write(data)
            
    
            st.write("---")
        
        
        
            ############### Training & Testing - Split data into input (X) and output (y) variables ###############
            X = filtered_virgin_islands_df.drop(["price"], axis=1)
            y = filtered_virgin_islands_df["price"]

            ############### Define regression models and scoring metrics ###############
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
            
            
            
            ############### Select a Scoring metric ###############
            scoring_options = ["r2_score", "mean_squared_error", "mean_absolute_error"]
            scoring_selected = st.selectbox("Select a scoring metric", scoring_options)
            
            
            ############### Select a Regression model ###############
            model_options = ["LassoCV", "Ridge", "ElasticNet"]
            model_selected = st.selectbox("Select a model for regression analysis", model_options)

            ############### Add Title for Model Training ###############
            st.subheader("Press the button 🔘 below to train the model")
            
            if st.button("Train Model"):
                ############### Split data into training and testing sets ###############
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

                ############### Initialize the model ###############
                if model_selected == "LassoCV":
                    model = LassoCV()
                elif model_selected == "Ridge":
                    model = Ridge()
                else:
                    model = ElasticNet()

                ############### Train the model ###############
                model.fit(X_train, y_train)

                ############### Test the model ###############
                y_pred = model.predict(X_test)

                ############### Calculate scoring metric ###############
                if scoring_selected == "r2_score":
                    score = r2_score(y_test, y_pred)
                elif scoring_selected == "mean_squared_error":
                    score = mean_squared_error(y_test, y_pred)
                else:
                    score = mean_absolute_error(y_test, y_pred)

                ############### Display results ###############
                st.write(f"Model: <b>{model_selected}</b>",unsafe_allow_html=True)
                st.write(f"Scoring metric: <b>{scoring_selected}</b>",unsafe_allow_html=True)
                st.write(f"Score: <b>{score:.2f}</b>",unsafe_allow_html=True)
    
    
    
#######################################################################################################################    