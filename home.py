############### Real-estate predictor data app using Machine Learning for regions like - U.S. Mainland state, Puerto Rico & U.S. Virgin Islands ###############

############### Import Librarires ###############
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px # Visualizations

from sklearn.model_selection import train_test_split # Train/ Test package
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error # Scoring metrics
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
mainland = load_data("./Resources/data_files/mainland_data.csv")
puerto_rico = load_data("./Resources/data_files/puerto_rico_data.csv")
virgin_islands = load_data("./Resources/data_files/virgin_islands_data.csv")

# st.write(mainland["state"].unique())


#######################################################################################################################


############### U.S. Mainland by zipcode - Create dataframes with aggregated by Zip Code Values ###############
mainland_by_zip = mainland.groupby('zip_code').agg({'price': 'median', 'bed': 'median'}).round(2)



############### Puerto Rico - Create dataframes with aggregated by Zip Code Values ###############
puerto_rico_by_zip = puerto_rico.groupby('zip_code').agg({'price': 'median', 'bed': 'median'}).round(2)


############### U.S. Virgin Islands - Create dataframes with aggregated by Zip Code Values ###############
virgin_islands_by_zip = virgin_islands.groupby('zip_code').agg({'price': 'median', 'bed': 'median'}).round(2)


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
    state = ["Connecticut", "Delaware", "Maine", "Massachusetts", "New Hampshire", "New Jersey", "New York", "Pennsylvania", "Rhode Island", "Vermont", "West Virginia", "Wyoming"]
    # Sort states alphabetically
    state.sort()


    ############### Create a selectbox for the region ###############
    state_selected = st.selectbox("Select a U.S. mainland state", state)
    
    
    ############### Display number of bedrooms dropdown menu ###############
    bedrooms_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    bedrooms_selected = st.selectbox(f"Select number of bedroom(s) in the state of {state_selected}", bedrooms_options)


    ############### Display number of bedrooms dropdown menu ###############
    bathrooms_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    bathrooms_selected = st.selectbox(f"Select number of bathroom(s) in the state of {state_selected}", bathrooms_options)


    ############### Create new datafram - Filter data based on user selections ###############
    filtered_mainland_df = mainland[(mainland["state"] == state_selected) & 
                     (mainland["bed"] == bedrooms_selected) & 
                     (mainland["bath"] == bathrooms_selected)]
    
    
    
    ############### Show list of top zip codes based on Median price ###############
    top_zipcodes = filtered_mainland_df.groupby("zip_code")["price"].median().reset_index().sort_values(by="price", ascending=False)
    top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
    top_zipcodes["price"] = top_zipcodes["price"].round(2)

    st.write(f"<b>Here is a list of all the zip codes by median descening price in {state_selected} for the above {bedrooms_selected} bed & {bathrooms_selected} bath criteria ⬇️</b>",unsafe_allow_html=True)
    st.write(top_zipcodes.set_index("zip_code").drop_duplicates().sort_values(by="price", ascending=False))

    
    
    st.write("---")
    
    
    ############### Show property listings of a selected zipcode from the top list ###############
    
    ############### Create dropdown to select a specific zipcode ###############
    selected_zipcode = st.selectbox("Select a zipcode from the above top list", top_zipcodes["zip_code"])
    
    ############### Create & display dataframe for selected zipcode ###############
    selected_zipcode_df = filtered_mainland_df[filtered_mainland_df["zip_code"] == int(selected_zipcode)].sort_values(by="price", ascending=False)

    
    ############### Assign label to each row displayed ###############
    selected_zipcode_df = mainland[mainland["zip_code"] == selected_zipcode].sort_values(by="price", ascending=False)

    ############### Add new column with labels ###############
    selected_zipcode_df["label"] = [f"Property {i+1}" for i in range(len(selected_zipcode_df))]

    ############### Drop state & zip_code columns ###############
    selected_zipcode_df = selected_zipcode_df.drop(columns=["state", "zip_code"], axis=1)

    ############### Show unique values of the dataframe ###############
    unique_properties_df = selected_zipcode_df.drop_duplicates()

    st.write(f"<b>Here is a list of {len(unique_properties_df)} property listings for the zipcode {selected_zipcode} in {state_selected}:</b>",unsafe_allow_html=True)
    st.write(unique_properties_df.set_index("label"))


    
    ############### Display bar chart ###############

        
    
    st.write("---")
    
    
        
    ############### Selected property dropdown menu ###############
    property_selected = st.selectbox(f"Select a property label below for price prediction for the zipcode {selected_zipcode}", selected_zipcode_df["label"])

    # Extract numeric part of the property label
    property_number = int(property_selected.split()[-1])

    # Filter the DataFrame to show only the selected property
    selected_property_df = selected_zipcode_df[selected_zipcode_df["label"] == f"Property {property_number}"]

    # Display the DataFrame for the selected property
    st.write(f"<b>Here is the data for {property_selected}:</b>", unsafe_allow_html=True)
    st.write(selected_property_df)



    st.write("---")



    ############### Select a Regression model ###############

    model_options = ["LassoCV", "Ridge", "ElasticNet", "BaggingRegressor", "GradientBoostingRegressor", "RandomForestRegressor"]

    ############### Add Title for Model Training ###############

    if st.button(f"Run price prediction for {selected_zipcode} zipcode"):


        ############### Training & Testing - Split data into input (X) and output (y) variables ###############
        X = selected_zipcode_df[["house_size"]]
        y = selected_zipcode_df["price"]

        ############### Split data into training and testing sets ###############
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
        st.write(f"X Training set size:", len(X_train))
        st.write(f"X Testing set size:", len(X_test))
        st.write(f"y Training set size:", len(y_train))
        st.write(f"y Testing set size:", len(y_test))


        ############### Define the models ###############
        models = [RandomForestRegressor(random_state=10),
                  BaggingRegressor(random_state=10),
                  GradientBoostingRegressor(random_state=10),
                  LassoCV(random_state=10),
                  Ridge(random_state=10),
                  ElasticNet(random_state=10)]

        ############### Train and evaluate the models ###############
        best_model = None # Declare initial variable
        best_score = float("inf") # Declare datatype
        # Iterate through each model in the 'models' array
        for model in models:

            # Fit train & test model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Use Mean Squared Error metrics for scoring metrics
            score = mean_squared_error(y_test, y_pred)
            # st.write(f"Score",score)
            # Move to use RMSE
            rmse = np.sqrt(score)
            # st.write(f"RMSE",rmse)

            # st.write(f"{type(model).__name__} - Root Mean Squared Error: {score:.2f}")
            if score < best_score:
                best_score = score
                best_model = model

        ############### Display the best model and its metrics ###############
        st.write(f"Best model: {type(best_model).__name__} + Root Mean Squared Error - {best_score:.2f}")
        st.balloons()
        st.write(f"Price predicted $XYZ USD for time X, signifying a increase/decrease of Y%")
        st.write(f"<b>🚧 UNDER CONSTRUCTION: Add metrics and price prediction explainations 🚧</b>",unsafe_allow_html=True)            
                
                
                
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

    
    
    st.write("---")
        
        
        
    ############### Create new datafram - Filter data based on user selections ###############
    filtered_puerto_rico_df = puerto_rico[(puerto_rico["bed"] == bedrooms_selected) & (puerto_rico["bath"] == bathrooms_selected)]
    
    
    
    if st.checkbox(f"Display data for the above criteria for Puerto Rico"):
        ############### Show table ###############
        st.write(f"<b>Dataframe for the above selected criteria for Puerto Rico</b>",unsafe_allow_html=True)
        st.write(f"We found <b>{filtered_puerto_rico_df.count().price}</b> property listings 🏠 matching your criteria! <br>Here's a little preview of the data ⬇️",unsafe_allow_html=True)
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
    
    
    ############### Show list of top zip codes based on overall price ###############  
    if st.checkbox(f"Display top zip codes by median price for Puerto Rico"):
        top_zipcodes = filtered_puerto_rico_df.groupby("zip_code")["price"].mean().reset_index().sort_values(by="price", ascending=False).head(15)
        top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
        top_zipcodes["price"] = top_zipcodes["price"].round(2)

        st.write("top zip codes:")
        st.write(top_zipcodes.set_index("zip_code").drop_duplicates())
    
    
    st.write("---")
    
    
    ############### Select a single zipcode from the top 15 list for further analysis ###############
    if st.checkbox(f"Display price predictions for {bedrooms_selected} bedroom(s) & {bathrooms_selected} bathroom(s) in  the top 15 Puerto Rico zipcodes"):
        ############### Check if top_zipcodes is empty ###############
        if top_zipcodes.empty:
            st.write("Please select the Display data checkbox above to populate top zip codes.")
        else:
            zip_selected = st.selectbox(f"Select from the top zip codes in Puerto Rico for price prediction", top_zipcodes["zip_code"].unique())

            ############### Filter the dataframe by the selected zip code ###############
            data = filtered_puerto_rico_df[filtered_puerto_rico_df['zip_code'] == zip_selected]
            data = data.drop_duplicates()
            st.write(f"Number of available properties in <b>{zip_selected}</b> zip code: <b>{len(data)}</b>",unsafe_allow_html=True)
            st.write(data)
            
    
            
            st.write("---")
        
        
        
            ############### Training & Testing - Split data into input (X) and output (y) variables ###############
            X = filtered_puerto_rico_df.drop(["price"], axis=1)
            y = filtered_puerto_rico_df["price"]

            
            ############### Add Title for Model Training ###############

            
            if st.button(f"Run price prediction for Puerto Rico zip: {zip_selected}"):
                
                ############### Split data into training and testing sets ###############
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

                ############### Initialize the model ###############
                model = BaggingRegressor()


                ############### Train the model ###############
                model.fit(X_train, y_train)

                ############### Test the model ###############
                y_pred = model.predict(X_test)

                ############### Calculate scoring metric ###############
                score = mean_absolute_error(y_test, y_pred)

                ############### Display results ###############
                st.write(f"Mean Absolute Error Score (MAE) + BaggingRegressor(): <b>{score:.2f}</b>",unsafe_allow_html=True)
                st.balloons()
                st.write(f"<b>🚧 UNDER CONSTRUCTION: Add metrics and price prediction explainations 🚧</b>",unsafe_allow_html=True)
    
    
                


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
    
    
    
    st.write("---")
        
        
        
    ############### Create new datafram - Filter data based on user selections ###############
    filtered_virgin_islands_df = virgin_islands[(virgin_islands["bed"] == bedrooms_selected) & (virgin_islands["bath"] == bathrooms_selected)]
    
    
    
    if st.checkbox(f"Display data for the above criteria for U.S. Virgin Islands"):
        ############### Show table ###############
        st.write(f"<b>Dataframe for the above selected criteria for U.S. Virgin Islands</b>",unsafe_allow_html=True)
        st.write(f"We found <b>{filtered_virgin_islands_df.count().price}</b> property listings 🏠 matching your criteria! <br>Here's a little preview of the data ⬇️",unsafe_allow_html=True)
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
    
    
    ############### Show list of top zip codes based on overall price ###############  
    if st.checkbox(f"Display top zip codes by median price for U.S. Virgin Islands"):
        top_zipcodes = filtered_virgin_islands_df.groupby("zip_code")["price"].mean().reset_index().sort_values(by="price", ascending=False).head(15)
        top_zipcodes["zip_code"] = top_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
        top_zipcodes["price"] = top_zipcodes["price"].round(2)

        st.write("top zip codes:")
        st.write(top_zipcodes.set_index("zip_code").drop_duplicates())
    
    
    st.write("---")
    
    
    ############### Select a single zipcode from the top 15 list for further analysis ###############
    if st.checkbox(f"Display price predictions for {bedrooms_selected} bedroom(s) & {bathrooms_selected} bathroom(s) in  the top 15 U.S. Virgin Islands zipcodes"):
        ############### Check if top_zipcodes is empty ###############
        if top_zipcodes.empty:
            st.write("Please select the Display data checkbox above to populate top zip codes.")
        else:
            zip_selected = st.selectbox(f"Select from the top zip codes in U.S. Virgin Islands for price prediction", top_zipcodes["zip_code"].unique())

            ############### Filter the dataframe by the selected zip code ###############
            data = filtered_virgin_islands_df[filtered_virgin_islands_df['zip_code'] == zip_selected]
            data = data.drop_duplicates()
            st.write(f"Number of available properties in <b>{zip_selected}</b> zip code: <b>{len(data)}</b>",unsafe_allow_html=True)
            st.write(data)
            
    
            
            st.write("---")
        
        
        
            ############### Training & Testing - Split data into input (X) and output (y) variables ###############
            X = filtered_virgin_islands_df.drop(["price"], axis=1)
            y = filtered_virgin_islands_df["price"]

            
            ############### Add Title for Model Training ###############

            
            if st.button(f"Run price prediction for U.S. Virgin Islands zip: {zip_selected}"):
                
                ############### Split data into training and testing sets ###############
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

                ############### Initialize the model ###############
                model = GradientBoostingRegressor()


                ############### Train the model ###############
                model.fit(X_train, y_train)

                ############### Test the model ###############
                y_pred = model.predict(X_test)

                ############### Calculate scoring metric ###############
                score = mean_absolute_error(y_test, y_pred)

                ############### Display results ###############
                st.write(f"Mean Absolute Error (MAE) Score + GradientBoostingRegressor(): <b>{score:.2f}</b>",unsafe_allow_html=True)
                st.balloons()
                st.write(f"<b>🚧 UNDER CONSTRUCTION: Add metrics and price prediction explainations 🚧</b>",unsafe_allow_html=True)
    
    
    
#######################################################################################################################    