############### Real-estate predictor data app using Machine Learning for regions like - U.S. Mainland state, Puerto Rico & U.S. Virgin Islands ###############

############### Import Librarires ###############
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt # Visualizations

from sklearn.model_selection import train_test_split # Train/ Test package
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error # Scoring metrics
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression # Machine Learning Models
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor # Regression ML Models
import xgboost as xgb # Regression ML Model
from sklearn.neighbors import KNeighborsRegressor

import scipy.stats as stats #Significance scoring

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
mainland = load_data("../Resources/data_files/mainland_data.csv")
puerto_rico = load_data("../Resources/data_files/puerto_rico_data.csv")
virgin_islands = load_data("../Resources/data_files/virgin_islands_data.csv")

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
st.markdown(f"This project involves analyzing real estate data from Kaggle for the mainland United States, Puerto Rico, and the US Virgin Islands to predict property prices. The data is cleaned, features are engineered, and a variety of regression models are tested with RMSE scoring metrics to find the best one for each region. The analysis includes visualizing all available properties for a given location and highlights future directions for the project.",unsafe_allow_html=True)
st.write("---")


#######################################################################################################################


############### Initial app in Streamlit tab format ###############
tab1, tab2, tab3, tab4 = st.tabs(["U.S. Mainland", "Puerto Rico", "U.S. Virgin Islands", "Next steps"])


#######################################################################################################################


############### U.S. Mainland ###############
with tab1:
      
    st.subheader("U.S. Mainland data")
    
    ############### Define a dictionary to map states to U.S. mainland ###############
    state = [" Select a U.S. mainland state", "Connecticut", "Delaware", "Maine", "Massachusetts", "New Hampshire", "New Jersey", "New York", "Pennsylvania", "Rhode Island", "Vermont", "West Virginia", "Wyoming"]
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
    top_mainland_zipcodes = filtered_mainland_df.groupby("zip_code")["price"].median().reset_index().sort_values(by="price", ascending=False)
    top_mainland_zipcodes["zip_code"] = top_mainland_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
    top_mainland_zipcodes["price"] = top_mainland_zipcodes["price"].round(2)

    st.write(f"<b>Here is a list of all the zip codes by median descening price in {state_selected} for the above {bedrooms_selected} bed & {bathrooms_selected} bath criteria ⬇️</b>",unsafe_allow_html=True)
    st.write(top_mainland_zipcodes.set_index("zip_code").drop_duplicates().sort_values(by="price", ascending=False))

    
    
    st.write("---")
    
    
    ############### Show property listings of a selected zipcode from the top list ###############
    
    ############### Create dropdown to select a specific zipcode ###############
    selected_mainland_zipcode = st.selectbox("Select a zipcode from the above top list", top_mainland_zipcodes["zip_code"])
    
    ############### Create & display dataframe for selected zipcode ###############
    selected_mainland_zipcode_df = filtered_mainland_df[filtered_mainland_df["zip_code"] == int(selected_mainland_zipcode)].sort_values(by="price", ascending=False)
    


    ############### Add new column with labels ###############
    selected_mainland_zipcode_df["label"] = [f"Home {i+1}" for i in range(len(selected_mainland_zipcode_df))]

    ############### Drop state & zip_code columns ###############
    selected_mainland_zipcode_df = selected_mainland_zipcode_df.drop(columns=["state", "zip_code"], axis=1)

    
    
    ############### Title ###############
    st.write(f"<b>Here is a list of {len(selected_mainland_zipcode_df)} property listings for the zipcode {selected_mainland_zipcode} in {state_selected}:</b>",unsafe_allow_html=True)
    
    
    
    
    ############### Re-arrange columns & Display ###############
    selected_mainland_zipcode_df = selected_mainland_zipcode_df.reindex(columns=["label", "house_size", "bed", "bath", "acre_lot", "price"])
    ############### Set index column to `label`
    selected_mainland_zipcode_df = selected_mainland_zipcode_df.reset_index(drop=True)

    st.write(selected_mainland_zipcode_df)

    
    ############### Display bar chart ###############
    
    labels=selected_mainland_zipcode_df["label"].values
    price=selected_mainland_zipcode_df["price"].values
    
    
    ############### Create a bar chart
    fig_prop_listings, ax = plt.subplots()
    ax.bar(labels, price)
    ax.set_xlabel("Property label")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"Listings preview for zipcode {selected_mainland_zipcode}, {state_selected}")
    # rotate the x-axis tick labels
    ax.set_xticklabels(labels, rotation=45, ha="right")
    # Set X-axis to 1, instead of 0
    #ax.set_xlim(0.5, len(top_mainland_zipcodes)+0.5)

    ############### Display the chart in Streamlit
    st.pyplot(fig_prop_listings)
        
    
    st.write("---")
    
    

    ############### Add Title for Model Training ###############

    if st.button(f"Run price prediction ML models for {selected_mainland_zipcode} zipcode"):


        ############### Training & Testing - Split data into input (X) and output (y) variables 
        predictors = ["house_size", "bed", "bath", "acre_lot"] # Add additional features 
        X = selected_mainland_zipcode_df[predictors]
        y = selected_mainland_zipcode_df["price"]

        ############### Split data into training and testing sets ###############
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


        ############### Define the models ###############
        models = [
            RandomForestRegressor(random_state=10),      
            BaggingRegressor(random_state=10),
            GradientBoostingRegressor(random_state=10),
            Ridge(random_state=10),
            ElasticNet(random_state=10),
            xgb.XGBRegressor(seed=10),
            LinearRegression()
                 ]

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
            # Refactor to use RMSE
            rmse = np.sqrt(score)
            # st.write(f"RMSE",rmse)

            ############### Find best score and best model
            if score < best_score:
                best_score = score
                best_model = model


        
        ############### assuming X_test is your test data and y_test is your test target
        price_predictions_mainland = best_model.predict(selected_mainland_zipcode_df[predictors])


        ############### create a new dataframe with a new column for the predicted values
        price_predictions_mainland_df = selected_mainland_zipcode_df.copy()
        price_predictions_mainland_df["predictions"]= price_predictions_mainland
        
        
        ############### Display final predicted pricings
        st.write(price_predictions_mainland_df.round(2))
        # st.write(price_predictions_df.sort_values(by="price_predictions", ascending=False))
        
        
        
        
        
        ############### Conformal prediction theory implication ###############
        
        
        ############### Calculate the t and p-values for the linear regression coefficients
        _, p_values = stats.ttest_ind(price_predictions_mainland_df["price"], price_predictions_mainland_df["predictions"])

        ############### Set the significance level
        alpha = 0.05

        ############### Check if any of the p-values are less than the significance level
        if (p_values < alpha).any():
            st.write(f"<b>Prediction confidence at 90% significance level</b> - At least one coefficient is statistically significant. 🥳",unsafe_allow_html=True)
        else:
            st.write(f"<b>Prediction confidence at 90% significance level</b> - None of the coefficients are statistically significant. 👎",unsafe_allow_html=True)

            
        
        
        st.balloons()
        
        
                
#######################################################################################################################                
                
############### Puerto Rico ###############
with tab2:

    st.subheader("Puerto Rico")
    
    ############### Display number of bedrooms dropdown menu ###############
    bedrooms_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    bedrooms_selected = st.selectbox(f"Select number of bedroom(s) in the territory of Puerto Rico", bedrooms_options)


    ############### Display number of bedrooms dropdown menu ###############
    bathrooms_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    bathrooms_selected = st.selectbox(f"Select number of bathroom(s) in the territory of Puerto Rico", bathrooms_options)


        
    ############### Create new datafram - Filter data based on user selections ###############
    filtered_puerto_rico_df = puerto_rico[(puerto_rico["bed"] == bedrooms_selected) & (puerto_rico["bath"] == bathrooms_selected)]
    
    
    ############### Show list of top zip codes based on Median price ###############
    top_puerto_rico_zipcodes = filtered_puerto_rico_df.groupby("zip_code")["price"].median().reset_index().sort_values(by="price", ascending=False)
    top_puerto_rico_zipcodes["zip_code"] = top_puerto_rico_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
    top_puerto_rico_zipcodes["price"] = top_puerto_rico_zipcodes["price"].round(2)

    st.write(f"<b>Here is a list of all the zip codes by median descening price in PR for the above {bedrooms_selected} bed & {bathrooms_selected} bath criteria ⬇️</b>",unsafe_allow_html=True)
    st.write(top_puerto_rico_zipcodes.set_index("zip_code").drop_duplicates().sort_values(by="price", ascending=False))

    
    
    st.write("---")
    
    
    
    ############### Show property listings of a selected zipcode from the top list ###############
    
    ############### Create dropdown to select a specific zipcode ###############
    selected_puerto_rico_zipcode = st.selectbox("Select a zipcode from the above top list", top_puerto_rico_zipcodes["zip_code"])
    
    ############### Create & display dataframe for selected zipcode ###############
    selected_puerto_rico_zipcode_df = filtered_puerto_rico_df[filtered_puerto_rico_df["zip_code"] == int(selected_puerto_rico_zipcode)].sort_values(by="price", ascending=False)
    


    ############### Add new column with labels ###############
    selected_puerto_rico_zipcode_df["label"] = [f"Home {i+1}" for i in range(len(selected_puerto_rico_zipcode_df))]

    ############### Drop state & zip_code columns ###############
    selected_puerto_rico_zipcode_df = selected_puerto_rico_zipcode_df.drop(columns=["zip_code"], axis=1)

    
    
    ############### Title ###############
    st.write(f"<b>Here is a list of {len(selected_puerto_rico_zipcode_df)} property listings for the zipcode {selected_puerto_rico_zipcode} in Puerto Rico:</b>",unsafe_allow_html=True)
    
    
    
    
    ############### Re-arrange columns & Display ###############
    selected_puerto_rico_zipcode_df = selected_puerto_rico_zipcode_df.reindex(columns=["label", "house_size", "bed", "bath", "acre_lot", "price"])
    ############### Set index column to `label`
    selected_puerto_rico_zipcode_df = selected_puerto_rico_zipcode_df.reset_index(drop=True)

    st.write(selected_puerto_rico_zipcode_df)

    
    ############### Display bar chart ###############
    
    labels=selected_puerto_rico_zipcode_df["label"].values
    price=selected_puerto_rico_zipcode_df["price"].values
    
    
    ############### Create a bar chart
    fig_prop_listings, ax = plt.subplots()
    ax.bar(labels, price)
    ax.set_xlabel("Property label")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"Listings preview for zipcode {selected_puerto_rico_zipcode}")
    
    # rotate the x-axis tick labels
    ax.set_xticklabels(labels, rotation=45, ha="right")


    ############### Display the chart in Streamlit
    st.pyplot(fig_prop_listings)
    
        
    
    st.write("---")
    
    
    
    ############### Add Title for Model Training ###############

    if st.button(f"Run price prediction ML models for {selected_puerto_rico_zipcode} zipcode"):
        
        
        ############### Training & Testing - Split data into input (X) and output (y) variables 
        predictors = ["house_size", "bed", "bath", "acre_lot"] # Add additional features 
        X = selected_puerto_rico_zipcode_df[predictors]
        y = selected_puerto_rico_zipcode_df["price"]

        ############### Split data into training and testing sets ###############
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


        ############### Define the models ###############
        model = BaggingRegressor()

        ############### Train the model ###############
        model.fit(X_train, y_train)

        ############### Test the model ###############
        y_pred = model.predict(X_test)


        # Use Mean Squared Error metrics for scoring metrics
        score = mean_squared_error(y_test, y_pred)
        # st.write(f"Score",score)
        # Refactor to use RMSE
        rmse = np.sqrt(score)
        # st.write(f"RMSE",rmse)


        
        ############### assuming X_test is your test data and y_test is your test target
        price_predictions_puerto_rico = model.predict(selected_puerto_rico_zipcode_df[predictors])


        ############### create a new dataframe with a new column for the predicted values
        price_predictions_puerto_rico_df = selected_puerto_rico_zipcode_df.copy()
        price_predictions_puerto_rico_df["predictions"]= price_predictions_puerto_rico
        
        
        ############### Display final predicted pricings
        st.write(price_predictions_puerto_rico_df.round(2))
        
        
        
        ############### Conformal prediction theory implication ###############
        
        
        ############### Calculate the t and p-values for the linear regression coefficients
        _, p_values = stats.ttest_ind(price_predictions_puerto_rico_df["price"], price_predictions_puerto_rico_df["predictions"])

        ############### Set the significance level
        alpha = 0.05

        ############### Check if any of the p-values are less than the significance level
        if (p_values < alpha).any():
            st.write(f"<b>Prediction confidence at 90% significance level</b> - At least one coefficient is statistically significant. 🥳",unsafe_allow_html=True)
        else:
            st.write(f"<b>Prediction confidence at 90% significance level</b> - None of the coefficients are statistically significant. 👎",unsafe_allow_html=True)
        
        
        
        
        st.balloons()
    
                


#######################################################################################################################           
            

############### U.S. Virgin Islands ###############
with tab3:
    
    st.subheader("U.S. Virgin Islands")
    
    ############### Display number of bedrooms dropdown menu ###############
    bedrooms_options = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    bedrooms_selected = st.selectbox(f"Select number of bedroom(s) in the territory of U.S. Virgin Islands", bedrooms_options)


    ############### Display number of bedrooms dropdown menu ###############
    bathrooms_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    bathrooms_selected = st.selectbox(f"Select number of bathroom(s) in the territory of U.S. Virgin Islands", bathrooms_options)
        
        
        
    ############### Create new datafram - Filter data based on user selections ###############
    filtered_virgin_islands_df = virgin_islands[(virgin_islands["bed"] == bedrooms_selected) & (puerto_rico["bath"] == bathrooms_selected)]
    
    
    ############### Show list of top zip codes based on Median price ###############
    top_virgin_islands_zipcodes = filtered_virgin_islands_df.groupby("zip_code")["price"].median().reset_index().sort_values(by="price", ascending=False)
    top_virgin_islands_zipcodes["zip_code"] = top_virgin_islands_zipcodes["zip_code"].apply(lambda x: '{0:0>5}'.format(x))
    top_virgin_islands_zipcodes["price"] = top_virgin_islands_zipcodes["price"].round(2)

    st.write(f"<b>Here is a list of all the zip codes by median descening price in U.S. VI for the above {bedrooms_selected} bed & {bathrooms_selected} bath criteria ⬇️</b>",unsafe_allow_html=True)
    st.write(top_virgin_islands_zipcodes.set_index("zip_code").drop_duplicates().sort_values(by="price", ascending=False))

    
    
    st.write("---")
    
    
    
    ############### Show property listings of a selected zipcode from the top list ###############
    
    ############### Create dropdown to select a specific zipcode ###############
    selected_virgin_islands_zipcode = st.selectbox("Select a zipcode from the above top list", top_virgin_islands_zipcodes["zip_code"])
    
    ############### Create & display dataframe for selected zipcode ###############
    selected_virgin_islands_zipcode_df = filtered_virgin_islands_df[filtered_virgin_islands_df["zip_code"] == int(selected_virgin_islands_zipcode)].sort_values(by="price", ascending=False)
    


    ############### Add new column with labels ###############
    selected_virgin_islands_zipcode_df["label"] = [f"Home {i+1}" for i in range(len(selected_virgin_islands_zipcode_df))]

    ############### Drop state & zip_code columns ###############
    selected_virgin_islands_zipcode_df = selected_virgin_islands_zipcode_df.drop(columns=["zip_code"], axis=1)

    
    
    ############### Title ###############
    st.write(f"<b>Here is a list of {len(selected_virgin_islands_zipcode_df)} property listings for the zipcode {selected_virgin_islands_zipcode} in U.S. Virgin Islands:</b>",unsafe_allow_html=True)
    
    
    
    
    ############### Re-arrange columns & Display ###############
    selected_virgin_islands_zipcode_df = selected_virgin_islands_zipcode_df.reindex(columns=["label", "house_size", "bed", "bath", "acre_lot", "price"])
    ############### Set index column to `label`
    selected_virgin_islands_zipcode_df = selected_virgin_islands_zipcode_df.reset_index(drop=True)

    st.write(selected_virgin_islands_zipcode_df)

    
    ############### Display bar chart ###############
    
    labels=selected_virgin_islands_zipcode_df["label"].values
    price=selected_virgin_islands_zipcode_df["price"].values
    
    
    ############### Create a bar chart
    fig_prop_listings, ax = plt.subplots()
    ax.bar(labels, price)
    ax.set_xlabel("Property label")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"Listings preview for zipcode {selected_virgin_islands_zipcode}")
    
    # rotate the x-axis tick labels
    ax.set_xticklabels(labels, rotation=45, ha="right")


    ############### Display the chart in Streamlit
    st.pyplot(fig_prop_listings)
    
        
    
    st.write("---")
    
    
    
    ############### Add Title for Model Training ###############

    if st.button(f"Run price prediction ML models for {selected_virgin_islands_zipcode} zipcode"):
        
        
        ############### Training & Testing - Split data into input (X) and output (y) variables 
        predictors = ["house_size", "bed", "bath", "acre_lot"] # Add additional features 
        X = selected_virgin_islands_zipcode_df[predictors]
        y = selected_virgin_islands_zipcode_df["price"]

        ############### Split data into training and testing sets ###############
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


        ############### Define the models ###############
        model = GradientBoostingRegressor()

        ############### Train the model ###############
        model.fit(X_train, y_train)

        ############### Test the model ###############
        y_pred = model.predict(X_test)


        # Use Mean Squared Error metrics for scoring metrics
        score = mean_squared_error(y_test, y_pred)
        # st.write(f"Score",score)
        # Refactor to use RMSE
        rmse = np.sqrt(score)
        # st.write(f"RMSE",rmse)


        
        ############### assuming X_test is your test data and y_test is your test target
        price_predictions_virgin_islands = model.predict(selected_virgin_islands_zipcode_df[predictors])


        ############### create a new dataframe with a new column for the predicted values
        price_predictions_virgin_islands_df = selected_virgin_islands_zipcode_df.copy()
        price_predictions_virgin_islands_df["predictions"]= price_predictions_virgin_islands
        
        
        ############### Display final predicted pricings
        st.write(price_predictions_virgin_islands_df.round(2))
        
        
        
        ############### Conformal prediction theory implication ###############
        
        
        ############### Calculate the t and p-values for the linear regression coefficients
        _, p_values = stats.ttest_ind(price_predictions_virgin_islands_df["price"], price_predictions_virgin_islands_df["predictions"])

        ############### Set the significance level
        alpha = 0.05

        ############### Check if any of the p-values are less than the significance level
        if (p_values < alpha).any():
            st.write(f"<b>Prediction confidence at 90% significance level</b> - At least one coefficient is statistically significant. 🥳",unsafe_allow_html=True)
        else:
            st.write(f"<b>Prediction confidence at 90% significance level</b> - None of the coefficients are statistically significant. 👎",unsafe_allow_html=True)
        
        
        
        
        st.balloons()
    
    
    
#######################################################################################################################


############### Next steps ###############
with tab4:
    
    st.subheader("What can we do next...🚧")
    st.info("Initial data capture - better, more robust data sets, time series data, paid APIs")
    st.info("Include attributes like `price_per_sq_ft`, `address`, `date_sold`, etc.")
    st.info("Use conformal prediction to find the best ranked sore for price predictions")
    st.info("`Blockchain` implementation: By recording real estate data on a blockchain, we can create a transparent and immutable record of real estate transactions to prevent fraud and corruption, as all transactions are publicly visible and immutable or can't be altered.")
    st.info("Rental predictions based on `bed`, `bath`and `sq_footage`")
    st.info("Real-estate portfolio creation and management for investors")
    st.info("House price prediction based on `sq_footage` price using time series data")
    

    

#######################################################################################################################


