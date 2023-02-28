import streamlit as st
import pandas as pd

import lazypredict as lp # which models works better without any parameter tuning


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
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# load css file
local_css("./style/style.css")


#------------------------------------------------------------------#

# Read ticker symbols from a CSV file
try:
    data = pd.read_csv("./Resources/data.csv")
except:
    logging.error('Cannot find the CSV file')

# Title/ header
st.header("Real estate price predictor")
st.write("Select from different Machine Learning models to view the best housing predictor for your budget")
st.write("---")


#------------------------------------------------------------------#

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Original data", "Data cleanup", "Lazypredict", "ML models", "Recommendations", "U.S. Mainland", "Puerto Rico", "U.S. Virgin Islands"])

with tab1:
    
    # Show tickers list
    if st.checkbox("View Original dataset"):
        st.write(data)
        st.write("---")

    if st.checkbox("View Original datatypes"):
        st.write(data.dtypes)

        
#------------------------------------------------------------------#        

with tab2:
    if st.checkbox("View refactored data"):
        # Drop columns
        data = data.drop(columns=["sold_date", "status", "full_address", "street"])

        # Drop rows with NaN values in the zip_code column
        data = data.dropna(subset=["zip_code"])

        # Convert the zip_code column to string with leading zeros
        data["zip_code"] = data["zip_code"].apply(lambda x: '{0:0>5}'.format(int(x)))

        # Show tickers list
        st.write(data)
        st.write("---")
        
    if st.checkbox("View refactored datatypes"):
        st.write(data.dtypes)
        st.write("---")

    if st.checkbox("View states/ territories"):    
        # View all unique state values
        states = data["state"].unique()
        st.write(states)
        st.write("---")

    if st.checkbox("View grouped datasets"):    
        
        # Create a new dataframe for Puerto Rico and Virgin Islands
        pr_vi_data = data[(data["state"] == "Puerto Rico") | (data["state"] == "Virgin Islands")]

        # Create a second dataframe for the rest of the values in the state column
        mainland_data = data[(data["state"] != "Puerto Rico") & (data["state"] != "Virgin Islands")]

        # Display the resulting dataframes
        st.write(f"<b>U.S. Territories</b>",unsafe_allow_html=True)
        st.write(pr_vi_data)
        st.write(f"<b>U.S. Mainland states</b>",unsafe_allow_html=True)
        st.write(mainland_data)

#------------------------------------------------------------------#        
 
## LAZYPREDICT ##

with tab3:
    if st.checkbox("View Lazypredict data info"):
        
        # Classification
        st.write("TESTING...")
        
        from sklearn.model_selection import train_test_split
        from lazypredict.Supervised import LazyClassifier
        
        X = data.drop(columns=["price"])
        st.write(X)
        y = data["price"]
        st.write(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =1)

        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        models,predictions = clf.fit(X_train, X_test, y_train, y_test)

        st.write(models)
        
        # Regression
        
        from lazypredict.Supervised import LazyRegressor
        from sklearn.utils import shuffle
        import numpy as np


        X, y = shuffle(data.data, data.target, random_state=1)
        X = X.astype(np.float32)

        offset = int(X.shape[0] * 0.9)

        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]

        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)

        st.write(models)




#------------------------------------------------------------------#        
        
with tab4:
    if st.checkbox("ML Models implementation"):
        st.write (f"<b>WIP...⏳</b>",unsafe_allow_html=True)
    

#------------------------------------------------------------------#

with tab5:
    if st.checkbox("Recommendations"):
        st.write (f"<b>WIP...⏳</b>",unsafe_allow_html=True)
        option = st.selectbox(
        'Which recommendation would you like to view?',
        ('Calculate the rental return by metro in comparison to home values', 
         'Markets with CAP rate from 0.7% and above', 
         'Appreciation potential till 2025 from 5% and above'))

    st.write('You selected:', option)
    


#------------------------------------------------------------------#

with tab6:
    if st.checkbox("U.S. Mainland Recommendations"):
        st.write (f"<b>WIP...⏳</b>",unsafe_allow_html=True)
    


#------------------------------------------------------------------#

with tab7:
    if st.checkbox("Puerto Rico Recommendations"):
        st.write (f"<b>WIP...⏳</b>",unsafe_allow_html=True)
        

#------------------------------------------------------------------#

with tab8:
    if st.checkbox("U.S. Virgin Islands Recommendations"):
        st.write (f"<b>WIP...⏳</b>",unsafe_allow_html=True)
        
#------------------------------------------------------------------#