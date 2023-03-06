# Real-estate-price-predictor

Real estate price predictor using Machine Learning models - U.S. Mainland, Puerto Rico & U.S. Virgin Islands


<img
  src="./Resources/home.png"
  alt="Real Estate price predictor"
  title="Real Estate price predictor"
  style="display: inline-block; margin: 0 auto; max-width: 75px">
  

## Business Case

A real estate company wants to predict the selling prices of houses based on various features such as the number of bedrooms, bathrooms, square footage, and location. They have collected data on past house sales, which includes these features as well as the final selling price. The company wants to use this data to train a machine learning model that can accurately predict the selling price of a house given its features. They also want to be able to evaluate the performance of different models using different scoring metrics.


  
  
## Objective

## Technologies

- `python`
- `anaconda`
- `pandas    : 1.3.5`
- `matplotlib: 3.5.3`
- `streamlit : 1.18.1`

## Environment setup

- `conda create -n [name] python=3.9`
- `conda activate [name]`
- `git clone` repo
- `pip install -r requirements.txt`


## Deployment

- In Terminal `cd` into cloned repo
- `cd` to directory where `home.py` file is located
- `streamlit run home.py`
- The app will run on `http://localhost:8501/`


## Contributors

[Christine Pham](https://github.com/cpham35?tab=repositories) - `cpham35`

[Kranthi C Mitta](https://github.com/kranthicmitta?tab=repositories) - `kranthicmitta` 

[Ayush Jha](https://github.com/jha-ayush?tab=repositories) - `jha-ayush`


## Summary
The provided codebase can be used to calculate different scoring metrics for a machine learning model that predicts the selling price of houses based on various features. The codebase takes as input the actual selling prices of a test set of houses and the predicted selling prices of the same test set of houses using a machine learning model. Based on the selected scoring metric, the codebase calculates the corresponding score and assigns it to the variable "score". The scoring metrics that can be selected are "r2_score", "mean_squared_error", and "mean_absolute_error". This codebase can be used to evaluate the performance of different machine learning models that predict the selling price of houses.



## Next steps


<sub>Note: This app is for educational purposes only.</sub>