<img
  src="./Images/home.png"
  alt="Real Estate price predictor"
  title="Real Estate price predictor"
  style="display: inline-block; margin: 0 auto; max-width: 60px">

# Real-estate-price-predictor

Real estate price predictor using Machine Learning models - U.S. Mainland, Puerto Rico & U.S. Virgin Islands




## Preview


**Project flowchart**

<img
  src="./Images/app_flowchart.png"
  alt="App flowchart"
  title="App flowchart"
  style="display: inline-block; margin: 0 auto; max-width: 55px">
  


**UI flow - screenshot 01**

<img
  src="./Images/app_screenshot_01.png"
  alt="App screenshot 01"
  title="App screenshot 01"
  style="display: inline-block; margin: 0 auto; max-width: 55px">
  

**UI flow - screenshot 02**

<img
  src="./Images/app_screenshot_02.png"
  alt="App screenshot 02"
  title="App screenshot 02"
  style="display: inline-block; margin: 0 auto; max-width: 55px">
  

**UI flow - screenshot 03**

<img
  src="./Images/app_screenshot_03.png"
  alt="App screenshot 03"
  title="App screenshot 03"
  style="display: inline-block; margin: 0 auto; max-width: 55px">



## Table of contents

- [Objective](#objective)
- [Conformal prediction analysis](#conformal-prediction-analysis)
- [Preview](#preview)
- [Environment setup](#environment-setup)
- [Technologies](#technologies)
- [Deployment](#deployment)
- [Contributors](#contributors)
- [Next steps](#next-steps)
- [License](#license)



## Objective
[(Back to top)](#table-of-contents)

The objective of this project is to predict the future selling prices of houses for a real estate app. Static historical data was gathered from [Kaggle](https://www.kaggle.com/datasets), on past house sales, including the number of `bed`, `bath`, `acre_lot`, `house_size` and `state` or United States territory, along with the final selling price. 


By integrating multiple machine learning models in conjunction with Conformal prediction analysis, the app evaluates the ML performance against scoring metrics, to provide the user with model `price_predictions` future values and then evaluates via conformal analysis to inform the user whether the prediced future price has over 95% confidence level or not.



## Conformal prediction analysis
[(Back to top)](#table-of-contents)


**What is Conformal Prediction Analysis?**

[Conformal prediction analysis](https://medium.com/low-code-for-advanced-data-science/conformal-prediction-theory-explained-14a35226df80) is a machine learning framework that provides a way to assign confidence levels to individual predictions, in our case - real-estate future prices, based on a given level of significance.


**How is Conformal Prediction Analysis used in this repo?**

By using multiple regression models & scoring metrics via the [`lazypredict`](https://pypi.org/project/lazypredict/) python package, we first selected the top ML models results that were ranked based on their performance against `R^2` & `RMSE` scoring metrics, from `lazypredict` analysis.


After top ML model selection(s), the `nonconformist` python package assigned a confidence level to each prediction based on the error rate and the desired level of significance (95%).

By selecting the model with the highest confidence level for each prediction, we were able to provide price predictions and then assess those predictions against the 95% price prediction confidence level.




## Environment setup
[(Back to top)](#table-of-contents)

- `conda create -n [name] python=3.9`
- `conda activate [name]`
- `git init` inside a directory where you want to save this repo
- `git clone` repo
- `pip install -r requirements.txt`


## Technologies
[(Back to top)](#table-of-contents)

`python`, `anaconda`, `numpy`, `matplotlib`, `pandas`, `streamlit`, `sklearn`




## Deployment
[(Back to top)](#table-of-contents)


**Localhost**

- In Terminal `cd` into cloned repo
- `cd` to `src` directory where `home.py` file is located
- `streamlit run home.py`
- The app will run on `http://localhost:8501/`



## Contributors
[(Back to top)](#table-of-contents)

[Christine Pham](https://github.com/cpham35?tab=repositories) - `cpham35`

[Kranthi C Mitta](https://github.com/kranthicmitta?tab=repositories) - `kranthicmitta` 

[Ayush Jha](https://github.com/jha-ayush?tab=repositories) - `jha-ayush`



## Next steps
[(Back to top)](#table-of-contents)

- Initial data capture - better, more robust data sets, time series data, paid APIs.
- Include attributes like `sq_ft`, `price_per_sq_ft`, `address`, `date_sold`, etc.
- Use enhanced conformal prediction to find the best ranked sore for price predictions.
- `Blockchain` implementation: By recording real estate data on a blockchain, we can create a transparent and immutable record of real estate transactions to prevent fraud and corruption, as all transactions are publicly visible and immutable or can't be altered.
- Rental predictions based on `bed`, `bath` and `sq_ft`.
- Real-estate portfolio creation and management for investors.



## License
[(Back to top)](#table-of-contents)

**MIT License**

Copyright ??  [2023]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



<sub>**Note:** This app is for educational purposes only.</sub>