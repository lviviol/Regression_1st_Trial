### Regression Modelling (1st Trial)
* This exercise trial attempts to predict public housing prices using regression model based on historical data.   

### Purpose
* To practice applying data science libraries and making use of suitable models to predict housing prices results.  
* To present insightful information that enable buyers, policy makers and investors towards making informed decisions.


### Data Science
<details> 
  <summary>Libraries</summary>  
Python, Pandas, Matplotlib, Plotly, Scikit-Learn
</details>

### Dataset
<details> 
  <summary>Variables & Source</summary>  
  
* Dataset comprises of variables such as period, town, housing type, storey range, floor area, lease and prices.
* Data sources is Singapore's Open Data Portal
</details>

### Scope
<details> 
  <summary>Period & Predictions</summary>  
  
* Period: Historical housing prices spans between 2017-01 & 2025-02 (YYYY-MM)
* Prediction: Up to 2026-12
* Selected Variables: We begin this data modelling exercise with 1 variable, prices.
</details>

### Methodology & Implementation
<details> 
  <summary>Experimental</summary>  
  
* We examine all data, including outliers.
* We check and exclude missing data to minimize bias in the analysis
* Data is split into training (2017-01 - 2024-10) and testing (2024-11 - 2025-02), i.e. using quarterly data to test the model.
* Random Forest Regression is selected since housing price is a time series data.
* The model predicts housing prices
</details>

### Results & Visualization
  
  1. OOB = 13.2% & R-Square = -0.026
  2. This means the model accurate predicts prices only 13.2% of the time.
  3. Negative correlation shows the model is performing worse than a simple mean predictions

![alt text](https://github.com/lviviol/Regression_Modelling_Trial/blob/main/LinePlot.png?raw=true)
 
### Future Improvements
<details> 
  <summary>Ideas & Plan</summary>  

* We plan to include more variables into the modelling.
* We continue to use OOB > 80% and R-Square > 0.8 as guiding criteria for verifying the model's predicted prices.
* We are researching into a more robust model in place of Random Forest Regression model.
</details>


### Appendix
<details> 
  <summary>Disclaimer</summary>
This project is currently in early stage of development and analysis.  The predictions and insights generated are preliminary and should not be used for financial, investment and policy decisions.  Contributions and feedback are welcome as this analysis evolves. 
</details>


<details> 
  <summary>Credits & Acknowledgements</summary>

* Model Selection
  1. IBM, https://www.ibm.com/think/topics/random-forest
  2. ZiZheng Li, https://www.researchgate.net/publication/383112591_A_Comparative_Study_of_Regression_Models_for_Housing_Price_Prediction

* Learning Tutorial
  1. Rob Mulla, https://www.youtube.com/watch?v=vV12dGe_Fho
  2. Om Pramod, https://medium.com/@ompramod9921/random-forests-32be04c8bf76
  
* Data Source
  1. Open data portal, https://data.gov.sg
</details>

### Charts
[View Interactive Chart](https://lviviol.github.io/Regression_1st_Trial/Regression1stTrial.html) 	







