![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png)
# Project 4: Chicago West Nile Virus Prediction

### **DSI-23: Ray Tan, Ash Ang, Timothy Chan, Joey Kang**

### Executive Summary
The West Nile Virus (WNV) is a mosquito-borne disease commonly transmitted to humans by the bite of an infected mosquitos. Weather conditions including temperature, relative humidity, precipitation and wind may affect the survival and reproduction rates of mosquitos, increasing the chances of spreading the WNV. Given that the WNV is a key public health challenge in the United States, we aim to train a classification model based on the weather conditions, species of mosquitos and locations of the traps to predict whether the WNV would be present at a specific location, which will support us in developing an effective plan to deploy pesticides throughout the city. A total of 4 models were evaluated - (i) Logistic Regression; (ii) Support Vector Machine; (iii) K-Nearest Neighbors; and (iv) Random Forest Classifier. A summary of the results are as follows. 

| Optimized Model          |Training Accuracy | Testing Accuracy | Training Recall Score | Testing Recall Score |AUC Score |
|:-----------------------:|:----------------:|:----------------:|:------------:|:---------:|:---------:|
| Logistic Regression     |      0.640       |      0.657       |    0.816     |   0.818 |     0.79    |
| Support Vector Machine  |      0.831       |      0.781       |    0.991     | 0.562  |     0.78    | 
| K-Nearest Neighbors     |      0.947       |      0.947       |    0.009     | 0.000  |     0.73    |
| Random Forest Classifier|      0.901       |      0.863       |    0.772     | 0.307  |     0.68    |


We decided on the Logistic Regression model as it has the highest AUC and recall score. Optimizing recall score means that our model is able to minimize False Negatives which is critical as the impact of incorrectly predicting an area without WNV when it actually has could be detrimental to the health of the people if measures to curb the WNV were not implemented. We also conducted a cost-benefit analysis where cost is defined as the total expenditure associated with spraying pesticide on adult mosquitoes in incurred in a year, while benefits will be measured by the cost avoidance or savings in the form of healthcare costs and productivity lost associated with the potential reduction in number of human WNV cases from spraying pesticide in Chicago. Based on our analysis, we strongly recommend IDPH to control mosquitoes population in Chicago by spraying pesticide in a more targeted fashion. In conclusion, we think that the model can give us a good prediction of the areas that have a higher presence of WNV so that we can target the spraying of pesticide at these high-risk areas more regularly to control the mosquito population and reduce the presence of WNV. To take this project further, we could explore improving the model by performing over-sampling on the minority class (positive cases). We could also factor in the benefit derived (in terms of the cost saved) not just from reducing cases of WNV but also other forms of mosquito-borne diseases such as Zika, Chikungunya, dengue, and malaria. Lastly, the predictive model can be modified and applied to help predict and direct spray efforts in other cities of the USA.

### Problem Statement 
We work for Disease And Treatment Agency, in the division of Societal Cures In Epidemiology and New Creative Engineering (DATA-SCIENCE). Due to the recent epidemic of West Nile Virus in the Windy City, the Illinois Department of Public Health (IDPH) has set up a surveillance and control system. Pesticides are a necessary evil in the fight for public health and safety, not to mention expensive. As part of setting up the control system, IDPH has engaged our agency to devise a cost-effective plan to deploy pesticides throughout the city. A cost-effective plan will provide insights for IDPH to make sound funding and policy decisions in combating WNV in Chicago.

Using weather, location, testing, and spraying data, we set out to achieve the following:
1. Create a few classification models, including Logistic Regression, K-Nearest Neighbour, Random Forest Classifier, Linear Support Vector Machine (SVM) to predict when and where different species of mosquitos will test positive for West Nile virus.
2. Evaluate the performance of the models using area under curve (AUC), accuracy and recall as the key metrics, and recommend a suitable model for prediction. A suitable model is one which outperforms the baseline model with higher AUC, accuracy and recall. 
3. Perform a cost-benefit analysis to determine the potential trade-off between spraying pesticide and the number of human WNV cases in Chicago.
4. Recommend a cost-effective plan to guide when, where and how much pesticide to spray to minimize incidence of WNV in mosquitoes.

### Background & Research
It was summer 1999 in New York City. There was an outbreak of a human virus which caused brain damage that can lead to death. Crows were seen falling from the sky. Many exotic birds at a zoological park were suddenly dying [(source)](https://science.sciencemag.org/content/286/5448/2333.abstract). These scenes marked the beginning of the West Nile Virus (WNV) in the United States (U.S.). The WNV is a mosquito-borne disease and is commonly transmitted to humans by the bite of an infected mosquito. It cannot be spread from human to human.

Today, it has become the leading cause of mosquito-borne disease and a key public health challenge in the United States. Around 80% of those infected typically display few or no symptoms. About 20% of infected people develop a fever, vomiting, or a rash. In the remaining less than 1% of those infected suffer from neuroinvasive disease (e.g encephalitis or meningitis). The mortality rate among those who suffer from neuroinvasive disease is estimated to be 10% [(source)](https://www.cdc.gov/westnile/index.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fwestnile%2Ffaq%2FgenQuestions.html). From 1999 to 2012, it is estimated that a cumulative of 778 million USD in health care expenditures and lost productivity were incurred from hospitalized cases [(source)](https://www.ajtmh.org/view/journals/tpmd/90/3/article-p402.xml).

Infection with West Nile Virus is seasonal in temperate zones like the U.S. Weather conditions including temperature, precipitation and wind may affect the survival and reproduction rates of mosquitoes, suitable habitats, distribution, and abundance.  


### Data Sources
A list of the clean datasets is given below:

Trap Locations
* [`train_clean.csv`](../data/train_clean.csv): Clean Trap Locations Training Dataset
* [`test_clean.csv`](../data/test_clean.csv): Clean Trap Locations Testing Dataset

Weather Stataion Readings
* [`weather_clean.csv`](../data/weather_clean.csv): Clean Weather Station Readings Dataset

Spray Locations
* [`spray_clean.csv`](../data/spray_clean.csv): Clean Spray Locations Dataset

Trap Locations & Weather Station Readings
* [`train_weather_clean.csv`](../data/train_weather_clean.csv): Clean Trap Locations Training & Weather Station Readings Dataset
* [`test_weather_clean.csv`](../data/test_weather_clean.csv): Clean Trap Locations Testing & Weather Station Readings Dataset

### Data Dictionary
A description of the variables in the clean datasets is given below:

**Table 1: Trap-Related Variables**

| Variable | Type | In Dataset<br>train_clean.csv | In Dataset<br>test_clean.csv | In Dataset<br>train_weather_clean.csv | In Dataset<br>test_weather_clean.csv | Description |
|:---|:---|:---|:---|:---|:---|:---|
| Id | integer | No | Yes | No | Yes | Id of the row for the Kaggle Challenge |
| Date | datetime | Yes | Yes | Yes | Yes | Date in YYYY-MM-DD the WNV test is performed |
| Day | integer | Yes | Yes | Yes | Yes | Day in DD the WNV test is performed |
| Month | integer | Yes | Yes | Yes | Yes | Month in MM the WNV test is performed |
| Year | integer | Yes | Yes | Yes | Yes | Year in YYYY the WNV test is performed |
| Species | object | Yes | Yes | Yes | Yes | Species of mosquitoes in the trap |
| Trap | object | Yes | Yes | Yes | Yes | Id of the trap |
| Latitude | float | Yes | Yes | Yes | Yes | Latitude of the trap |
| Longitude | float | Yes | Yes | Yes | Yes | Longitude of the trap |
| Station | integer | Yes | Yes | Yes | Yes | Assigned weather station to the trap |
| NumMosquitos | integer | Yes | No | Yes | No | Number of mosquitoes caught in the trap |
| WnvPresent | integer | Yes | No | Yes | No | 1 for presence of West Nile Virus in the trap<br>0 for absence of West Nile Virus in the trap |

**Table 2: Weather-Related Variables**

| Variable | Type | In Dataset<br>weather_clean.csv | In Dataset<br>train_weather_clean.csv | In Dataset<br>test_weather_clean.csv | Description |
|:---|:---|:---|:---|:---|:---|
| Date | datetime | Yes | Yes | Yes | Date in YYYY-MM-DD the weather readings are taken |
| Station | integer | Yes | Yes | Yes | 1 for Weather Station 1 (Lat 41.995 Lon -87.933)<br>2 for Weather Station 2 (Lat 41.786 Lon -87.752) |
| Latitude | float | Yes | No | No | Latitude of the weather station |
| Longitude | float | Yes | No | No | Longitude of the weather station |
| Tmax | float | Yes | Yes | Yes | Maximum dry bulb temperature in degrees Fahrenheit |
| Tmin | float | Yes | Yes | Yes | Minimum dry bulb temperature in degrees Fahrenheit |
| Tavg | float | Yes | Yes | Yes | Average dry bulb temperature in degrees Fahrenheit |
| RH | float | Yes | Yes | Yes | Relative humidity in percentage |
| DewPoint | float | Yes | Yes | Yes | Dew point temperature in degrees Fahrenheit |
| WetBulb | float | Yes | Yes | Yes | Wet bulb temperature in degrees Fahrenheit |
| Heat | float | Yes | Yes | Yes | 65 - Average dry bulb temperature in degrees Fahrenheit |
| Cool | float | Yes | Yes | Yes | Average dry bulb temperature in degrees Fahrenheit - 65 |
| PrecipTotal | float | Yes | Yes | Yes | Total precipitation in inches |
| SeaLevel | float | Yes | Yes | Yes | Average sea level pressure in inches of Hg |
| ResultSpeed | float | Yes | Yes | Yes | Resultant wind speed in miles per hour |
| ResultDir | float | Yes | Yes | Yes | Resultant wind direction in whole degrees |
| AvgSpeed | float | Yes | Yes | Yes | Average wind speed in miles per hour |

**Table 3: Spray-Related Variables**

| Variable | Type | In Dataset<br>spray_clean.csv | Description |
|:---|:---|:---|:---|
| Date | datetime | Yes | Date in YYYY-MM-DD the spray is conducted |
| Latitude | float | Yes | Latitude of the spray |
| Longitude | float | Yes | Longitude of the spray |
| Year | integer | Yes | Year in YYYY the spray is conducted|
| Month | integer | Yes | Month in MM the spray is conducted |
| Week | integer | Yes | Week of the Year the spray is conducted |
| Day | integer | Yes | Day in DD the spray is conducted|

### Results and Analysis
Summarizing the `Accuracy Score`, `Recall Score` and `AUC Score` in the table below for easy comparison:

| Optimized Model          |Training Accuracy | Testing Accuracy | Training Recall Score | Testing Recall Score |AUC Score |
|:-----------------------:|:----------------:|:----------------:|:------------:|:---------:|:---------:|
| Logistic Regression     |      0.640       |      0.657       |    0.816     |   0.818 |     0.79    |
| Support Vector Machine  |      0.831       |      0.781       |    0.991     | 0.562  |     0.78    | 
| K-Nearest Neighbors     |      0.947       |      0.947       |    0.009     | 0.000  |     0.73    |
| Random Forest Classifier|      0.901       |      0.863       |    0.772     | 0.307  |     0.68    |

Having evaluated 4 different classification models, the optimized Logistic Regression Classifier is the best performing model.

The confusion matrix for the optimized logistic regression model shows that the number of correctly predicted entries where WNV is present is 112 while 25 where wrongly classified as being absent of WNV. This corresponds to our recall score of 0.818 which implies that 81.8% of entries are correctly predicted to have presence of WNV. The accuracy of the model is less important to our model performance as it is heavily influenced by the imbalanced dataset, given that 95% of entries in the dataset have 'WNVPresent' = 0. 

The AUC for the ROC curve before and after optimization are both 0.79, which is better than the baseline AUC of 0.50. There is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.  

From the logistic regression coefficients, the top 2 variables with the highest odds coefficient are Month_8 (August) and Month_9 (September) with odds coefficients of 2.893 and 2.010 respectively. The top 5 weather variables with the highest coefficients are 'Tavg', 'SeaLevel', 'ResultDir', 'RH' and 'PrecipTotal'. Their respective values are 1.484, 1.174, 1.141, 1.048 and 1.032. The two species of mosquitoes with the highest coefficients are Culex Pipiens, 1.266, and Culex Pipiens/Restuans, 1.232. As for areas with higher coefficients, the 5 coordinate groups with the highest coefficients are Group 17, Group 18, Group 23, Group 16 and Group 4.


| Group | Latitude | Longitude |
|:---:|:---:|:---:|
|17| 41.9 to 42.0| -87.9 to -87.8 |
|18|41.9 to 42.0| -87.8 to -87.7| 
|23|42.0 to 42.1 | -87.8 to -87.7 |
|16|41.9 to 42.0| -88.0 to -87.9|
|4| 41.6 to 41.7| -87.7 to -87.6| 

The second best performing model is the optimized SVM model. From the confusion matrix, we can observe that the model correctly classified 77 entries with presence of WNV but wrongly classified 60 entries as be absent of WNV. This corresponds to a recall score of 0.562 which implies that 56.2% of all positive entries are correctly classified. 

The AUC for the ROC curve before and after optimization are 0.80 and 0.78 respectively. Although the AUC decreases by 0.02 after optimizing the hyperparamters, the model performance after optimization is still better than the baseline AUC of 0.50. There is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.  

Comparing the two models, the optimized logistic regression model will be used to predict the presence of WNV. This is because it has a higher recall score of 0.818 compared to 0.562. The AUC score of 0.79 is also higher than the optimized SVM model of 0.78.

### Cost Benefit Analysis
As aforementioned in our EDA, we observed that the traps tested positive for WNV are scattered randomly. On closer inspection between spray dates in 2013 and number of mosquitoes tested positive for WNV, we found some evidence that spraying reduced the number of mosquitoes trapped. This in turn lowered the number of mosquitoes tested positive for WNV. Despite higher frequency of spraying in 2013, the number of mosquitoes tested positive for WNV is few times higher than 2011 across the same weeks. We believe there are two key explanations to this. One, pesticide spraying on adult mosquitoes has not been optimized in terms of locations and frequency. Two, confounding factors like weather conditions in 2013 accelerated the growth rate of mosquitoes, outweighing the effects of pesticide spraying. 

In the absence of data or study on effectiveness of pesticide spraying on incidence of human WNV case, we felt it would be meaningful to approach the cost-benefit analysis from a trade-off perspective. In other words, the number of human WNV cases that should have been avoided if we were to use our selected model to determine the pesticide coverage level.

We defined cost as the total expenditure associated with spraying pesticide on adult mosquitoes in incurred in a year. Without our classification model, a reasonable estimate coverage level would be the whole of Chicago, spanning an area of about 149,770 acres. Benefits will be measured by the cost avoidance or savings in the form of healthcare costs and productivity lost associated with the potential reduction in number of human WNV cases from spraying pesticide in Chicago.

### Recommendations
Reconciling findings from our EDA, selected model, and the above cost-benefit analysis, we strongly recommend IDPH to control mosquitoes population in Chicago by spraying pesticide in a more targeted fashion. First, IDPH should commence pesticide spraying from the start of July to the end of August, when higher average temperature accelerates growth of adult mosquito population. The recommended spraying frequency would be weekly for the pesticide to take effect. Second, repeated spraying can be performed on 'hot' zones in coordinates Groups 4, 16, 17, 18 and 23. Finally, it would be helpful to determine and target locations where majority of the Culex Pipiens and Culex Pipiens/Restuans populations are found.

### Conclusion
The best performing model for classifying whether WNV is present for the imbalanced dataset is the optimized logistic regression model. It has an accuracy score of 0.657, recall score of 0.818, and AUC score of 0.790 on the partitioned test dataset. In other words, the model was able to correctly predict 81.8% of positive cases. The results from the model can give us a good prediction of the areas that have a higher presence of WNV. With this information, we can target the spraying of pesticide at these high-risk areas more regularly to control the mosquito population and reduce the presence of WNV.
 
At the same time, we can make improvements to the model by performing over-sampling on the minority class (positive cases) in the dataset. This would increase the number of positive WNV cases to balance the distribution of positive and negative classes. This dataset can then be used to train a new logistic regression model that could correctly predict more positive cases.
 
All, if not most types of mosquitoes, share a similar life cycle. Most mosquito control methods focus on the larval stages because they limit mosquito populations from the start. Spraying pesticide on mosquitoes is only one of the many possible control measures. However, they require repeated applications, have potential negative ecological repercussions, and pose a health risk to the human population. Alternatively, the use of minnows, which are small freshwater fishes, to control mosquito populations may provide many purported benefits in the form of its low maintenance, cost effectiveness, environmental friendliness, and minimal implications on public health.
 
It cannot be over-emphasized that mosquito control requires a concerted effort from individuals, the community, and the government. Relying on one control method alone is also likely to be insufficient in curbing the spread of WNV. A multi-pronged approach, with the participation and support of all stakeholders, is what is needed to keep the WNV at bay.
 
Future steps of this project would entail factoring in the benefit derived (in terms of the cost saved) not just from reducing cases of WNV but also other forms of mosquito-borne diseases such as Zika, Chikungunya, dengue, and malaria. It would also be interesting to analyse the cost-benefit of employing other forms of mosquito eradication measures aside from pesticide spraying, such as the aforementioned use of minnows. Lastly, the predictive model can be modified and applied to help predict and direct spray efforts in other cities of the USA.

### References
[1] https://science.sciencemag.org/content/286/5448/2333.abstract <br>
[2] https://www.cdc.gov/westnile/index.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fwestnile%2Ffaq%2FgenQuestions.html <br>
[3] https://www.ajtmh.org/view/journals/tpmd/90/3/article-p402.xml <br>
[4] https://www.epa.gov/climate-indicators/climate-change-indicators-west-nile-virus <br>
[5] https://www.chicago.gov/content/dam/city/depts/cdph/Mosquito-Borne-Diseases/Zenivex.pdf <br>
[6] https://www.forestrydistributing.com/aqua-zenivex-e20-ulv-insecticide-zeocon