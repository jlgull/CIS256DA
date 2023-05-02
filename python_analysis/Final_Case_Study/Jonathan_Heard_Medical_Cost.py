#!/usr/bin/python3
# coding: utf-8

# # The Medical Costs case study for section 1
# 
# Prerequisite chapters: 2-4
# 
# Scenario
# 
# You are an analyst employed by a health care company. Your boss has asked you to produce a report that answers the following questions about medical insurance charges.
# 
# Instructions
# 
# 1. Create a Notebook for the study, name the Notebook first_last_medical_costs where first_last specifies your first and last name.
# 2. Complete the tasks listed
# 3. Create a heading for the question.
# 4. Create a table or plot that answers the question and use a heading or raw text to summarize what the table or plot tells you.
# 5. Repeat steps 2-4 for each question.
# 
# Tasks
# 
# 1. Read the data from the CSV file into a DataFrame (2 pts.).
# 2. Display the first five rows of data (2 pts.).
# 3. Display information about each of the columns (2 pts.).
# 4. Display the number of unique values in each column (2 pts.).
# 
# Questions
# 
# 1. How is age related to medical costs (5 pts.)?
# 2. How is number of children related to medical costs (5 pts.)?
# 3. How is the number of people distributed by region (5 pts.)?
# 4. How is the number of people distributed by age (5 pts.)?
# 5. How are the charges distributed (5 pts.)?
# 
# Note
# 
#     If you encounter warnings or errors as you work on this case study, search the Internet for a solution to the problem and implement it.
# 
# After completing the assignment, please submit your .ipynb file.

# # The Medical Costs case study for section 2
# 
# Prerequisite chapters: 2-8
# 
# Scenario
# 
# Your boss liked your previous report so much that he wants you to expand on it by completing the following tasks and answering the following questions.
# 
# Tasks and questions
# 
# 1. Bin the bmi column. To do that, search the Internet to determine how you should bin and label the data (5 pts.).
# 2. How are the charges related to the bmi (5 pts.)?
# 3. How is the smoker status related to the charges (5 pts.)?
# 4. How are the charges related to the region (5 pts.)?
# 5. Which region has the highest obesity percentage (5 pts.)?
# 
# Instructions
# 
# Add heading and code cells to the Notebook that you created for section 1 so that it completes the tasks and answers the questions shown above.
# 
# After completing the assignment, please submit your .ipynb file.
# 

# ## The Medical Costs case study for section 3
# 
# Scenario
# 
# Once again, your boss wants you to expand on your previous report by completing the following tasks and answering the following questions.
# 
# Tasks and questions
# 
# 1. Create a simple regression to show the relationship between charges and age (10 pts.).
# 2. How does this relationship change with smoking status (10 pts.)?
# 3. How does this relationship change with BMI (10 pts.)?
# 
# The following 3 tasks and questions are EXTRA CREDIT
# 
# 1. Create a multiple regression model to predict charges. To do that, you’ll need to dummy encode and rescale the data (10 pts.).
# 2. Make predictions with your multiple regression model and evaluate how well your model is working (10 pts.).
# 3. What is the optimal number of parameters for the multiple regression (10 pts.)?
# 
# Instructions
# 
# Add heading and code cells to the Notebook that you created for section 2 so that it answers the questions shown above.
# 
# After completing the assignment, please submit your .ipynb file.

# ## The Medical Costs case study for section 1
# 
# Prerequisite chapters: 2-4
# 
# Scenario
# 
# You are an analyst employed by a health care company. Your boss has asked you to produce a report that answers the following questions about medical insurance charges.
# 
# Instructions
# 
# 1. Create a Notebook for the study, name the Notebook first_last_medical_costs where first_last specifies your first and last name.
# 2. Complete the tasks listed
# 3. Create a heading for the question.
# 4. Create a table or plot that answers the question and use a heading or raw text to summarize what the table or plot tells you.
# 5. Repeat steps 2-4 for each question.
# 

# ## Import all required modules

# In[1]:


# Import pandas and assign the alias pd
import pandas as pd

# Import Seaborn and assign the alias sns
import seaborn as sns

# Import the Scikit-learn library
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import additional SciKit-learn class
from sklearn.preprocessing import StandardScaler

# Import additional SciKit-learn classes

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

# Import mathplotlib.pyplot and assign the alias plt

import matplotlib.pyplot as plt


# ## Tasks for Section 1 of the Medical Costs case study.

# In[2]:


# 1. Read the data from the CSV file into a DataFrame.

cost_data = pd.read_csv('insurance.csv')


# In[3]:


# 2. Display the first five rows of data.

cost_data.head()


# In[4]:


# 3. Display information about each of the columns.

cost_data.info(memory_usage = 'deep')


# In[5]:


# 4. Display the number of unique values in each column.

cost_data.nunique()

""" Based on the nunique() method count, it looks like sex, smoker and even region can be made
  into dummy variables in future regression work.
"""
# In[6]:


# Renamed the charges column to cost and then display the first five lines.

cost_data.rename(columns = {'charges':'cost'}, inplace = True)

cost_data.head()


# In[7]:


# Change the region column data to titlecase and display the first five lines. 

cost_data.region = cost_data.apply(lambda x: x.region.title(), axis = 1)

cost_data.head()


# ## Questions for Section 1 of the Medical Costs case study.
# 
# 1. How is age related to medical costs (5 pts.)?
# 2. How is number of children related to medical costs (5 pts.)?
# 3. How is the number of people distributed by region (5 pts.)?
# 4. How is the number of people distributed by age (5 pts.)?
# 5. How are the charges distributed (5 pts.)?

# In[8]:


# Question 1. How is age related to medical costs?


# In[9]:


# Determine the average cost by age, using the groupby() method.

cost_by_age = cost_data.groupby(['age']).cost.mean()

cost_by_age = cost_by_age.reset_index().sort_values('cost', ascending = False)

cost_by_age.head()


# In[10]:


# Plot the relationship between age and medical costs.

# Found a method to increase the plot size, while researching on the internet on the following
#   site: https://www.geeksforgeeks.org/how-to-set-a-seaborn-chart-figure-size/
#   I had to import import matplotlib.pyplot as plt to make this code work.

fig, ax = plt.subplots(figsize=(8, 8))

ax = sns.regplot(data = cost_by_age, x = 'age', y = 'cost', ci = None)

ax.set(title = 'Average Cost versus Age', xlabel = 'Age', ylabel = 'Average Cost');

plt.show()

""" 
Based on the trend line in the graph above, there is a positive relationship between Average Cost
  and the Age of the primary insured person. This data leaves some things sort of hanging. I have to assume that
  the cost/charge is related to the entire family, not simply the primary person whose age is listed.
"""
# In[11]:


# Question 2. How is number of children related to medical costs?


# In[12]:


# Examine the number and range of the children column.

cost_data.children.value_counts()


# In[13]:


# Determine the average cost by the number of children, using the groupby() method.

cost_by_child = cost_data.groupby(['children']).cost.mean()

cost_by_child = cost_by_child.reset_index().sort_values('cost', ascending = False)

cost_by_child.head(10)


# In[14]:


cost_by_child.plot(x = 'children', y = 'cost', kind = 'bar', 
                   title = 'Average cost and number of Childern',
                  xlabel = 'Number of Children',
                  ylabel = 'Average Cost', legend = None);

"""
Based on the bar chart above, there does not seem to be a relationship
  between the number of Childeren and the Average Cost.
"""
# In[15]:


# Question 3. How is the number of people distributed by region?


# In[16]:


# Find the number of people in each family.

cost_data['familySize'] = cost_data.children + 1

cost_data.head()


# In[17]:


# Find the total number of people by region using the groupby() method.

family_by_region = cost_data.groupby('region').familySize.sum()

family_by_region = family_by_region.reset_index().sort_values('familySize', ascending = False)

family_by_region.head()


# In[18]:


# Plot the results of summing the familySize vs. region.

family_by_region.plot(x = 'region', y = 'familySize', kind = 'bar',
                     title = 'Total family size for each Region',
                     ylabel = 'Total Family Size', rot = 45, legend = None);

"""
The total family size seems to be evenly spread accross the for region.
With the Southeast having the maximum at 746 and the Northeast the minimum at 663.
"""
# In[19]:


# Question 4. How is the number of people distributed by age?


# In[20]:


# Use the describe() method to gather and show the data distribution of the age column.

cost_data.age.describe()


# In[21]:


# Use various statistical tools to print the data distribution of the age column.

print(f'\nThe Youngest age listed is {cost_data.age.min()} years.\n')

print(f'The mean (or Average) age is {cost_data.age.mean():.2f} years.\n')

print(f'The Oldest age listed is {cost_data.age.max()} years.\n')

print(f'The Standard Deviation (STD) for the age data is {cost_data.age.std():.2f} years.\n')


# In[22]:


# Create a KDE plot of the age column and draw a line at the Mean age.

g = sns.displot(data = cost_data, x = 'age', kind = 'kde')

for ax in g.axes.flat:
    ax.axvline(cost_data.age.mean(), ls = '--', color = 'red')
    ax.set(title = 'KDE plot of Age, with a red line at the Mean age.\n\n',
          xlabel = 'Age')

plt.show()
"""
 Based on the data and plot, the age distribution is not a standard bell curve.
   The Youngest (minimum) age is 18 and the Oldest (maximum) age is 64.
   A simple middle of those two numbers would be 41; but the actual mean of the data is 39.41 years.
   The graph shows a greater number of people below 41, which is why there is a double peak in the data.
"""
# In[23]:


# Question 5. How are the charges distributed (5 pts.)?

"""
I ended up doing 2 analysises of the charges data, realize I renamed the charges column to cost.

The first method was a duplicate of the analysis I did for question 4.

The second method was the method used in the Laptop Case study presented by the intructor.
"""
# In[24]:


# Use the describe() method to gather and show the data distribution of the cost column.

cost_data.cost.describe()


# In[25]:


# Use various statistical tools to print the data distribution of the cost column.

print(f'\nThe Lowest cost listed is ${cost_data.cost.min():.2f}.\n')

print(f'The mean (or Average) cost is ${cost_data.cost.mean():.2f}.\n')

print(f'The Highest cost listed is ${cost_data.cost.max():.2f}.\n')

print(f'The Standard Deviation (STD) for the cost data is ${cost_data.cost.std():.2f}.\n')


# In[26]:


# Create a KDE plot of the cost column and draw a line at the Mean cost.

g = sns.displot(data = cost_data, x = 'cost', kind = 'kde')

for ax in g.axes.flat:
    
    ax.axvline(cost_data.cost.mean(), ls = '--', color = 'red')
    ax.axvline(cost_data.cost.mean() - cost_data.cost.std(), ls = '--', color = 'green')
    ax.axvline(cost_data.cost.mean() + cost_data.cost.std(), ls = '--', color = 'green') 
    
    ax.set(title = 'KDE plot of Cost data, with a red line at the Mean cost\n and green lines 1 STD above and below the Mean cost.\n\n' ,
          xlabel = 'Cost in dollars.')

plt.show()

# In[27]:


g = sns.displot(data = cost_data, x = 'cost', bins = 15, kde = True)

for ax in g.axes.flat:
    ax.set(xlabel = 'Cost in dollars.');

plt.show()
    
"""
It was interesting to see that the KDE plot line appears to be the same; but there was no data for below 0.

Based on the review of the 2 sets of plots, the cost data seems to be have a larger number of data points or counts
in the lower end of the Cost scale.
"""

# # The Medical Costs case study for section 2
# 
# Prerequisite chapters: 2-8
# 
# Scenario
# 
# Your boss liked your previous report so much that he wants you to expand on it by completing the following tasks and answering the following questions.
# 

# ## Tasks and questions for Section 2 of the Medical Costs case study.
# 
# 1. Bin the bmi column. To do that, search the Internet to determine how you should bin and label the data (5 pts.).
# 2. How are the charges related to the bmi (5 pts.)?
# 3. How is the smoker status related to the charges (5 pts.)?
# 4. How are the charges related to the region (5 pts.)?
# 5. Which region has the highest obesity percentage (5 pts.)?

# In[28]:


# Task 1. Bin the bmi column. To do that, search the Internet to determine how you should bin and label the data.


# In[29]:


# Display the basic bmi column data.

cost_data.bmi.head


# In[30]:


# Use the describe() method to gather and show the data distribution of the bmi column.

cost_data.bmi.describe()


# In[31]:


# Plot the bmi data to further review the data. 

# I tried serveral bin counts and 15 seemed to match the KDE curve.

g = sns.displot(data = cost_data, x = 'bmi', bins = 15, kde = True)

for ax in g.axes.flat:
    ax.set(xlabel = 'BMI value');
    


# In[32]:


# Use cut(), for equal-sized bin to get a look at the data.

pd.cut(cost_data.bmi, bins = 5)


# In[33]:


# Use qcut() to get same number of bmi values in each bin, to get a look at the data.

pd.qcut(cost_data.bmi, q = 5)

"""
After reviewing both the cut() and qcut() methods, the cut() method seems to show a the bmi values best.Went looking on the internet for how best to establish the bin range and labels. 
  Found the best sorce at this url: 
   https://www.ncbi.nlm.nih.gov/books/NBK541070/, 
   BMI Classification Percentile And Cut Off Points.
 
The general design for my bins and labels, I didn't try to break down the Obesity ranges
  for this study. Based on this I chose to label any bmi value greater than 30 as obese,
  not as obesity, from the listing below.
   
Severely underweight - BMI less than 16.5kg/m^2
Underweight - BMI under 18.5 kg/m^2
Normal weight - BMI greater than or equal to 18.5 to 24.9 kg/m^2
Overweight – BMI greater than or equal to 25 to 29.9 kg/m^2
Obesity – BMI greater than or equal to 30 kg/m^2
Obesity class I – BMI 30 to 34.9 kg/m^2
Obesity class II – BMI 35 to 39.9 kg/m^2
Obesity class III – BMI greater than or equal to 40 kg/m^2 (also referred to as severe, extreme, or massive obesity)
"""
# In[34]:


# Use cut(), to create bins and labels. Then add a bmiBins column and list the first five rows.

cost_data['bmiBins'] = pd.cut(cost_data.bmi, bins = [0, 16.5, 18.5, 25, 30, 55],
      labels = ['Severly underweight', 'Underweight', 'Normal Weight', 'Overweight', 'Obese'])

cost_data.head()


# In[35]:


# Now to show the breakdown by the bmi bins.

cost_data.bmiBins.value_counts()


# In[36]:


# Question 2. How are the charges related to the bmi?


# In[37]:


# Determine the average cost by bmiBins, using the groupby() method.

cost_by_bmi = cost_data.groupby(['bmiBins']).cost.mean()

cost_by_bmi = cost_by_bmi.reset_index().sort_values('cost', ascending = False)

cost_by_bmi.head()


# In[38]:


# Plot the relationship between bmi and medical costs.

ax = sns.catplot(data = cost_by_bmi, y = 'bmiBins', x = 'cost', kind = 'bar', aspect = 2.0)

ax.set(title = 'Average Cost versus BMI data\nGeneral labels for BMI ranges used.', ylabel = 'BMI Bins', xlabel = 'Average Cost');

"""
Based on this chart, there does seem to be a relationship between cost/charges and bmi.

The "Severly underweight" data is an outlier; since there is only 1 person in that bmi bin.
"""
# In[39]:


# Question 3. How is the smoker status related to the charges?


# In[40]:


# Display the basic smoker column data.

cost_data.smoker.head()


# In[41]:


# After reviewing the data, I am changing the yes/no to Yes/No.
#   Used the title() method.

cost_data.smoker = cost_data.smoker.str.title()


# In[42]:


# Now to show the breakdown by the smoker column.

cost_data.smoker.value_counts()


# In[43]:


# Determine the average cost by smoker, using the groupby() method.

cost_by_smoker = cost_data.groupby(['smoker']).cost.mean()

cost_by_smoker = cost_by_smoker.reset_index().sort_values('cost', ascending = False)

cost_by_smoker.head()


# In[44]:


# Plot the data for the average cost/charge by smoker (Yes or No).

g = sns.catplot(data = cost_by_smoker, x = 'smoker', y = 'cost', kind = 'bar')

for ax in g.axes.flat:
    ax.set(title = 'Average cost by Smoker status',
          xlabel = 'Smoker Status',
          ylabel = 'Average Cost');
    
"""
The answer to question 3, how is the smoker status related to the charges, appears
  to be that the if the Smoker Status is "Yes" the Average Cost is increased from
  $8,434.27 to $32,050.23 
"""
# In[45]:


# Question 4. How are the charges related to the region?


# In[46]:


# Find the Average cost by region using the groupby() method.

cost_data_region = cost_data.groupby('region').cost.mean()

cost_data_region = cost_data_region.reset_index().sort_values('cost', ascending = False)

cost_data_region.head()


# In[47]:


# Plot the data for the average cost/charge by region.

g = sns.catplot(data = cost_data_region, x = 'region', y = 'cost', kind = 'bar')

for ax in g.axes.flat:
    ax.set(title = 'Average cost by Region',
          xlabel = 'Region',
          ylabel = 'Average Cost',
          ylim = (12000,15000));
    
"""
Based on the plot above, the average cost/charge in the Southeast is the highest; but more research would 
  be required to determine if there is some underlying driver for the difference. The difference between the 
  highest ($ 14,735.41) and the lowest ($ 12,346.94) is $ 2,388.47.
"""  
# In[48]:


# Question 5. Which region has the highest obesity percentage?


# In[49]:


# Determine the count of bmi items by region, using the groupby() method.
#    Then convert the Series to a DataFrame and rename the bmi column to bmiCount.

bmi_region_count_sr = cost_data.groupby(['region']).bmi.count()

bmi_region_count = bmi_region_count_sr.to_frame().reset_index()

bmi_region_count.rename(columns = {'bmi':'bmiCount'}, inplace = True)

bmi_region_count.info()

bmi_region_count.head()


# In[50]:


# Determine the count of bmiBins for each region, using the groupby() method.
#    Then convert the Series to a DataFrame.

bmi_by_region_sr = cost_data.groupby(['region','bmiBins']).bmi.count()

bmi_by_region = bmi_by_region_sr.to_frame().reset_index()

bmi_by_region.info()

bmi_by_region.head()


# In[51]:


# Join the two bmi region based DataFrames for future math and show the first five lines.

bmi_joined = bmi_by_region.merge(bmi_region_count, on = 'region')

bmi_joined.head()


# In[52]:


# Reduce the bmi_joined DataFrame to just the Obese entries
#    Then calculate the percent of Obese for each region and sort the data.

bmi_pct_obese = bmi_joined[['region', 'bmiBins', 'bmi', 'bmiCount']].query('bmiBins == "Obese"')

bmi_pct_obese['pct_Obese'] = ((bmi_pct_obese.bmi / bmi_pct_obese.bmiCount) * 100).round(2)

bmi_pct_obese[['region', 'pct_Obese']].sort_values('pct_Obese', ascending = False).head()

"""
Based on the data displayed above, the Southeast region has highest obesity percentage.
"""

# # The Medical Costs case study for section 3
# 
# Scenario
# 
# Once again, your boss wants you to expand on your previous report by completing the following tasks and answering the following questions.
# 

# ## Tasks and questions for Section 3 of the Medical Costs case study.
# 
# 1. Create a simple regression to show the relationship between charges and age (10 pts.).
# 2. How does this relationship change with smoking status (10 pts.)?
# 3. How does this relationship change with BMI (10 pts.)?
# 

# ## Question 1. Create a simple regression to show the relationship between charges and age.

# In[53]:


# First I looked at the correlation of all the data in cost_data DataFrame.

cost_data.corr()[['cost']].sort_values(by ='cost', ascending = False)

"""
Based on the r-values, there is neglibible positive correlation between the age and cost values.
"""
# In[54]:


# Then using the Seaborn lmplot() methods to create a simple regression between age and cost (charge).

g = sns.lmplot(data = cost_data, x = 'age', y = 'cost', height = 8, 
               aspect = 2, ci = None, scatter_kws = {'s':3}, line_kws = {'color':'red'} )

for ax in g.axes.flat:
    ax.set(title = 'Cost vs Age\n',
          xlabel = 'Age',
          ylabel = 'Charge / Cost');

"""
Based on the plot above there does seem to be a positive relationship, though there is considerable scatter.
"""
# In[55]:


#  Create the test and training datasets using age and cost(charge).

x_train, x_test, y_train, y_test = train_test_split(cost_data[['age']], cost_data[['cost']],
                                                   test_size = 0.20, random_state = 42)


# In[56]:


# Create a linear regression model from the training dataset.

costModel = LinearRegression()
costModel.fit(x_train, y_train)


# In[57]:


# Score the model using the test dataset.

costModel.score(x_test, y_test)


# In[58]:


# Score the model using the training dataset.

costModel.score(x_train, y_train)

"""
Based on the 2 r-values, there is neglibible positive correlation between the age and cost values.
I went ahead and completed a prediction analysis of the data.
"""
# In[59]:


# Predict the y values based on the x values in the test dataset, and store the results in a variable.
#   Then, put the predicted values in a new DataFrame.

y_predicted_cost = costModel.predict(x_test)

predicted_cost = pd.DataFrame(y_predicted_cost, columns = ['cost_predicted'])

predicted_cost.head()
    


# In[60]:


# Join the y_test and predicted data with the x_test data, save the combined data in a new DataFrame, 
#           and then display the first five rows of data.

combined_cost = predicted_cost.join([y_test.reset_index(drop = True),
                                  x_test.reset_index(drop = True)])

combined_cost.head()


# In[61]:


# Add the residual value to the new DataFrame.

combined_cost['cost_residual'] = combined_cost.cost - combined_cost.cost_predicted

combined_cost.head()


# In[62]:


# Plot the residual valuses in a Seaborn KDE plot.

sns.displot(data = combined_cost, kind = 'kde', x = 'cost_residual');

"""
Based on all the linear regression work, there is not a strong relationship between cost and age.

"""
# ## Question 2. How does this relationship change with smoking status?

# In[63]:


# Using the Seaborn lmplot() methods, compare the age to cost (charge) when the smoker status is added.

sns.set_style('whitegrid')
g = sns.lmplot(data = cost_data, x = 'age', y = 'cost', hue = 'smoker', height = 8, 
               aspect = 2, ci = None)

for ax in g.axes.flat:
    ax.set(title = 'Cost vs Age\n with smoker status controlling the Hue\n',
          xlabel = 'Age',
          ylabel = 'Charge / Cost');

"""
The slope of the line for smokers "Yes" and "No" appears to be similar; but the smoker "Yes" line starts greater than $20,000 above the smoker "No". There is still a large amount of scatter.
"""
# ## Question 3. How does this relationship change with BMI?
"""
I did the lmplot twice, simply switching which values went on x or y.
"""
# In[64]:


# Using the Seaborn lmplot() methods, compare the age to cost (charge) when the bmi bins are added.

g = sns.lmplot(data = cost_data, x = 'age', y = 'cost', hue = 'bmiBins', height = 8, 
               aspect = 2, ci = None)

for ax in g.axes.flat:
    ax.set(title = 'Cost vs Age\n with BMI bins controlling the Hue\n',
          xlabel = 'Age',
          ylabel = 'Charge / Cost');
    


# In[65]:


# Using the Seaborn lmplot() methods, compare the age to cost (charge) when the bmi bins are added.

g = sns.lmplot(data = cost_data, y = 'age', x = 'cost', hue = 'bmiBins', height = 8, 
               aspect = 2, ci = None)

for ax in g.axes.flat:
    ax.set(title = 'Cost vs Age\n with BMI bins controlling the Hue\n',
          ylabel = 'Age',
          xlabel = 'Charge / Cost');
    
"""
Based on the second plot, the bmi bin for Obese seems to flatter; but extends further into the higher cost portion of the plot. The line for Normal Weight has a steeper line; but ends above the $30,000 area. There is still a lot of scatter present; but the Obese group has a higher cost (charge) over all.
"""
# # The following 3 tasks and questions are EXTRA CREDIT
# 
# 1. Create a multiple regression model to predict charges. To do that, you’ll need to dummy encode and rescale the data (10 pts.).
# 2. Make predictions with your multiple regression model and evaluate how well your model is working (10 pts.).
# 3. What is the optimal number of parameters for the multiple regression (10 pts.)?

# ## Question 1. Create a multiple regression model to predict charges. To do that, you’ll need to dummy encode and rescale the data.

# In[66]:


# For a review, display the cost_data info(). This is being done to review which column has "ojbect" data for creating the dummy columns.

cost_data.info()


# In[67]:


# Get the object data type columns and store them in a list. They are sex, smoker, and region. 
#             Then, convert those columns to dummy variables, and store the results in a new DataFrame.

objectColumns = ['sex', 'smoker', 'region']

dummyObjects = pd.get_dummies(cost_data[objectColumns])


# In[68]:


# Create a new DataFrame that is a copy of the original cost_data DataFrame.
#     Join the DataFrame with the dummy variables to it then drop 3 object columns.
#     Store the result in a DataFrame named cost_dataDummies, 
#     and then use the info() method to display the resulting columns.


cost_data_drop = cost_data

cost_dataDummies = dummyObjects.join(cost_data)

cost_dataDummies = cost_dataDummies.drop(columns = objectColumns)

cost_dataDummies.info()


# In[69]:


# Rescale the data in the numeric columns, and then display the rescaled data.
#   Note: I did not rescale the cost column as it was the dependent variable column.
#         I also didn't inclue the bmiBins, as it was a category data type.


numCols = ['age', 'bmi', 'children', 'familySize']

scaler = StandardScaler()

cost_dataDummies[numCols] = scaler.fit_transform(cost_dataDummies[numCols])

cost_dataDummies.head()


# In[70]:


# Display the correlation data for the cost (charge) column, using all the dummy colums.

cost_dataDummies.corr()[['cost']].sort_values(by = 'cost', ascending = False)

"""
Based on the correlation data above, the only 2 variables with a strong correlation are the smoker_Yes and smoker_No.
"""
# In[71]:


cost_dataDummies.info()


# In[72]:


# Step 1: Split the data into test and training datasets. The test dataset will consist of 20% of the total dataset,
#         and I am specify 42 for the random_state parameter, to have the ability to repeat the results..
#         Note that there aren’t any non-numeric columns to drop.

costTrain, costTest = train_test_split(cost_dataDummies.drop(columns = ['bmiBins']), test_size = 0.20, random_state = 42)


# In[73]:


# Step 2: Create the multiple regression model with all numeric and dummy variables.

model = LinearRegression()

xCols = ['smoker_Yes', 'smoker_No']

model.fit(costTrain[xCols], costTrain['cost'])


# In[74]:


# Step 3: Create and store the test Score and train Score values in variables.

test_Score = model.score(costTest[xCols], costTest["cost"])

train_Score = model.score(costTrain[xCols], costTrain['cost'])


# In[75]:


# Printing the 2 r-values.

print(f'The test r-score is: \t\t{test_Score:.9f}')

print(f'The training r-score is:\t {train_Score:.9f}')

# Printing the delta of the 2 R values.

# print(f'\n\tScore delta is:\t {model.score(costTrain[xCols], costTrain["cost"]) - model.score(costTest[xCols], carsTest["price"]):.9f}')

print(f'\n\tScore delta is:\t {train_Score - test_Score:.9f}')

"""
Even though the r-scores are low, I am going to create and plot the predicted data.
"""
# In[76]:


# Using the model to make predictions from the test dataset.

y_predicted = model.predict(costTest[xCols])


# In[77]:


# Use the predicted data to create a DataFrame for the predicted and actual cost.

# create DataFrame for the cost predictions
predicted = pd.DataFrame(y_predicted, columns=['predictedCost'])

# combine the test data and the predicted data into a new DataFrame.
final = predicted.join([costTest.reset_index(drop=True)])

# display the first five lines of the new comined DataFrame; but just three columns.
final[['smoker_Yes','smoker_No', 'cost', 'predictedCost']].head()


# In[78]:


# Calculate the residual and create a new column in the DataFrame.

final['residual'] = final.cost - final.predictedCost

# display the first five lines of the revised DataFrame.
final[['cost', 'predictedCost', 'residual']].head()


# In[79]:


# Calculate and print the Standard Deviation and Mean values for the residual data.

std_data = final['residual'].std()

mean_data = final['residual'].mean()

print(f'The mean of the residual data is = {mean_data:.0f}')

print(f'One Standard Deviation (STD) = {std_data:.0f}')


# In[80]:


# Add lines for the Mean and 1, 2 & 3 Std on the KDE plot of the residual data.

graph = sns.kdeplot(data = final, x = 'residual')

graph.axvline(-3 * std_data + mean_data, ls = '--', color = 'green')
graph.axvline(-2 * std_data + mean_data, ls = '--', color = 'blue')
graph.axvline(-1 * std_data + mean_data, ls = '--', color = 'red')
graph.axvline(mean_data, ls = '--', color = 'green')
graph.axvline(1 * std_data + mean_data, ls = '--', color = 'red')
graph.axvline(2 * std_data + mean_data, ls = '--', color = 'blue')
graph.axvline(3 * std_data + mean_data, ls = '--', color = 'green')
graph.set(title = f'KDE Plot with lines as 1, 2 & 3 STD\n\nMean = {mean_data:.0f}\n\n1 STD = {std_data:.0f}\n');

"""
The plot shows that the highest density, using smoker_Yes and smoker_No, has a negative residual value.
"""
# ## Question 2. Make predictions with your multiple regression model and evaluate how well your model is working.
"""
Based on the data, r-values and the plot of residuals, I am going to repeat all the steps; but use all of the data as independent variables.
"""
# In[81]:


# Step 1: Split the data into test and training datasets. The test dataset will consist of 20% of the total dataset,
#         and I am specify 42 for the random_state parameter, to have the ability to repeat the results..
#         Note that there aren’t any non-numeric columns to drop.

costTrain, costTest = train_test_split(cost_dataDummies.drop(columns = ['bmiBins']), test_size = 0.20, random_state = 42)


# In[82]:


# Step 2: Create the multiple regression model with all numeric and dummy variables.

model = LinearRegression()

xCols = cost_dataDummies.corr().drop(columns = ['cost']).columns.tolist()

model.fit(costTrain[xCols], costTrain['cost'])


# In[83]:


# Step 3: Create and store the test Score and train Score values in variables.

test_Score = model.score(costTest[xCols], costTest["cost"])

train_Score = model.score(costTrain[xCols], costTrain['cost'])


# In[84]:


# Printing the 2 r-values.

print(f'The test r-score is: \t\t{test_Score:.9f}')

print(f'The training r-score is:\t {train_Score:.9f}')

# Printing the delta of the 2 R values.

# print(f'\n\tScore delta is:\t {model.score(costTrain[xCols], costTrain["cost"]) - model.score(costTest[xCols], carsTest["price"]):.9f}')

print(f'\n\tScore delta is:\t {train_Score - test_Score:.9f}')

"""
The r-scores are much higher, in the Strong positive range and the delta is smaller.
"""
# In[85]:


# Using the model to make predictions from the test dataset.

y_predicted = model.predict(costTest[xCols])


# In[86]:


# Use the predicted data to create a DataFrame for the predicted and actual cost.

# create DataFrame for the cost predictions
predicted = pd.DataFrame(y_predicted, columns=['predictedCost'])

# combine the test data and the predicted data into a new DataFrame.
final = predicted.join([costTest.reset_index(drop=True)])

# display the first five lines of the new comined DataFrame; but just three columns.
final[['smoker_Yes','smoker_No', 'cost', 'predictedCost']].head()


# In[87]:


# Calculate the residual and create a new column in the DataFrame.

final['residual'] = final.cost - final.predictedCost

# display the first five lines of the revised DataFrame.
final[['cost', 'predictedCost', 'residual']].head()


# In[88]:


# Calculate and print the Standard Deviation and Mean values for the residual data.

std_data = final['residual'].std()

mean_data = final['residual'].mean()

print(f'The mean of the residual data is = {mean_data:.0f}')

print(f'One Standard Deviation (STD) = {std_data:.0f}')


# In[89]:


# Add lines for the Mean and 1, 2 & 3 Std on the KDE plot of the residual data.

graph = sns.kdeplot(data = final, x = 'residual')

graph.axvline(-3 * std_data + mean_data, ls = '--', color = 'green')
graph.axvline(-2 * std_data + mean_data, ls = '--', color = 'blue')
graph.axvline(-1 * std_data + mean_data, ls = '--', color = 'red')
graph.axvline(mean_data, ls = '--', color = 'green')
graph.axvline(1 * std_data + mean_data, ls = '--', color = 'red')
graph.axvline(2 * std_data + mean_data, ls = '--', color = 'blue')
graph.axvline(3 * std_data + mean_data, ls = '--', color = 'green')
graph.set(title = f'KDE Plot with lines as 1, 2 & 3 STD\n\nMean = {mean_data:.0f}\n\n1 STD = {std_data:.0f}\n');


"""The plot shows that the highest density, using all the independent data, is still a negative residual value;
  but the data is more uniformily distributed. It is interesting that when the mean of this data is compared with the earlier
  version, it is larger; but the Standard Deviation is is smaller.
"""
# ## Question 3. What is the optimal number of parameters for the multiple regression?
"""
Now I am going to step through the independent varibles and try to arrive at the optimal number.
"""
# In[90]:


# Now to use a for loop to score the model for varying numbers of features.

model = LinearRegression()
testScores = []
trainScores = []

for i in range(1, len(costTrain.columns)):
    fs = SelectKBest(score_func=mutual_info_regression, k=i)
    fs.fit(costTrain.drop(columns=['cost']), costTrain['cost'])

    x_train_fs = fs.transform(costTrain.drop(columns=['cost']))
    x_test_fs = fs.transform(costTest.drop(columns=['cost']))

    model.fit(x_train_fs, costTrain['cost'])
    
    testScore = model.score(x_test_fs, costTest['cost'])
    trainScore = model.score(x_train_fs, costTrain['cost'])
    testScores.append(testScore)
    trainScores.append(trainScore)
    


# In[91]:


# Now to plot the test and training scores

df = pd.DataFrame(data={'testScores':testScores, 'trainScores':trainScores})
df.reset_index(inplace=True)
df.rename(columns={'index':'numFeatures'}, inplace=True)
df.numFeatures = df.numFeatures + 1
df.plot(x='numFeatures', y=['testScores','trainScores']);


# In[92]:


# Or, if you only want to plot the gap, you could do it like this...

df['gap'] = df['trainScores'] - df['testScores']
df.plot(x='numFeatures', y=['gap']);

"""
The answer to question 3, what is the optimal number of parameters for the multiple regression?
    The answer is that as few as 2 parameters are required to reach a very high r-value relationship between the training and test scores,
    the number could be raised to 4 with a marginal gain; but the r-value doesn't change much with the addition of the 2 additional 
    parameters.
"""
