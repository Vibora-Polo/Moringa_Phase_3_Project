# Phase 3 Project
### Student name: John Awallah
### Student Pace: Part-time
### Instructor Name: Brian Chacha and Lians Wanjiku

# SyriaTel Customer Churn Prediction
Project Overview

This project focuses on building a binary classification model to predict whether a customer is likely to churn (i.e., stop doing business) with SyriaTel, a telecommunications company.

Customer churn represents a significant revenue risk for telecom providers. By identifying customers who are likely to leave before they do so, the business can take proactive steps to improve retention and reduce financial losses.

## Business Problem
SyriaTel is experiencing customer attrition, resulting in revenue loss and increased costs associated with acquiring new customers. Retaining existing customers is significantly more cost-effective than acquiring new ones, making churn reduction a strategic priority.

However, the company currently lacks a data-driven system to proactively identify customers who are likely to discontinue their services. Without early detection, retention efforts are reactive rather than preventive, leading to missed opportunities to intervene before customers leave.

The business challenge, therefore, is to determine whether customer usage patterns and service interactions can be used to predict churn and enable targeted retention strategies.

## Objectives
The primary objective of this project is to develop a predictive model that identifies customers who are likely to churn from SyriaTel’s services.

Specifically, the project aims to:

1. Explore customer usage patterns and service characteristics to identify behavioral trends associated with churn.

2. Build and compare multiple classification models using an iterative modeling approach.

3. Optimize model performance by tuning hyperparameters and prioritizing recall to minimize missed churn cases.

4. Select the most suitable model based on business-aligned evaluation metrics.

5. Provide actionable business recommendations to reduce customer attrition and revenue loss.

## Dataset Description

The dataset contains customer-level information including:

Day, evening, night, and international call usage

Customer service call frequency

Subscription to international and voicemail plans

Billing-related usage metrics

The target variable indicates whether a customer churned.

Initial data exploration revealed class imbalance, with significantly more customers staying than leaving. This informed the choice of evaluation metrics and modeling strategy.

# Baseline Model: Logistic Regression

We begin with a simple baseline model using Logistic Regression which is appropriate for binary classification problems and provides easily interpretable coefficients.

Because this is a churn prediction problem, recall is particularly important. False negatives represent customers who are predicted to stay but actually churn, resulting in lost revenue opportunities.

We will evaluate this baseline model using:

Recall

Precision

F1-score

ROC-AUC


#### Model Performance Interpretation

Using class weights and a lower classification threshold, the model’s recall for churned customers is 69% meaning the model correctly identifies approximately 69% of customers who are at risk of churning. In a churn prediction context, this is critical because failing to detect churners (false negatives) leads directly to lost revenue and missed retention opportunities.

Although precision stands at 33%, meaning that only one-third of customers predicted to churn actually do so, this trade-off with recall is acceptable for the business objective. It is generally more costly to miss a customer who will leave than to incorrectly target a customer who was likely to stay. 

The overall accuracy is 76%, which is expected when prioritizing recall in an imbalanced dataset.

The ROC-AUC score of 0.79 indicates that the model maintains good overall ability to distinguish between churners and non-churners. 

# Baseline Model Evaluation

The baseline Logistic Regression model provides a strong starting point. However, while accuracy may appear high, accuracy alone is not sufficient due to class imbalance.

Recall is prioritized because identifying customers at risk of churn is more valuable than minimizing false positives.

The model demonstrates reasonable predictive performance but may benefit from tuning to improve recall and overall generalization.

# Iteration 2: Decision Tree Model

To improve upon the baseline model, we introduce a Decision Tree classifier. Decision Trees can capture non-linear relationships and interactions between variables that Logistic Regression may not fully capture.

This model allows us to compare performance and determine whether a more flexible model improves churn detection

# Decision Tree Interpretation
The tuned Decision Tree model achieved a recall of 62% for churned customers, meaning it correctly identifies the majority of customers at risk of leaving. While slightly lower than the recall-optimized logistic regression, it provides significantly higher precision at 82%, indicating that most customers predicted to churn actually do so.

The model achieved an F1-score of 0.70 and an overall accuracy of 92%, demonstrating strong and balanced performance. Compared to Logistic Regression, the Decision Tree provides a better trade-off between detecting churners and minimizing unnecessary retention efforts.

Overall, the tuned Decision Tree model offers the best balance between recall and precision, making it the preferred model for practical business implementation.

# Tuned Model Evaluation

Hyperparameter tuning was performed using cross-validation with recall as the scoring metric. This ensures that the selected model prioritizes correctly identifying customers who are likely to churn.

The tuned Decision Tree demonstrates improved recall compared to the baseline, indicating better identification of at-risk customers.

## Final Model Choice

After comparing the adjusted Logistic Regression and the tuned Decision Tree models, the Decision Tree was selected as the final model. While Logistic Regression achieved slightly higher recall, it had very low precision, resulting in many false positives and inefficient targeting.

The tuned Decision Tree achieved a better balance between recall (62%) and precision (82%), along with a strong F1-score of 0.70 and overall accuracy of 92%. This balance makes it more practical for identifying at-risk customers while minimizing unnecessary retention efforts, aligning well with the business objective of cost-effective churn reduction.

# Confusion Matrix Interpretation

The Decision Tree model correctly classified 696 non-churning customers and 75 churners. This demonstrates strong overall performance and effective identification of at-risk customers.

Importantly, only 46 churners were missed (false negatives), which is significantly lower than the baseline model. Additionally, the model produced only 17 false positives, indicating that retention efforts would be efficiently targeted with minimal unnecessary intervention.

Overall, the confusion matrix confirms that the tuned Decision Tree achieves a strong balance between detecting churn and minimizing wasted retention resources, making it suitable for business implementation.

## Feature Importance
This is to determine which features drive churn

 ## Feature Importance Interpretation

The Decision Tree model identifies total day minutes as the most influential predictor of churn, followed by customer service calls and total international minutes. This suggests that high daytime usage and frequent interaction with customer service are strongly associated with increased churn risk.

The importance of customer service calls indicates that customer dissatisfaction may be a key driver of churn. Customers who frequently contact support may be experiencing service issues, billing concerns, or unmet expectations. This highlights an opportunity for SyriaTel to improve service responsiveness and proactively engage customers who repeatedly contact support.

The significance of international plan subscription and international usage metrics further suggests that customers with higher international activity may be more price-sensitive or affected by billing complexity. Reviewing pricing structures, plan transparency, and service quality for international users could reduce churn in this segment.

Overall, the feature importance results provide actionable insight into behavioral patterns associated with churn and help the business focus retention strategies on high-usage and high-interaction customers.

# Business Recommendations

Based on the results of the tuned Decision Tree model, SyriaTel can implement a data-driven churn prevention strategy focused on early identification and targeted intervention.

First, the model should be integrated into the company’s customer management system to regularly score customers and flag those at high risk of churn. With a recall of 62% and precision of 82%, the model effectively identifies most at-risk customers while minimizing unnecessary retention efforts.

Second, customers with high total day minutes should be monitored closely. High daytime usage may indicate heavy reliance on the service, and dissatisfaction within this segment could result in significant revenue loss. Proactive engagement, loyalty incentives, or tailored service plans could reduce churn among these high-value users.

Third, the strong influence of customer service calls suggests that frequent interactions with support are linked to dissatisfaction. SyriaTel should investigate recurring service issues, improve response times, and implement follow-up protocols for customers with repeated service complaints.

Additionally, the importance of international usage and international plan subscription indicates potential pricing sensitivity or billing complexity. Reviewing international plan pricing structures and improving transparency may reduce churn in this segment.

Overall, leveraging predictive modeling alongside targeted retention strategies can reduce revenue loss, improve customer lifetime value, and enhance overall service satisfaction.

# Limitations

While the model demonstrates strong performance, several limitations should be considered.

First, the dataset does not include customer satisfaction scores, income levels, competitor pricing, or qualitative feedback, all of which may influence churn behavior. The absence of these variables may limit the model’s ability to fully capture customer decision drivers.

Second, the dataset is imbalanced, with significantly more non-churners than churners. Although tuning and recall prioritization were applied, class imbalance may still influence model stability and performance.

Third, the model is based on historical data and assumes that future customer behavior will follow similar patterns. Changes in market conditions, pricing strategies, or competitive dynamics may reduce predictive accuracy over time.

Finally, while Decision Trees are interpretable, they may be sensitive to data variation. Continuous monitoring and periodic retraining of the model will be necessary to maintain reliability and relevance.

## Executive Summary

This project developed a predictive model to identify customers at risk of churning from SyriaTel’s services. Using historical customer data, multiple classification models were built and evaluated through an iterative modeling approach. After comparing a baseline Logistic Regression model with a tuned Decision Tree, the Decision Tree was selected as the final model due to its strong balance between recall and precision.

The final model achieved a recall of 62% and precision of 82% for churned customers, with an overall accuracy of 92%. This indicates that the model successfully identifies a majority of at-risk customers while minimizing unnecessary retention efforts. Feature importance analysis revealed that total day minutes, customer service calls, and international usage are key drivers of churn, highlighting areas where proactive intervention can reduce customer attrition.

By integrating this model into operational systems, SyriaTel can implement targeted retention strategies, improve service quality for high-risk segments, and reduce revenue loss due to churn. With continuous monitoring and periodic retraining, the predictive framework can serve as a sustainable tool for enhancing customer lifetime value and long-term profitability.
