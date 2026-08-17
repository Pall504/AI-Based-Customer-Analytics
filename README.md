AI-Driven Customer Analytics for Purchase Prediction, Segmentation & Business Intelligence

Project Overview

This project focuses on using machine learning to analyze and predict consumer purchase behavior from customer and e-commerce datasets. The proposed framework combines Machine Learning (ML), Explainable AI (XAI) using SHAP and LIME, and an interactive Streamlit Business Intelligence dashboard to convert customer data into actionable business insights.

The overall workflow is:

Customer Data → Machine Learning → Explainable Insights → Business Decisions

Problem Statement

Customer purchase behavior is difficult to understand and predict because organizations generate complex datasets containing:

Transactions

Customer demographics

Spending patterns

Online behavior

Important relationships and purchasing patterns may remain hidden inside customer data. Organizations also need to predict:

Future purchases

Customer churn

Customer lifetime value

The project aims to convert these analytical findings into actionable marketing and sales decisions.

Objectives

To understand customer demographics, purchasing behavior, purchase patterns, and RFM analysis.

To predict future purchases, purchase frequency, customer churn, and customer lifetime value.

To optimize customer segmentation, product recommendation, customer engagement, and business decisions.

Literature Survey

The project reviews recent research from Springer Nature, IEEE, and ACM covering areas such as:

Customer buying behavior prediction

E-commerce customer shopping behavior

Consumer buying-intention prediction

Market basket analysis

Consumer purchase behavior

E-commerce recommendation systems

Online shopping behavior prediction

Purchase-intention classification

Machine learning in e-commerce

Intelligent customer segmentation

The selected literature is focused on purchase prediction, customer segmentation, recommendation, and e-commerce analytics.

Research Gap

The project identifies four major gaps:

Fragmented Analysis Customer analytics tasks are often handled independently.

Limited Explainability Machine learning predictions may not clearly show which features drive the output.

Business Integration ML models need stronger integration with interactive decision-support dashboards.

Multiple Objectives Purchase prediction, segmentation, churn, and customer lifetime value can be addressed together.

Research Direction

Machine Learning + Explainable AI + Interactive BI

Proposed Methodology

Data Sources The proposed system uses:

UCI Online Retail II

Customer Personality

E-Commerce Behavior

Data Preprocessing The preprocessing stage includes:

Missing-value handling

Duplicate removal

Encoding

Scaling

Outlier handling

Exploratory Data Analysis & Feature Engineering The analysis focuses on:

Customer demographics

Purchase patterns

RFM analysis

Correlation

Feature selection

Machine Learning Models The project compares multiple machine learning approaches:

Logistic Regression

Decision Tree

Random Forest

XGBoost

CatBoost

Model Evaluation The models are evaluated using:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

Explainable AI The framework uses:

SHAP

LIME

These techniques are used to identify and communicate factors influencing model predictions.

Interactive Dashboard Streamlit is used to connect analytical outputs with interactive business decision support and visualizations.

Business Insights The final stage converts:

Data → Learning → Explanation → Decision

Innovation

The proposed framework is distinctive because it combines:

Integrated Analytics

Purchase prediction, segmentation, churn, customer lifetime value, and recommendations are combined into one workflow.

Multi-Model Learning

Logistic Regression, Decision Tree, Random Forest, XGBoost, and CatBoost are compared.

Explainable AI

SHAP and LIME are used to explain factors influencing model predictions.

Interactive Business Intelligence

Streamlit connects analytical outputs with interactive business decision support.

Technology Stack

The project identifies the following tools and technologies:

Python

Pandas

NumPy

Scikit-learn

XGBoost

CatBoost

SHAP

LIME

Streamlit

The project is designed around commonly available data-science tools and does not require specialized hardware.

System Architecture

The proposed system follows eight stages:

Data Sources UCI Retail II, Customer Personality, and E-Commerce Behavior

Data Preprocessing Cleaning, encoding, scaling, and feature engineering

EDA & Analysis Demographics, purchase patterns, RFM, and segmentation

ML Models Logistic Regression, Decision Tree, Random Forest, XGBoost, and CatBoost

Evaluation Accuracy, Precision, Recall, F1-Score, and ROC-AUC

Explainability SHAP and LIME

BI Dashboard Interactive Streamlit visualizations

Business Outcomes Predictions, segmentation, recommendations, and insights

Expected Outcomes

The completed project is intended to:

Analyze customer demographics and purchasing behavior.

Predict future customer purchases.

Identify customer segments using data-driven techniques.

Predict customer churn and customer lifetime value.

Explain model predictions using SHAP and LIME.

Develop an interactive Streamlit dashboard.

Compare multiple machine learning approaches.

Generate actionable insights to improve customer engagement and sales.

Feasibility

Technical Feasibility

The project can be implemented using:

Python

Pandas / NumPy

Scikit-learn

XGBoost / CatBoost

SHAP / LIME

Streamlit

Operational Feasibility

The proposed system supports:

Interactive dashboard

Customer analytics

Prediction support

Explainable results

Business insights

Economic Feasibility

The project uses:

Open-source tools

Benchmark datasets

Standard computing

No specialized hardware

Scalable workflow

Overall Assessment

The project is considered implementable with commonly available data-science tools.

Project Plan

Phase 1 -- Research & Dataset

Literature study

Problem definition

Dataset selection

Phase 2 -- Data Preparation

Data cleaning

Exploratory analysis

Missing values and outliers

Phase 3 -- ML Development

Feature engineering

Model training

Model comparison

Phase 4 -- Evaluation & XAI

Performance metrics

SHAP analysis

LIME analysis

Phase 5 -- Deployment

Streamlit dashboard

Business insights

Final analysis

Roadmap

Research → Data → Models → Evaluation → Dashboard → Final Insights

Each phase produces the inputs required for the next stage.

Literature References

The PPT identifies the following recent references:

Z. Lou, S. Wang, X. Yu, and W. Song, "Predicting customer buying habits using convolutional neural network," Journal of Big Data, vol. 13, art. 122, 2026. Springer. DOI: 10.1186/s40537-026-01464-y.

H. Zhang and Y. Ju, "Analysis of customer shopping behavior on E-commerce platform supported by fuzzy clustering algorithm," Discover Computing, vol. 29, art. 281, 2026. Springer. DOI: 10.1007/s10791-026-10151-8.

C. K. K. Bharathi and K. Elakkiyan, "Leveraging artificial intelligence for predictive modelling of consumer buying intentions on E-Commerce platforms," Discover Artificial Intelligence, vol. 6, art. 108, 2026. Springer. DOI: 10.1007/s44163-025-00815-7.

I. Surana, M. C. Belavagi, and Ramyashree, "KALEFormer: a transformer based model integrating KMeans clustering and LDA for market basket analysis," Discover Computing, vol. 29, art. 224, 2026. Springer. DOI: 10.1007/s10791-026-10103-2.

A. M. Vyas and G. S. Kushwaha, "Enhancing consumer decision-making in shopping 4.0: the impact of augmented reality on emotional engagement and purchase behavior," Future Business Journal, vol. 12, art. 207, 2026. Springer. DOI: 10.1186/s43093-026-00906-6.

A. Poniszewska-Maranda, M. Pakula, and B. Borowska, "Recommendation systems in e-commerce applications with machine learning methods," EASE 2025, pp. 720--725, 2025. ACM. DOI: 10.1145/3756681.3757082.

T. Wang, J. Lin, Y. Zhang, and J. Zhang, "Machine Learning-Based Random Forest Prediction of Online Shopping Behavior in the Digital Economy," DEIS 2025, pp. 315--320, 2025. ACM. DOI: 10.1145/3745133.3745186.

X. Wu, "Category prediction of users' purchase intentions on e-commerce platforms based on machine learning algorithms," BDEIM 2025, 2026. ACM. DOI: 10.1145/3800000.3800178.

E. Dritsas and M. Trigka, "Machine Learning in E-Commerce: Trends, Applications, and Future Challenges," IEEE Access, vol. 13, pp. 99048--99075, 2025. DOI: 10.1109/ACCESS.2025.3572865.

N. Kumar, "Intelligent customer segmentation: unveiling consumer patterns with machine learning," Journal of Umm Al-Qura University for Engineering and Architecture, vol. 16, pp. 774--783, 2025. Springer. DOI: 10.1007/s43995-025-00180-7.

Team

2420030373 -- M. Pallavi

2420030481 -- G. Divija Medha

2420090067 -- Sheri Saswitha

Final Takeaway

The project follows the progression:

UNDERSTAND → Customer Behavior

PREDICT → Purchases & Churn

EXPLAIN → SHAP & LIME

VISUALIZE → Streamlit

DECIDE → Business Insights

The final goal is to transform customer data into explainable insights that support better business decisions.
