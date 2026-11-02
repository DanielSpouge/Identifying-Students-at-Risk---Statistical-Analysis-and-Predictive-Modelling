
# Identifying Students at Risk: Statistical Analysis and Predictive Modelling

## Overview

This project investigates how students' socio-environmental characteristics are associated with academic outcomes in mathematics, and whether statistical analysis and machine learning can assist in the early identification of students who may benefit from additional educational support.

Using a dataset of Portuguese secondary-school students, the analysis explores relationships between academic performance and a range of demographic, social, behavioural and educational factors. Predictive models are developed to assess non-completion risk and final academic attainment, with findings ultimately combined into an interpretable framework for prioritising academic interventions.

The project was undertaken as an independent, exploratory learning exercise to develop practical experience with an end-to-end data science workflow, from exploratory analysis and statistical testing to preprocessing, feature engineering, model development and evaluation. Education was selected as an area of genuine interest, providing a meaningful context in which to investigate and apply these methods.

The scope of the investigation is deliberately broad relative to the size of the dataset. This reflects the exploratory nature of the project, which, beyond its predictive objectives, served as an opportunity for the author to learn and experiment with a wide range of statistical and machine learning techniques and gain practical experience applying them within a complete data science workflow.

## Objectives

The project addresses three main questions:

1. Which socio-environmental and academic factors are associated with student non-completion and final academic performance?
2. How effectively can machine learning predict non-completion risk and final grades using information available earlier in the academic year?
3. Can these findings be combined into a simple, interpretable framework to help prioritise students for additional support?

## Dataset

**Source:** [Math Students — Kaggle](https://www.kaggle.com/datasets/adilshamim8/math-students/data)

**Original research dataset:** [UCI Machine Learning Repository – Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)

The analysis uses the mathematics dataset, containing 399 students and information on academic performance, family background, study habits and social behaviours among other characteristics.

Final grades are recorded on a scale from 0 to 20. The first-period grade (G1) provides an early measure of academic performance, whilst G2 represents a subsequent assessment.

For this investigation, a final grade of zero (G3 = 0) is treated as a proxy for **non-completion**. However, the dataset does not explicitly identify student withdrawals, and a zero final grade may have alternative explanations. The rationale for this working definition, as well as its limitations, is discussed in the exploratory data analysis notebook.

## Project Structure

The project is organised into four Jupyter notebooks, supported by a reusable Python preprocessing module.

| Component | Description |
|---|---|
| **01 — Exploratory Data Analysis** | Investigates relationships between student characteristics, non-completion and final academic performance through visualisation and statistical testing. |
| **02 — Data Preprocessing** | Establishes preprocessing decisions, evaluates candidate feature transformations and develops reusable preprocessing functions. |
| **03 — Predictive Modelling** | Develops, compares, tunes and evaluates models for non-completion risk, final-grade regression and low-attainment identification. |
| **04 — Proposed Intervention Framework** | Implements and evaluates a three-level intervention prioritisation system combining predicted non-completion risk with early academic attainment. |
| **preprocessing_utils.py** | Contains the reusable feature engineering, encoding and scaling functions developed in the preprocessing notebook and imported into subsequent notebooks. |

The notebooks follow the progression of the investigation, from understanding the data to developing and evaluating a proposed practical application.

## Methodology

### 1. Exploratory Data Analysis

Exploratory analysis examines the distribution of academic outcomes and their relationships with demographic, social, behavioural and educational characteristics.

Statistical tests and visualisations are used to investigate potential associations with non-completion and final grades.

The findings inform subsequent preprocessing decisions and provide an initial understanding of which characteristics may be relevant to predictive modelling.

### 2. Data Preprocessing

The preprocessing stage establishes a consistent approach to preparing the dataset for two distinct prediction tasks: non-completion classification and final-grade regression.

Key decisions include:

- **Target definition:** Non-completion is represented by a binary indicator derived from G3 = 0. Final-grade regression is restricted to students with G3 > 0.
- **Feature consolidation:** Correlated parental education and alcohol consumption variables are evaluated using alternative representations. Maximum parental education (`Pedu_max`) and average alcohol consumption (`alc_avg`) are retained.
- **Feature transformation:** Absences are log-transformed to reduce skewness in the academic performance dataset.
- **Leakage and redundancy:** G2 is excluded from predictive modelling because of its close relationship with the final outcome and its potential to introduce target leakage in non-completion classification.
- **Task-specific exclusions:** Absences are excluded from non-completion modelling because their zero values may represent missing information rather than genuine absence counts.
- **Encoding and scaling:** Categorical variables are encoded, and numerical predictors are scaled using Min-Max scaling.

The preprocessing functions also support models with and without G1, allowing the predictive contribution of early academic performance to be assessed separately from socio-environmental characteristics.

These transformations are consolidated into a reusable preprocessing module. During modelling, scaling parameters are fitted using training data and subsequently applied to test data to reduce the risk of data leakage.

### 3. Predictive Modelling

Three related modelling problems are investigated.

**Non-Completion Risk Classification**

Classification models are developed to estimate the probability of a student receiving a final grade of zero.

The analysis compares baseline, linear and tree-based approaches, alongside additional feature engineering and hyperparameter tuning. Model performance is assessed using probability-based and discrimination metrics, including Log Loss, Brier Score, ROC-AUC and Average Precision.

A regularised logistic regression model is ultimately selected for use in the proposed intervention framework.

**Final Academic Performance Regression**

Regression models are developed to predict final grades among students with G3 > 0.

Models are compared using Mean Absolute Error, Root Mean Squared Error and R², with particular emphasis on benchmarking against a simple G1 persistence baseline.

Experiments with and without G1 investigate the additional predictive value of early academic performance relative to socio-environmental factors alone.

**Low-Attainment Identification**

Classification experiments investigate the identification of students at risk of low final attainment.

The performance of machine learning approaches are compared against a simple G1 threshold to assess whether more complex models offer meaningful improvements in identifying students at risk of low attainment.

### 4. Proposed Intervention Framework

The final notebook translates the modelling findings into an exploratory framework for prioritising students who may benefit from additional academic support.

The framework combines two complementary indicators:

- **Non-completion risk:** A probability estimate from an L1-regularised logistic regression model, with a risk threshold of 0.20.
- **Early academic attainment:** A first-period grade threshold of G1 ≤ 8.

Students are assigned to three intervention priority levels:

| Priority | Criteria |
|---|---|
| **High** | Flagged by both the risk model and the G1 threshold |
| **Moderate** | Flagged by exactly one of the two methods |
| **Low** | Flagged by neither method |

The final implementation integrates preprocessing and risk prediction into a reusable modelling pipeline, with the priority rules applied to the resulting predictions.

The framework is assessed using held-out test observations, comparing final-grade distributions, low-attainment rates and non-completion outcomes across the three priority groups.

The purpose is to demonstrate how predictive analysis can be translated into an interpretable decision-support approach, rather than to propose an independently validated intervention policy.

## Key Findings

- Early academic performance, particularly G1, is a strong predictor of final attainment.
- Non-completion is associated with several academic and socio-environmental characteristics, although the relatively small number of non-completers limits the reliability of these relationships.
- Incorporating G1 substantially improves predictive performance compared with using socio-environmental characteristics alone.
- More complex regression models offer limited improvement over a simple G1 persistence baseline, highlighting the importance of comparing machine learning models against straightforward alternatives.
- A simple early-grade threshold provides an effective benchmark for identifying students at risk of low attainment.
- Combining the G1 threshold with predicted non-completion risk produces an interpretable framework that distinguishes groups with substantially different academic outcomes in the held-out test sample.

Overall, the findings illustrate both the potential value and the limitations of predictive modelling in an educational setting. In particular, they demonstrate that model complexity does not necessarily translate into meaningful practical improvements.

## Limitations

Several limitations should be considered when interpreting the results:

- The dataset is relatively small, particularly for non-completion classification.
- The Kaggle-distributed dataset contains 399 observations, compared with 395 in the standard UCI mathematics dataset; the discrepancy remains unresolved.
- A final grade of zero is used as a proxy for non-completion rather than a verified withdrawal outcome.
- The dataset is observational, so identified associations should not be interpreted as causal relationships.
- Model evaluation is based on a limited held-out sample, making performance estimates sensitive to individual observations.
- Findings may not generalise to other schools, educational systems or student populations.
- The proposed intervention framework is exploratory and would require further validation, including assessment of fairness, practical feasibility and the consequences of incorrect classifications, before deployment.

The framework is intended to illustrate a potential decision-support application. It should not be used to determine access to educational support without appropriate human oversight.

## Technologies

- **Programming and data manipulation:** Python, pandas, NumPy
- **Statistical analysis:** SciPy, statsmodels
- **Visualisation:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn, XGBoost
- **Development environment:** Jupyter Notebook

## Reproducing the Analysis

1. Clone or download this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Launch Jupyter Notebook with the `notebooks/` directory as the working directory.
4. Run the four notebooks in numerical order.

Each notebook loads the raw dataset independently from `../data/raw/`. The modelling and intervention notebooks import shared preprocessing functions from `preprocessing_utils.py`.

The notebooks are organised to be independently executable, although the methodological decisions and analysis are intended to be followed chronologically.

## Use of AI Tools

Large language models were used as supplementary learning and development resources throughout the project. This included clarifying technical concepts, exploring implementation approaches, assisting with code troubleshooting and refining written explanations and Markdown documentation.

The project was undertaken as an independent, exploratory learning exercise. The overall direction, methodlological decisions and interpretation of results were guided by the author's own reasoning and judgement.


## Data Attribution

The dataset used in this project was downloaded from [Math Students on Kaggle, published by adilshamim8](https://www.kaggle.com/datasets/adilshamim8/math-students/data).

The underlying Student Performance dataset is associated with:

Cortez, P. and Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of the 5th Future Business Technology Conference.

Original dataset: [UCI Machine Learning Repository – Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).