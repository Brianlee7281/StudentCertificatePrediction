# StudentCertificatePrediction


1. Objective
    The objective of this Machine Learning program is to classify whether a university student successfully complete their degree. By analyzing demographic data, academic history, and extracurricular engagements, this machine learning classification model aims to identify as-risk students for proactive intervention and improved student retention strategies.
2. Data
    This machine learning model leverages various academic and demographic features, including: (generation, school, major data, class…)
    To prepare the raw data for machine learning algorithms, categorical encoding was applied to convert all object and boolean datatypes into numerical formats. LabelEncoder was applied to transform categorical text data into machine-readable integers. The dataset was split into training and testing sets to evaluate model generalization on data. For preprocessing, StandardScaler was applied to normalize the numerical features, ensuing that features with larger magnitudes do not disproportionately influence the model
3. Models Used
    To evaluate a wide range of machine learning algorithms, PyCaret was utilized for initial model testing. AutoML helped to quickly identify which algorithms performed well on the dataset. 
    Based on the insights gained from PyCaret, three tree-based models were selected as base learners. GuassianNB, DecisionTreeClassifier, and XGBClassifier were trained and evaluated independently. To maximize predictive performance and reduce variance, Stacking Classifier was implemented to combine the three selected models.
4. Outcome
    The model achieved an overall accuracy of f1 score: 0.277, accuracy score: 0.61, precision score: 0.24, recall score: 0.32 with the validation data
