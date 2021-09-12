# Machine Learning Projects

The repo contain ML projects completed by me for academic, self learning, and hobby purposes. Presented in the form of iPython Notebooks markdown files.

- Car Quality Detection ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/master/Used_Car_Quality_Detection.ipynb#top), [<img src="https://img.icons8.com/office/40/000000/blog.png"/>](https://blog.jovian.ai/machine-learning-with-python-implementing-xgboost-and-random-forest-fd51fa4f9f4c)): 

	- Problem Statement: One of the biggest challenges of an auto dealership purchasing a used car at an auto auction is the risk of that the vehicle might have serious issues that prevent it from being sold to customers.The challenge of this competition is to predict if the car purchased at the Auction is a Kick (bad buy).
	- Processed data over 72k records with over 30 features to predict the quality of a car.
	- Libraries used - pandas, numpy, sklearn, matplotlib and seaborn.
	- Machine learning models implement - Random Forest, XGBoost.
	- Performed hyperparameter tuning along with random search CV to achieve accuracy of 88%.
	- Submitted this model to Kaggle Competetion scoring in top 10 percent at the leaderbord. 


- Califonia Housing Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Califonia-Housing-Dataset/Califonia_Housing_Analysis.ipynb))
	- The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data. The object is to identify the median housing value in that area.
	- The dataset have over 20,000 records and 9 features. 
	- Libraries used - numpy, OS, requests, urllib, Pandas, sklearn.
	- Feature analysis, stratified shuffle split, Visualized data to gain insights.
	- Data cleaning and preprocessing acivities - duplicate check, null values, One-Hot encoding, Feature scaling.
	- Model implement - Linear regression, Decision tree.
	- Hyperparameter tuning using gridsearchCV to evalute best model. RMSE of **47362** is achivied on test set.

- Wine Quality Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Wine-Quality-Dataset/Wine_Quality_Dataset.ipynb))
	- Problem Statement - The Wine Quality Dataset involves predicting the quality of white wines on a scale given chemical measures of each wine.
	- The dataset have 5000 obseravation and 10 features.
	- Libarary used - Numpy, Pandas, Matplotlib and sklearn.
	- Feature analysis, Identiying relevant features, co relation of features with the target feature.
	- LinearRegression implemented and RMSE score of **0.75** is achivied.

- Bank Note Dateset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Bank-Note-Dataset/Bank_Note_Analysis.ipynb))
	- Problem Statement - The Banknote Dataset involves predicting whether a given banknote is authentic given a number of measures taken from a photograph.
	- The dataset have 1300 observeration of various noteparametes as features. It is a binary classification problem.
	- Data cleaning, Feature analysis and visuliazation using Pandas.
	- ML models implemented - Logistic regression, KNeighborsClassifier and SVM.
	- Hyperparamter tuning using GridsearchCV.
	- Model evluation - Precision and Recall calculated along with f1 scores.


- Abalone Dataset ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Abalone-Dataset/Abalone_Dataset_Analysis.ipynb))
	- Business case - Predicting the age of abalone on the given physical measures. 
	- The dataset have over 4000 observation along with 8 features.
	- Build pipeline, implemented StandardScaler and One-Hote encoding for numberical and categorical columns.
	- Model Implemented - Linear regression, Decision Tree and Random forest. Evluation matrix - RMSE score.
	- Hyperparamter tuning using grid search CV and achieved an RMSE score of **2.254**.

- Pima Indians Diabetes Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Pima-Indians-Diabetes-Dataset/Pima_Indians_Diabetes_Dataset.ipynb))
	- The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.
	- It is a binary (2-class) classification problem. There are 768 observations with 8 input variables and 1 output variable.
	- Libarary used - Numpy, Pandas, Matplotlib and sklearn. 
	- Implemented KNN classification. Parameter tuning using GridsearchCV.
	- The baseline performance of predicting the most prevalent class is a classification accuracy of approximately 65%. I achieved a classification accuracy of approximately 77%.

- Swedish Auto Insurance Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Swedish-Auto-Insurance-Dataset/Swedish_Auto_Insurance_Dataset.ipynb))
	- Problem Statement - The Swedish Auto Insurance Dataset involves predicting the total payment for all claims in thousands of Swedish Kronor.
	- Libarary used - Numpy, Pandas, Matplotlib and sklearn.
	- Folowing ML model implemented and evaluated against RMSE, MAE scores - Linear Regression, Decison trees, Random Forest
	- It is a regression problem.The model performance of predicting the mean value is an RMSE of approximately 118 thousand Kronor.

- Ionosphere Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/hargurjeet/MachineLearning/blob/Ionosphere/Ionosphere_Data_Analysis.ipynb))
	- Problem Statement - The Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.
	-  There are 351 observations with 34 input variables and 1 output variable.
	-  As the dataset beeing small, I implemented the k fold cross validations.
	-  ML models implemented - Logistic Regression,  KNeighborsClassifier, DecisionTreeClassifier, SVM)
	-  Have achieved the classification accuracy of 93%.

- Sonar Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Sonar-Dataset/Sonar_Dataset.ipynb))
	- The Sonar Dataset involves the prediction of whether or not an object is a mine or a rock given the strength of sonar returns at different angles.
	- It is a binary (2-class) classification problem with 200 observations and 61 features.
	- ML Models implemented - LogisticRegression, LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier, SVM.
	- Hyperparameter tuning and achieved a classification accuracy of approximately 93%.

- Wheat Seeds Dataset([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.jupyter.org/github/hargurjeet/MachineLearning/blob/Wheat-Seeds/Wheat_Seeds_Analysis_Pytorch.ipynb))
	- The Wheat Seeds Dataset involves the prediction of species given measurements of seeds from different varieties of wheat.
	- There are 199 observations with 7 input variables and 1 output variable.
	- Implemented a Feed forward neural network.
	- Accuray of 60% is achivied.

	_Tools: scikit-learn, Pandas, Seaborn, Matplotlib, NumPy, Plotly_ 


I also dabble in all other technology. You can access by complete portfolio [here](https://github.com/hargurjeet/Portfolio-Projects/blob/main/README.md)

If you liked what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at gurjeet333@gmail.com
