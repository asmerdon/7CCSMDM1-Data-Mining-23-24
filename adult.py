# If for whatever reason this code does not work with the structure for marking, I have left in my (commented out) testing code under
# every function, which gives a working example of the output.

# Imports
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.
def read_csv_1(data_file):
	df = pd.read_csv(data_file)
	df.drop(columns=['fnlwgt'], inplace=True) #drop fnlwgt
	return df

"""adult_path = "adult.csv"
adult_df = read_csv_1(adult_path)"""

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df.shape[0]

"""num_rows_adult = num_rows(adult_df)
print("Number of rows in adult DataFrame:", num_rows_adult)"""

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return list(df.columns)

"""column_names_adult = column_names(adult_df)
print("Column names in adult DataFrame:", column_names_adult)"""

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	missing = df.isna() #isna() gets missing vals
	num_missing = missing.sum().sum() #first sum gets total in each column, second total across all columns
	return num_missing

"""num_missing = missing_values(adult_df)
print("Number of missing values in adult DataFrame:", num_missing)"""

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	missing = df.isna().any() #.any() checks each column to see if there is missing val
	columns_missing = missing[missing].index.tolist() #gets column names (index) with missing val and addes to list
	return columns_missing

"""columns_missing_values = columns_with_missing_values(adult_df)
print("Columns with missing values in adult DataFrame:", columns_missing_values)"""

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	degrees = ['Bachelors', 'Masters']
	degrees_df = df[df['education'].isin(degrees)] #filters education to only include values in degrees list
	percentage = len(degrees_df) / len(df) * 100 #calculate percentage
	percentage_rounded = round(percentage, 1) #round to 1 decimal place
	return percentage_rounded

"""percentage_bachelors_masters = bachelors_masters_percentage(adult_df)
print("Percentage of individuals with Bachelors or Masters degree:", percentage_bachelors_masters)"""

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df_no_missing = df.dropna() #drop missing values
	return df_no_missing

"""df_no_missing_values = data_frame_without_missing_values(adult_df)
print("DataFrame without missing values:")
print(df_no_missing_values.head())"""

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	features_df = df.drop(columns=['class']) #get rid of target attribute (which is class)
	df_encoded = pd.get_dummies(features_df) #get_dummies() one hot encoding
	return df_encoded

"""df_encoded = one_hot_encoding(df_no_missing_values) 
print("DataFrame after one-hot encoding and label encoding:")
print(df_encoded.head())"""

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	label_encoder = LabelEncoder() #sklearn label encoder
	encoded_labels = label_encoder.fit_transform(df['class']) #converts class to binary (0 or 1) depending on whether the row is over 50k or under 50k
	encoded_series = pd.Series(encoded_labels, name='encoded_class') #make new pandas series copy
	return encoded_series

"""encoded_series = label_encoding(df_no_missing_values)
print("Encoded labels series:")
print(encoded_series.head())"""

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
    tree = DecisionTreeClassifier() #decision tree
    tree.fit(X_train, y_train) #train in it on encoded dataframes
    y_pred = tree.predict(X_train) #see how model performs (predicted values)
    return pd.Series(y_pred, name='predicted_class') #return as pandas series

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
    error_rate = (y_pred != y_true).sum() / len(y_true) #get error rate, in my testing this came to 0.08279156162929548
    return error_rate

"""predicted_labels = dt_predict(df_encoded, encoded_series)
error_rate = dt_error_rate(predicted_labels, encoded_series)
print("Error rate:")
print(error_rate)"""

