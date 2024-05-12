import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('Gunsnew.csv')
#df.info()

'''
# Fill missing values with mean
#df.fillna(df.mean(numeric_only=True).round(1), inplace=True)
'''

# Identify continuous (numeric) type attributes
numeric_attributes = df.select_dtypes(include=['float64', 'int64']).columns


'''
# Initialize an empty list to store the statistics for numeric datatypes
statistics1 = []

# Calculate statistics for each numeric attribute
for attribute in numeric_attributes:
    total_values = df[attribute].count()
    missing_values_percentage = (1 - (total_values / len(df))) * 100
    cardinality = df[attribute].nunique()
    min_value = df[attribute].min()
    max_value = df[attribute].max()
    quartiles = df[attribute].quantile([0.25, 0.5, 0.75])
    median = df[attribute].median()
    std_deviation = df[attribute].std()
    average = df[attribute].mean()
    # Format average with one decimal place
    min_value = '{:.1f}'.format(min_value)
    quartiles = [f'{q:.1f}' for q in quartiles]
    median = '{:.1f}'.format(median)
    max_value = '{:.1f}'.format(max_value)

    # Append statistics to the list
    statistics1.append({
        'Attribute': attribute,
        'Total Values': total_values,
        'percMiss': missing_values_percentage,
        'Cardinality': cardinality,
        'Min': min_value,
        'q1': quartiles[0],
        'Average': average,
        'Median': median,
        'q3': quartiles[2],
        'Max': max_value,
        'Stand eviation': std_deviation
    })

# Create a DataFrame from the list of statistics
statistics_df1 = pd.DataFrame(statistics1)

# Print the DataFrame containing the statistics
print(statistics_df1)
'''

'''
# Handling outliers for numeric attributes
for attribute in numeric_attributes:
    # Calculate quartiles
    Q1 = df[attribute].quantile(0.25)
    Q3 = df[attribute].quantile(0.75)
    IQR = Q3 - Q1  # Interquartile range
    
    # Define the boundaries for outliers
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Replace outliers with NaN or adjust them
    df[attribute] = np.where((df[attribute] < lower_bound) | (df[attribute] > upper_bound), np.nan, df[attribute])
   
'''

#Identify category type attributes
category_attributes = df.select_dtypes(include=['object']).columns

'''
#Initialize an empty list to store the statistics for category datatypes
statistics2 = []

# Calculate statistics for each categorical attribute
for attribute in category_attributes:
    total_values = df[attribute].count()
    missing_values_percentage = (1 - (total_values / len(df))) * 100
    cardinality = df[attribute].nunique()
    mode_counts = df[attribute].value_counts()
    mode1_value = mode_counts.index[0]  # Mode1
    mode1_frequency = mode_counts.iloc[0]  # Frequency of Mode1
    mode1_percentage = '{:.1f}'.format((mode1_frequency / len(df)) * 100)  # Percentage of Mode1 rounded to 1 decimal place

    # Mode 2 statistics
    if len(mode_counts) > 1:  # Check if there is a second mode (mode 2)
        mode2_value = mode_counts.index[1]  # Mode 2
        mode2_frequency = mode_counts.iloc[1]  # Frequency of Mode 2
        mode2_percentage = '{:.1f}'.format((mode2_frequency / len(df)) * 100)  # Percentage of Mode 2 rounded to 1 decimal place
    else:
        mode2_value = None
        mode2_frequency = None
        mode2_percentage = None

    # Append statistics to the list
    statistics2.append({
        'Attribute': attribute,
        'Total Values': total_values,
        'perMiss': missing_values_percentage,
        'Cardinality': cardinality,
        'Mode1': mode1_value,
        'freqMode1': mode1_frequency,
        'percMode1': mode1_percentage,
        'Mode2': mode2_value,
        'freqMode2': mode2_frequency,
        'percMode2': mode2_percentage
    })



# Create a DataFrame from the list of statistics
statistics_df2 = pd.DataFrame(statistics2)

# Print the DataFrame containing the statistics
print(statistics_df2)
'''

'''
# Set a threshold for rare categories
threshold = 0.05

# Iterate over each categorical attribute
for attribute in category_attributes:
    # Calculate the frequency of each category
    category_freq = df[attribute].value_counts(normalize=True)
    
    # Identify categories with frequency below the threshold
    rare_categories = category_freq[category_freq < threshold].index
    
    # Replace rare categories with 'Other'
    df[attribute] = df[attribute].apply(lambda x: 'Other' if x in rare_categories else x)

# Now, the rare categories in categorical attributes have been grouped into 'Other'
'''


# Histograms for numerical attributes
for attribute in numeric_attributes:
    plt.figure(figsize=(8, 6))  # Set figure size
    
    # Calculate the number of bins using the formula
    num_bins = int(1 + 3.22 * np.log(len(df)))
    
    plt.hist(df[attribute], bins=num_bins, color='skyblue', edgecolor='black')  # Plot histogram
    plt.title(f'Histogram of {attribute}')  # Set title
    plt.xlabel('Values')  # Set x-axis label
    plt.ylabel('Frequency')  # Set y-axis label
    plt.grid(True)  # Show grid
    plt.show()  # Show plot


# Bar plots for categorical attributes
for attribute in category_attributes:
    plt.figure(figsize=(8, 6))  # Set figure size
    df[attribute].value_counts().plot(kind='bar', color='skyblue')  # Plot bar plot
    plt.title(f'Bar Plot of {attribute}')  # Set title
    plt.xlabel('Categories')  # Set x-axis label
    plt.ylabel('Frequency')  # Set y-axis label
    plt.grid(True)  # Show grid
    plt.show()  # Show plot




# Scatter plots for numeric attributes
for i in range(len(numeric_attributes)):
    for j in range(i + 1, len(numeric_attributes)):
        plt.figure(figsize=(8, 6))  # Set figure size
        plt.scatter(df[numeric_attributes[i]], df[numeric_attributes[j]], color='skyblue', alpha=0.6)  # Plot scatter plot
        plt.title(f'Scatter Plot between {numeric_attributes[i]} and {numeric_attributes[j]}')  # Set title
        plt.xlabel(numeric_attributes[i])  # Set x-axis label
        plt.ylabel(numeric_attributes[j])  # Set y-axis label
        plt.grid(True)  # Show grid
        plt.show()  # Show plot



# SPLOM diagram
sns.pairplot(df[numeric_attributes])
plt.show()
''


'''
# Using bar plots to investigate attribute dependency for categorical attributes
for attribute in category_attributes:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=attribute, y='target_variable', data=df)  # 'target_variable' yerine hedef değişkeninizin adını kullanın
    plt.title(f'Bar Plot of {attribute} vs Target Variable')
    plt.xlabel(attribute)
    plt.ylabel('Target Variable')
    plt.show()
'''

'''
# Rastgele 5 eyalet seçimi - choose random five states
random_states = random.sample(list(df['state'].unique()), 5)

# Seçilen eyaletlerin verisini filtreleme - filter the fata of states choseen
filtered_df = df[df['state'].isin(random_states)]

# Histogramlar için - for Histogram
for categorical_attribute in category_attributes:
    for numeric_attribute in numeric_attributes:
        plt.figure(figsize=(12, 6))
        
        # Histogram
        sns.histplot(x=numeric_attribute, hue=categorical_attribute, data=df, kde=True)
        plt.title(f'Histogram of {numeric_attribute} by {categorical_attribute}')
        plt.xlabel(numeric_attribute)
        plt.ylabel('Frequency')
        
        plt.tight_layout()  # Grafikler arasındaki boşluğu düzenle

        plt.show()  # Tüm grafikleri tek bir plt.show() komutu altında göster
'''
'''
# Box plotlar için
for categorical_attribute in category_attributes:
    for numeric_attribute in numeric_attributes:
        plt.figure(figsize=(12, 6))
        
        # Box Plot
        sns.boxplot(x=categorical_attribute, y=numeric_attribute, data=df)
        plt.title(f'Box Plot of {numeric_attribute} by {categorical_attribute}')
        plt.xlabel(categorical_attribute)
        plt.ylabel(numeric_attribute)

        plt.tight_layout()  # Grafikler arasındaki boşluğu düzenle

        plt.show()  # Tüm grafikleri tek bir plt.show() komutu altında göster
'''

'''
# Calculate covariance matrix
cov_matrix = df[numeric_attributes].cov()
print(cov_matrix)

# Calculate correlation matrix
corr_matrix = df[numeric_attributes].corr()
print(corr_matrix)

# Plot correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix Heatmap for Numeric Attributes')
plt.show()
'''

'''
# Min-Max normalizasyonu uygula
scaler = MinMaxScaler()
df[numeric_attributes] = scaler.fit_transform(df[numeric_attributes])

statistics_df = df.describe(percentiles=[.25, .5, .75])
print(statistics_df.loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
'''
'''
# Convert states to continuous variables
state_mapping = {state: idx + 1 for idx, state in enumerate(df['state'].unique())}
df['state'] = df['state'].map(state_mapping)

# Convert law categorical data to continuous variables
df['law'] = df['law'].map({'yes': 1, 'no': 0})

# Print the first few rows of the DataFrame to verify the transformations
print(df.sample(10))
'''
