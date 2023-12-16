

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv('/Users/gopalsagi/Downloads/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv')


df.drop(['patientid'], axis=1, inplace=True)


st.sidebar.title("Cardiovascular Disease Dashboard")
page = st.sidebar.selectbox("Choose a page", ["Exploration", "Visualization", "Prediction"])


if page == "Exploration":
    st.title("Cardiovascular Disease Dataset Exploration")

    
    st.write("### Basic Statistics of the Dataset")
    st.write(df.describe())

    
    st.write("### Sample Data")
    st.write(df.sample(5))


elif page == "Visualization":
    st.title("Cardiovascular Disease Data Visualization")

    
    st.write("### Age Distribution")
    plt.figure(figsize=(8, 6))
    hist = plt.hist(df['age'], bins=10, color='paleturquoise', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Histogram of age distribution', fontweight='bold')
    for i in range(len(hist[0])):
        plt.text(hist[1][i] + 3, hist[0][i], str(int(hist[0][i])), fontsize=10, ha='center')
    st.pyplot(plt.gcf()) 

    
    st.write("### Age Distribution by Gender")
    bins = [0, 12, 18, 35, 60, df['age'].max()]
    labels = ['Children', 'Teens', 'Young Adults', 'Adults', 'Elderly']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    age_group_counts = df['age_group'].value_counts()
    st.bar_chart(age_group_counts)

    
    st.write("### Age Distribution by Gender")
    male_data = df[df['gender'] == 1]['age']
    female_data = df[df['gender'] == 0]['age']
    male_counts = male_data.value_counts().sort_index()
    female_counts = female_data.value_counts().sort_index()
    age_values = df['age'].unique()
    age_values.sort()
    plt.figure(figsize=(10, 8))
    plt.barh(age_values, male_counts, color='skyblue', edgecolor='black', label='Male')
    plt.barh(age_values, -female_counts, color='salmon', edgecolor='black', label='Female')
    plt.xlabel('Frequency')
    plt.ylabel('Age')
    plt.title('Age distribution by gender', fontweight='bold')
    plt.legend()
    plt.grid(axis='x')
    st.pyplot(plt.gcf()) 

    
    st.write("### Chest Pain Types by Age & Gender")
    custom_palette = {0: 'salmon', 1: 'sandybrown'}
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='chestpain', y='age', hue='gender', data=df, palette=custom_palette)
    plt.title('Distribution of chest pain types by age & gender', fontweight='bold')
    plt.xlabel('Chest pain type')
    plt.ylabel('Age')
    plt.legend(title='Gender', loc='upper right')
    st.pyplot(plt.gcf())  


elif page == "Prediction":
    st.title("Cardiovascular Disease Prediction")

   
    st.write("### XGBoost Classifier")

   
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('target', axis=1),
        df['target'],
        test_size=0.25,
        random_state=42
    )

    
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    
    st.sidebar.header("Real-time Prediction Input")

    
    input_features = ['age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']

    
    if set(input_features) != set(X_train.columns):
        st.sidebar.error("Input features mismatch. Please select appropriate features.")
        st.sidebar.write("Features used for training:", X_train.columns)
        st.sidebar.write("Features provided for prediction:", input_features)
    else:
        user_input = pd.DataFrame({feature: [st.sidebar.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))] for feature in input_features})

        
        st.sidebar.subheader("User Input")
        st.sidebar.write(user_input)

        
        if st.sidebar.button("Predict"):
            prediction = model.predict(user_input)
            st.subheader("Real-time Prediction")
            st.write(f"The predicted outcome is: {prediction[0]}")


