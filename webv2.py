import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import os



def APP():
    st.sidebar.title('Zungumza')
    mode_selection = st.sidebar.selectbox("Proceed",('Find More About Us', 'Take A Test'))

    if mode_selection == 'Find More About Us':
        about()

    if mode_selection == 'Take A Test':
        application()

def about():
    st.title("Zungumza")
    aboutfile = "\\homepage.txt"
    path = os.getcwd() + aboutfile
    file = open(path, 'r+')
    st.markdown(
        """Zungumza is an online mental help kit that aids the user to check on the necessity to
        mental assistance from qualified HealthCare Expert (or) Medical Therapheutic Attention""")
    st.markdown( """Programmed By Granton Onyango  Supervisor Mr. Samson Ooko""")
    st.markdown("""Dataset available at kaggle.com""")
    

def application():
    inp_Age = st.slider('What is your age?',18,120,20)
    inp_Gender = st.selectbox('How do you Identify?', ('Male','Female','Trans'))
    inp_selfEmployed = st.selectbox('Are You self-employed?', ('Yes', 'No'))
    inp_workInterfere = st.radio('How Often Does Work Interfere With Your Personal Life?',
                                    ('Never','Rarely','Often','Sometimes')) #DONTKNOW
    inp_familyHistory = st.selectbox('Does any of your family member have a Mental Condition?',('Yes','No'))
    inp_remoteWork = st.selectbox('Are you a remote worker?',('Yes','No'))
    inp_benefits = st.radio('Do You Enjoy any work related benefits?',('Yes','No'))
    inp_careOptions = st.selectbox('Do you have any known Care Options i.e NHIF?',('Yes','No'))
    inp_wellnessProgram = st.selectbox('Are you a member of any wellness Programme?',('Yes','No'))
    inp_seekHelp = st.radio('Have you ever sought help in regards to your mental Health?',('Yes', 'No'))
    inp_leave = st.selectbox('How Easy is it for you to take leave off work?', 
                                ('Very Easy','Somewhat Easy', 'Somewhat Difficult', 'Very Difficult'))
    inp_mentalHealthConsequence = st.selectbox('Could you say you have work related stress?',('Yes', 'No'))
    inp_physcicalHealthConsequence = st.radio('Does your work involve frequent Physical Strain?',('Yes', 'No'))
    inp_coworkers = st.selectbox('Do You relate well with your Co-workers?',('Yes','No', 'Some of them'))
    inp_supervisors = st.selectbox('Do You relate well with your Supervisors?',('Yes','No', 'Some of them'))
    inp_mentalHealthInterview = st.selectbox("Have you ever taken a mental health interview before?",('Yes','No'))
    inp_physicalHealthInterview = st.selectbox('Have you ever taken a physical test before?',('Yes', 'No'))
    inp_treatment = 'No'


    if st.button('Submit'):

        train_df = pd.read_csv('cleandataset.csv')

        train_df = train_df.append({'Age':inp_Age, 'Gender':inp_Gender, 'self_employed':inp_selfEmployed, 'family_history': inp_familyHistory, 'treatment':inp_treatment,
       'work_interfere':inp_workInterfere, 'remote_work':inp_remoteWork, 'benefits':inp_benefits, 'care_options':inp_careOptions,
       'wellness_program':inp_wellnessProgram, 'seek_help':inp_seekHelp, 'leave':inp_leave, 'mental_health_consequence':inp_mentalHealthConsequence,
       'phys_health_consequence':inp_physcicalHealthConsequence, 'coworkers':inp_coworkers, 'supervisor':inp_supervisors,
       'mental_health_interview':inp_mentalHealthInterview, 'phys_health_interview':inp_physicalHealthInterview}, ignore_index = True)

        
        train_df.to_csv("appendedDf.csv", index=False)
        report = pd.read_csv("appendedDf.csv")


        for feature in train_df:
            le = preprocessing.LabelEncoder()
            le.fit(train_df[feature])
            train_df[feature] = le.transform(train_df[feature])


        scaler = MinMaxScaler()
        for columns in train_df:
            train_df[columns] = scaler.fit_transform(train_df[[columns]])


        def prediction():
            #Splitting dataset into training and testing sets
            feature_columns = ['Age','Gender', 'family_history', 'work_interfere','self_employed',
                            'remote_work', 'leave', 'mental_health_consequence',
                            'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                            'benefits', 'care_options', 'wellness_program',
                            'seek_help']
            X = train_df[feature_columns]
            y = train_df['treatment']

            X_train = X.iloc[:-1 , :]
            y_train = y.iloc[:-1]
            X_test = X.tail(1)
            y_test = y.tail(1)


            forest = RandomForestClassifier(max_depth = None,  min_samples_leaf= 8, min_samples_split=2, n_estimators = 20, random_state = 1)
            my_forest = forest.fit(X_train, y_train)
            y_pred_class = my_forest.predict(X_test)

            results = pd.DataFrame({'Index': X_test.index, 'Treatment': y_pred_class, 'Expected': y_test})
            results.to_csv('results.csv', index=False)
            print(results.head(20))

            #Replace the predicted value to report df
            report["treatment"] = report['treatment'].replace(["No"], y_pred_class)


            #Picking the last row from report df
            report_last_row = report.tail(1)
            print(report_last_row)

            '''
            #The below lines of code are used to instantiate the database; once initiated they can be commented
            analysis = pd.DataFrame(report_last_row, index=None)
            analysis.to_csv("analysis.csv", index=False)
            '''

            analysis = pd.read_csv("analysis24.csv", index_col=False)
            #Append is soon to depreciate, consider using concat
            analysis2 = analysis.append(report_last_row, ignore_index=True)
            analysis2.to_csv("analysis24.csv",index=False)

            return y_pred_class

        if prediction() == 1.0:
            print(train_df.shape)
            st.info('Kindly Consider Mental Assistance')
        elif prediction() == 0.0:
            print(train_df.shape)
            st.info('No Nessecity for Mental Assistance')

APP()