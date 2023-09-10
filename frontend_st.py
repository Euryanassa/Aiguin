import streamlit as st
from scripts.lib_pre_sklearn import ClsLibPreprocessing
from scripts.lib_ml_xgboost import ClsLibMlXGBoostClassifier
from scripts.lib_read_file import read_pandas_file
import json
import matplotlib.pyplot as plt
import ast

def clicked_read_my_file():
    st.session_state.clicked_read_my_file = True

if 'clicked' not in st.session_state:
    st.session_state.clicked_read_my_file = False

# Title of the application
st.title("Welcome to Aiguin Vanilla")
st.write("Currently we support only XGBoost Classifier but It will be updated.")




model_options = {
            'XGBoost':['Regression','Classification'],
            'CatBoost':['Test1','Test2']
                }
col1, col2 = st.columns([1,1])
with col1:
    xgb_unit_1 = st.selectbox('Model', options = model_options, key = 1)
with col2:
    xgb_unit_2 = st.selectbox('Algorithm', options = model_options[xgb_unit_1], key = 2)

if xgb_unit_1 == 'XGBoost' and xgb_unit_2 == 'Classification':
    # Model Parameters
    # name = st.text_input("Enter your name")

    # Upload and Read File
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        # st.write(uploaded_file)

        df_inp_1, df_inp_2, df_inp_3 = st.columns([1,1,1])
        with df_inp_1:
            add_current_date_select = st.selectbox('Add Current Date', options = ['Yes','No'])
            add_current_date_select_info = 0 if add_current_date_select == 'No' else 1
        with df_inp_2:
            header_select = st.selectbox('Header', options = ['Exist at First Column','Not Exist'])
            header_select_info = 0 if header_select == 'Exist at First Column' else None
        with df_inp_3:
            index_select = st.selectbox('Index', options = ['Exist at First Column','Not Exist'])
            index_select_info = 0 if index_select == 'Exist at First Column' else None

        #####
        train_method = st.selectbox('GridSearchCV or Detailed Training', options = ['GridSearchCV','Detailed Training'])

        #####
        if train_method == 'GridSearchCV':
            tm1, tm2 = st.columns([1,1])
            tm3, tm4 = st.columns([1,1])

            with tm1:
                test_size = st.text_input("Test Size", "0.3")
            with tm2:
                target_column = st.text_input("Target Column (Value of y)", "target_column")
            with tm3:
                eval_metric = st.text_input("Evaluation Metric", "['logloss','rmse','rmsle']")
            with tm4:
                early_stopping_rounds = st.text_input("Early Stopping Rounds", "10")
            #params_grid = st.text_input("Early Stopping Rounds", 
            #                            "{'n_estimators': [10, 100, 1000],'learning_rate': [0.01, 0.1, 1.0],'max_depth': [3, 5, 10]}")

        read_train_my_file = st.button("Read, Preprocess and Train!", on_click = clicked_read_my_file)
        if read_train_my_file:
            try:
                df = read_pandas_file(uploaded_file, # '../../mushrooms.csv'
                                    add_current_date = add_current_date_select_info,
                                    header = 0, 
                                    index_col = None, 
                                    target_column = 'class')
                st.success('Dataframe Read Successfully', icon = "✅")
            except:
                st.success('Unsupported Dataframe or Error Occured During Dataframe Was Reading', icon = "🚨")
            try:
                df_pre = ClsLibPreprocessing(df)
                df_pre.scale_data()
                df_pre.encode_categorical_columns()
                df_pre.drop_duplicates()

                st.success('Dataframe Preprocessing Completed', icon = "✅")
            except:
                st.success('Dataframe Preprocessing Error', icon = "🚨")

            model = ClsLibMlXGBoostClassifier(data = df_pre.data)

            
            if train_method == 'GridSearchCV':

                model.test_size = float(test_size)
                model.target_column = str(target_column)
                model.eval_metric = ast.literal_eval(eval_metric)
                model.early_stopping_rounds = int(early_stopping_rounds)
                st.success('Training Begins', icon = "✅")
                try:
                    model.train_xgboost_model()
                    st.success('Training Completed', icon = "✅")
                    
                    plt.plot(list(model.eval_results['validation_0'].items())[0][1], label = 'Val Loss')
                    plt.plot(list(model.eval_results['validation_0'].items())[1][1], label = 'RMSE')
                    plt.plot(list(model.eval_results['validation_0'].items())[2][1], label = 'RMSLE')
                    plt.legend(loc = 'best')
                    plt.title('Validation Loss')
                    st.pyplot(plt)
                except:
                    st.success('An Error Occured During Training', icon = "🚨")




        
    