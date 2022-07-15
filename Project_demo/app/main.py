# import requirements needed
from flask import Flask, render_template,request
from utils import get_base_url
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12340
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')


def make_prediction(X_test):
    # For making the prediction
    housing_df = pd.read_csv("housing.csv")
    housing = housing_df.drop("median_house_value", axis=1)     #drop copy the orignal datframe into housing
    housing_labels = housing_df["median_house_value"].copy()    # splitting the predictor and target variable

    # housing stores the predictors
    housing_num = housing.drop("ocean_proximity", axis=1) # means numerical features
    housing_cat = housing[["ocean_proximity"]] # means categorical features

    num_attribs = list(housing_num)             # columns with numerical attributes
    cat_attribs = ["ocean_proximity"]           # columns with categorical attributes

    num_pipeline =  Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler())])                    
                              # Feature Scaling with Standard Scaler
        # Imputing Missing values

    full_pipeline = ColumnTransformer([
          ("num", num_pipeline, num_attribs),        
          ("cat", OneHotEncoder(), cat_attribs),
    ])
    housing.loc[len(housing.index)] = X_test
    housing_prepared = full_pipeline.fit_transform(housing)
    #print(housing_prepared.shape)

    X_train = housing_prepared[:-1]
    X_test = housing_prepared[-1].reshape(1, 13)
    y_train = housing_labels
    model = KNeighborsRegressor(n_neighbors = 11)
    model.fit(X_train,y_train)
    
    pred = model.predict(X_test)
    return abs(pred[0])
    
    

# set up the routes and logic for the webserver
@app.route(f'{base_url}' , methods = ["GET","POST"])
def home():
    if request.method == "POST":
        values = [i for i in request.form.values()]
        print(len(values))
        list_names = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                      'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
        # convert the list into dataframe row
        data = pd.DataFrame(values).T
 
        # add columns
        data.columns = list_names
        
        prediction = round(make_prediction(values))

        print("predicted housing value is:", prediction)
        
        return_text = f'The predicated value of the house is approximatley {prediction}.'
        
        return render_template('index.html' , return_text = return_text)
        
        
    return render_template('index.html')

# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc18.ai-camp.dev/'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
