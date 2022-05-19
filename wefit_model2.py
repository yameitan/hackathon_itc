import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from flask import Flask
from flask import request
import requests
import json



# all_feats = []
dist_feats = ['x_coordinate', 'y_coordinate']
ir_feats = ["_id", "email", "password", "__v", 'mock_id']
sport_feats=['gender', 'age', 'fitness_level', 'exercise_frequency',
       'participate_running', 'participate_gym', 'participate_team',
       'participate_dance', 'participate_yoga', 'participate_swimming',
       'participate_lifting_weights', 'time_of_exercise_afternoon',
       'time_of_exercise_evening', 'health', 'area', 'x_coordinate',
       'y_coordinate', 'time_of_exercise_early_morning']


app = Flask(__name__)

@app.route('/get_matches')
def get_matches():
    print('1')
    id = int(request.args.get('id'))
    print(id)
    print('2')
    user_r = requests.get(f'https://itc-hackathon-be.herokuapp.com/user/query?mock_id={id}').json()
    print(user_r)
    print('3')
    user = pd.DataFrame(user_r['data'])[sport_feats+dist_feats+['area', 'dist']].astype('float64')
    print('4')
    data_request = requests.get(f'https://itc-hackathon-be.herokuapp.com/user/query?area={float(user.dist.values[0])}').json()
    print(5)
    filter_data = pd.DataFrame(data_request['data'])[sport_feats+dist_feats+['dist']].astype('float64')
    print(6)


    num_users = len(filter_data)
    print('a')
    neigh = NearestNeighbors(n_neighbors=100, metric='cosine')
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('neigh', neigh)])
    pipe.fit(filter_data[sport_feats])

    dist_filter = NearestNeighbors(n_neighbors=num_users)
    dist_filter.fit(filter_data[['x_coordinate', 'y_coordinate']].values)

    order = pipe.named_steps['neigh'].kneighbors(user[sport_feats].values.reshape(1, -1))[1][0][1:]
    users = dist_filter.radius_neighbors([user[dist_feats].values], dist)[1][0]
    user_order = []
    for user in order:
        if user in users:
            user_order.append(user)
    return str(user_order)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)