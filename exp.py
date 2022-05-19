import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from flask import Flask
from flask import request

data = pd.read_csv('hacakathon_data.csv')
data = data.astype('float64')
all_feats = data.columns
dist_feats=['x_coordinate', 'y_coordinate']
other_feats = data.drop(columns=['x_coordinate', 'y_coordinate', 'area']).columns

app = Flask(__name__)

@app.route('/get_matches')
def get_matches():
    id = int(request.args.get('id'))
    dist = float(request.args.get('dist'))
    user = data.iloc[id]
    filter_data = data[data.area == user.area]



    num_users = len(filter_data)
    neigh = NearestNeighbors(n_neighbors=num_users, metric='minkowski')
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('neigh', neigh)])
    pipe.fit(filter_data.drop(columns=['x_coordinate', 'y_coordinate', 'area']))
    user_sport = pipe.named_steps['scaler'].transform(user[other_feats].values.reshape(1, -1))
    dist_filter = NearestNeighbors(n_neighbors=num_users)
    dist_filter.fit(filter_data[['x_coordinate', 'y_coordinate']].values)

    order = pipe.named_steps['neigh'].kneighbors(user_sport)[1][0][1:]
    print(order)
    users = dist_filter.radius_neighbors([user[dist_feats].values], dist)[1][0]
    user_order = []
    for user in order:
        if user in users:
            user_order.append(user)
    return str(user_order)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

