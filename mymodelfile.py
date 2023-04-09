import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

class MyModel():

    def __init__(self):

        self.model = LGBMRegressor(random_seed=42)

    def preprocess(self, df_1, df_2):

        df_2['Venue'] = df_2['Venue'].str.split(',').str[0]

        battes = np.unique(df_1['batter'].values)
        bowlers = np.unique(df_1['bowler'].values)
        teams = np.unique(df_2['Team1'].values)
        venues = np.unique(df_2['Venue'].values)

        self.batter_dict = {name: i for i, name in enumerate(battes)}
        self.bowler_dict = {name: i for i, name in enumerate(bowlers)}
        self.team_dict = {name: i for i, name in enumerate(teams)}
        self.venue_dict = {name: i for i, name in enumerate(venues)}

        df_1 = df_1[df_1['innings'] <= 2]
        df_1 = df_1[df_1['overs'] <= 5]

        df_3 = df_1.groupby(['ID', 'innings', 'BattingTeam'])['total_run'].sum().reset_index()

        df_4 = df_1.groupby(['ID', 'innings', 'BattingTeam'])['batter'].apply(lambda x: np.unique(x.values)).reset_index()

        df_5 = df_1.groupby(['ID', 'innings', 'BattingTeam'])['bowler'].apply(lambda x: np.unique(x.values)).reset_index()

        df_3 = df_3.merge(df_4, on=['ID', 'innings', 'BattingTeam'], how='left').merge(df_5, on=['ID', 'innings', 'BattingTeam'], how='left')

        df_3 = df_3.merge(df_2[['ID', 'Team1', 'Team2', 'Venue']], on='ID', how='left')

        df_3['BowlingTeam'] = np.where(df_3['BattingTeam'] == df_3['Team1'], df_3['Team2'], df_3['Team1'])

        df_3 = df_3.drop(['Team1', 'Team2'], axis=1)

        df = df_3.to_dict('records')

        X = []
        y = []

        for i, item in enumerate(df):

            venue_index = self.venue_dict[item['Venue']]
            batting_team_index = self.team_dict[item['BattingTeam']]
            bowling_team_index = self.team_dict[item['BowlingTeam']]
            innings_index = item['innings']

            batsman_index = np.zeros(11)

            for j, jtem in enumerate(item['batter']):
                batsman_index[j] = self.batter_dict[jtem]

            bowler_index = np.zeros(6)

            for j, jtem in enumerate(item['bowler']):
                bowler_index[j] = self.bowler_dict[jtem]

            X.append([venue_index, batting_team_index, bowling_team_index, innings_index, *batsman_index.tolist(), *bowler_index.tolist()])
            y.append(item['total_run'])

        X = np.array(X)
        y = np.array(y)

        return X, y
    
    def preprocess_test(self, df):

        df = df.to_dict('records')

        X = []

        for i, item in enumerate(df):

            if item['venue'] in self.venue_dict:
                venue_index = self.venue_dict[item['venue']]
            else:
                venue_index = len(self.venue_dict)/2

            if item['batting_team'] in self.team_dict:
                batting_team_index = self.team_dict[item['batting_team']]
            else:
                batting_team_index = len(self.team_dict)/2

            if item['bowling_team'] in self.team_dict:
                bowling_team_index = self.team_dict[item['bowling_team']]
            else:
                bowling_team_index = len(self.team_dict)/2

            innings_index = item['innings']

            batsmen = item['batsmen'].replace("'", "").replace("[", "").replace("]", "").split(",")
            batsman_index = np.zeros(11)

            for j, jtem in enumerate(batsmen):

                if jtem in self.batter_dict:
                    batsman_index[j] = self.batter_dict[jtem]
                else:
                    batsman_index[j] = len(self.batter_dict)/2

            bowlers = item['bowlers'].replace("'", "").replace("[", "").replace("]", "").split(",")
            bowler_index = np.zeros(6)

            for j, jtem in enumerate(bowlers):

                if jtem in self.bowler_dict:
                    bowler_index[j] = self.bowler_dict[jtem]
                else:
                    bowler_index[j] = len(self.bowler_dict)/2

            X.append([venue_index, batting_team_index, bowling_team_index, innings_index, *batsman_index.tolist(), *bowler_index.tolist()])

        X = np.array(X)

        return X

    def fit(self, df):

        df_1 = df[0]
        df_2 = df[1]

        X, y = self.preprocess(df_1, df_2)

        self.model.fit(X, y, categorical_feature=np.arange(X.shape[1]).tolist())

        return self
    
    def predict(self, df):

        X = self.preprocess_test(df)

        y_pred = self.model.predict(X)

        output = pd.DataFrame({'id': np.arange(len(y_pred)), 'predicted_runs': y_pred})

        return output