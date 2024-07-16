import pandas as pd
import matplotlib as plt
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from mlxtend.plotting import heatmap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

#########################################################
######## Plaatsingswijzer ###############################
#########################################################

methodetoets_6 = pd.read_csv("2_methodetoets_6.csv")
methodetoets_7 = pd.read_csv("2_methodetoets_7.csv")
methodetoets_8 = pd.read_csv("2_methodetoets_8.csv")
plaatsingswijzer = pd.read_csv("2_plaatsingswijzer.csv")
tweede_advies = pd.read_csv("2_tweede_advies.csv")

weegfactoren = {'Begrijpend lezen': 1.4, 'Rekenen en wiskunde': 1.4, 'DMT': 0.6, 'Spelling': 0.6}

#Lijst om gemiddelde scores per niveau op te slaan
gemiddelde_scores = []

#Groepeer gegevens per niveau
for niveau, groep in plaatsingswijzer.groupby('Niveau'):
    gewogen_scores = []
    # gewogen gemiddelde score per vak
    for index, row in groep.iterrows():
        gewogen_score = (row['Begrijpend lezen'] * weegfactoren['Begrijpend lezen'] +
                         row['Rekenen en wiskunde'] * weegfactoren['Rekenen en wiskunde'] +
                         row['DMT'] * weegfactoren['DMT'] +
                         row['Spelling'] * weegfactoren['Spelling']) / sum(weegfactoren.values())
        gewogen_scores.append(gewogen_score)
    gemiddelde_score = sum(gewogen_scores) / len(gewogen_scores)
    gemiddelde_scores.append((niveau, gemiddelde_score))

gemiddelde_scores.sort(key=lambda x: x[1], reverse=False)
# Converteer de lijst naar een DataFrame
gewogen_gemiddelde = pd.DataFrame(gemiddelde_scores, columns=['Niveau', 'Gemiddelde score'])

print(gewogen_gemiddelde)

advies_plaatsingswijzer = gewogen_gemiddelde.loc[gewogen_gemiddelde['Niveau'] == 'VWO', 'Gemiddelde score'].values[0]

print("Het advies op basis van de plaatsingswijzer is:", advies_plaatsingswijzer)




##################################################################################################
############ Methode toetsen #####################################################################
##################################################################################################

methodetoets_6[['Taal', 'Rekenen en wiskunde', 'Spelling', 'Engels', 'Wereld orientatie']] = methodetoets_6[['Taal', 'Rekenen en wiskunde', 'Spelling', 'Engels', 'Wereld orientatie']].div(2)
methodetoets_7[['Taal', 'Rekenen en wiskunde', 'Spelling', 'Engels', 'Wereld orientatie']] = methodetoets_7[['Taal', 'Rekenen en wiskunde', 'Spelling', 'Engels', 'Wereld orientatie']].div(2)
methodetoets_8[['Taal', 'Rekenen en wiskunde', 'Spelling', 'Engels', 'Wereld orientatie']] = methodetoets_8[['Taal', 'Rekenen en wiskunde', 'Spelling', 'Engels', 'Wereld orientatie']].div(2)

weegfactoren_toetsen = {'Taal': 1.4, 'Rekenen en wiskunde': 1.4, 'Spelling': 0.8, 'Engels': 0.7, 'Wereld orientatie': 0.7}


gewogen_gemiddelde_score_groep_6 = (methodetoets_6['Taal'] * weegfactoren_toetsen['Taal'] +
                                    methodetoets_6['Rekenen en wiskunde'] * weegfactoren_toetsen['Rekenen en wiskunde'] +
                                    methodetoets_6['Spelling'] * weegfactoren_toetsen['Spelling'] +
                                    methodetoets_6['Engels'] * weegfactoren_toetsen['Engels'] +
                                    methodetoets_6['Wereld orientatie'] * weegfactoren_toetsen['Wereld orientatie']) \
                                    / sum(weegfactoren_toetsen.values())

gewogen_gemiddelde_score_groep_6 = gewogen_gemiddelde_score_groep_6.mean()

gewogen_gemiddelde_score_groep_7 = (methodetoets_7['Taal'] * weegfactoren_toetsen['Taal'] +
                                    methodetoets_7['Rekenen en wiskunde'] * weegfactoren_toetsen['Rekenen en wiskunde'] +
                                    methodetoets_7['Spelling'] * weegfactoren_toetsen['Spelling'] +
                                    methodetoets_7['Engels'] * weegfactoren_toetsen['Engels'] +
                                    methodetoets_7['Wereld orientatie'] * weegfactoren_toetsen['Wereld orientatie']) \
                                    / sum(weegfactoren_toetsen.values())

gewogen_gemiddelde_score_groep_7 = gewogen_gemiddelde_score_groep_7.mean()

gewogen_gemiddelde_score_groep_8 = (methodetoets_8['Taal'] * weegfactoren_toetsen['Taal'] +
                                    methodetoets_8['Rekenen en wiskunde'] * weegfactoren_toetsen['Rekenen en wiskunde'] +
                                    methodetoets_8['Spelling'] * weegfactoren_toetsen['Spelling'] +
                                    methodetoets_8['Engels'] * weegfactoren_toetsen['Engels'] +
                                    methodetoets_8['Wereld orientatie'] * weegfactoren_toetsen['Wereld orientatie']) \
                                    / sum(weegfactoren_toetsen.values())

gewogen_gemiddelde_score_groep_8 = gewogen_gemiddelde_score_groep_8.mean()

totaalscore_toetsen = (gewogen_gemiddelde_score_groep_6 + gewogen_gemiddelde_score_groep_7 + gewogen_gemiddelde_score_groep_8)/3
print(totaalscore_toetsen)


###################################################
############## WORK ETHIC #########################
###################################################

werkhouding_leerling_6 = pd.read_csv("1_motivatie_leerling_groep6.csv")
werkhouding_docent_6 = pd.read_csv("1_motivatie_docent_groep6.csv")
werkhouding_leerling_7 = pd.read_csv("1_motivatie_leerling_groep7.csv")
werkhouding_docent_7 = pd.read_csv("1_motivatie_docent_groep7.csv")
werkhouding_leerling_8 = pd.read_csv("1_motivatie_leerling_groep8.csv")
werkhouding_docent_8 = pd.read_csv("1_motivatie_docent_groep8.csv")

### Leerling
werkhouding_678_leerling = []

gemiddelde_6_leerling = werkhouding_leerling_6["Score"].mean()
werkhouding_678_leerling.append(gemiddelde_6_leerling)

gemiddelde_7_leerling = werkhouding_leerling_7["Score"].mean()
werkhouding_678_leerling.append(gemiddelde_7_leerling)

gemiddelde_8_leerling = werkhouding_leerling_8["Score"].mean()
werkhouding_678_leerling.append(gemiddelde_8_leerling)

### Docent

werkhouding_678_docent = []

gemiddelde_6_docent = werkhouding_docent_6["Score"].mean()
werkhouding_678_docent.append(gemiddelde_6_docent)

gemiddelde_7_docent = werkhouding_docent_7["Score"].mean()
werkhouding_678_docent.append(gemiddelde_7_docent)


gemiddelde_8_docent = werkhouding_docent_8["Score"].mean()
werkhouding_678_docent.append(gemiddelde_8_docent)

werkhouding = pd.DataFrame({"Werkhouding gemeten door leerling": werkhouding_678_leerling, "Werkhouding gemeten door docent": werkhouding_678_docent, "Groep": [6, 7, 8]})

werkhouding_leerling_mean = werkhouding["Werkhouding gemeten door leerling"].mean()
werkhouding_docent_mean = werkhouding["Werkhouding gemeten door docent"].mean()

werkhouding_mean = (werkhouding_docent_mean + werkhouding_leerling_mean)/2

##############################################################
##################### Berekend eerste advies #################
##############################################################
niveaus = ["VMBO-basis", "VMBO-kader", "VMBO-tl", "Havo", "VWO"]

eerste_advies = (advies_plaatsingswijzer * 0.9 + totaalscore_toetsen * 1.05 + werkhouding_mean * 1.05) / 3

if eerste_advies >= 4.2:
    eerste_niveau = "VWO"
    eerste_advies_nummer = 5
elif eerste_advies >= 3.4 and eerste_advies < 4.2:
    eerste_niveau = "Havo"
    eerste_advies_nummer = 4
elif eerste_advies >= 2.6 and eerste_advies < 3.4:
    eerste_niveau = "VMBO-tl"
    eerste_advies_nummer = 3
elif eerste_advies >= 1.8 and eerste_advies < 2.6:
    eerste_niveau = "VMBO-kader"
    eerste_advies_nummer = 2
else:
    eerste_niveau = "VMBO-basis"
    eerste_advies_nummer = 1


####################################
####### Tweede advies ##############
####################################

tweede_advies= tweede_advies["Tweede advies"].iloc[0]

count = 0
for advies in niveaus:
    count += 1
    if tweede_advies == advies:
        tweede_advies_nummer = count
        break

if tweede_advies_nummer > eerste_advies_nummer:
    niet_wel = "will"
    if eerste_advies >= 4.0:
        nieuw_advies = "VWO"
    elif eerste_advies >= 3.2:
        nieuw_advies = "Havo"
    elif eerste_advies >= 2.4:
        nieuw_advies = "VMBO-tl"
    elif eerste_advies >= 1.6:
        nieuw_advies = "VMBO-kader"
else:
    niet_wel = "will not"
    print("Het advies wordt niet bijgesteld op basis van de doorstroomtoets")


######################################################
################# VOORSPELLEN ########################
######################################################

data_set_prediction = pd.read_csv("Random_data_met_advies_rf.csv")
forest_classification_1 = data_set_prediction.dropna(subset=['Uitstroom_8_jaar'])


target = 'Uitstroom_8_jaar'
features = forest_classification_1.iloc[:, list(range(4)) + list(range(20, 35))]

#features = forest_classification_1.iloc[:, :35]

X = features.values
y = forest_classification_1[target].values
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=123)

forest = RandomForestRegressor(
    n_estimators=1000,
    criterion='squared_error',
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=1,
    n_jobs=-1)

forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f'MAE train: {mae_train:.2f}')
print(f'MAE test: {mae_test:.2f}')

r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')

plt.figure(figsize=(20,10))
plot_tree(forest.estimators_[0], feature_names=features.columns, filled=True, rounded=True)
plt.show()

########## daadwerkelijke voorspelling ###############

single_row_data = pd.read_csv("voorspeldata 2.csv")

features_single_row = single_row_data.iloc[:, list(range(4)) + list(range(20, 35))].values

#features_single_row = single_row_data.iloc[:, :35].values

predicted_value = forest.predict(features_single_row)

if predicted_value[0] >= 4.2:
    prediction = "VWO"
elif predicted_value[0] >= 3.4 and predicted_value[0] < 4.2:
    prediction = "Havo"
elif predicted_value[0] >= 2.6 and predicted_value[0] < 3.4:
    prediction = "VMBO-tl"
elif predicted_value[0] >= 1.8 and predicted_value[0] < 2.6:
    prediction = "VMBO-kader"
else:
    prediction = "VMBO-basis"

print(f"Voorspelde waarde voor Uitstroom_8_jaar: {prediction}")

########### Belangrijkste variabelen #################

importances = forest.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Belangrijkste kenmerken voor de voorspelling:")
print(feature_importance_df.head(5))

top_5_feature_names = feature_importance_df['Feature'].head(5)

################################################################
################## DASH BOARD ##################################
################################################################

niveaus = plaatsingswijzer['Niveau'].unique()

methodetoets_6.reset_index(drop=True, inplace=True)
methodetoets_7.reset_index(drop=True, inplace=True)
methodetoets_8.reset_index(drop=True, inplace=True)


#dash-app
app = dash.Dash(__name__)

app.title = "Dashboard 2"
#Layout van de app
app.layout = html.Div([
    html.Label('Select a high school level:'),
    dcc.Dropdown(
        id='niveau-dropdown',
        options=[{'label': niveau, 'value': niveau} for niveau in niveaus],
        value=niveaus[0]  #Standaardwaarde is het eerste niveau in de lijst
    ),
    dcc.Graph(id='line-plot'),
    dcc.Graph(id = "line-plot-2"),
    dcc.Graph(id= "line-plot-3"),
    html.Div([
        html.Label("First advice based on the tracking system tests, method tests and work ethic:"),
        html.P(f"{eerste_niveau}"),
        html.Label("Second advice based on the transition test:"),
        html.P(f"The second advice is: {tweede_advies}, and {niet_wel} be adjusted to above"),
        html.P(f"The final advice is: {nieuw_advies}") if niet_wel == "will" else None,
        html.Label("Prediction:"),
        html.P(f"The predicted high school level based on an AI algorithm: {prediction}"),
        html.P(f"Predicting attributes are: {', '.join(top_5_feature_names)}")
    ])
])


#bijwerken van de grafiek op basis van de geselecteerde niveau
@app.callback(
    Output('line-plot', 'figure'),
    [Input('niveau-dropdown', 'value')]
)

def update_graph(selected_niveau):
    #Filter de DF op het geselecteerde niveau
    niveau_data = plaatsingswijzer[plaatsingswijzer['Niveau'] == selected_niveau]

    niveau_data.rename(columns={
        'Begrijpend lezen': 'Reading Comprehension',
        'Rekenen en wiskunde': 'Mathematics and Arithmetic',
        'DMT': 'Technical Reading',
        'Spelling': 'Grammar'
    }, inplace=True)

    gem_niveau = gewogen_gemiddelde.loc[gewogen_gemiddelde['Niveau'] == selected_niveau, 'Gemiddelde score'].values[0]

    #lijnplot met de verschillende vakken als lijnen
    fig = px.line(niveau_data, x='Groep', y=['Reading Comprehension', 'Mathematics and Arithmetic', 'Technical Reading', 'Grammar'],
                  title=f'Scores per tracking test for high school level: {selected_niveau}',
                  labels={'Groep': 'Grade'})

    fig.update_yaxes(range=[0, 6])

    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text=f'The average score for {selected_niveau} is: {gem_niveau:.2f}',
        showarrow=False,
        font=dict(
            family="Arial",
            size=18,
            color="black"
        )
    )

    return fig

@app.callback(
    Output('line-plot-2', 'figure'),
    [Input('niveau-dropdown', 'value')]
)
def update_graph(selected_niveau):
    combined_data = pd.concat([methodetoets_6, methodetoets_7, methodetoets_8], axis=0)
    combined_data.reset_index(drop=True, inplace=True)

    combined_data.rename(columns={
        'Taal': 'Dutch language',
        'Rekenen en wiskunde': 'Mathematics and Arithmetic',
        'Spelling': 'Grammar',
        'Engels': 'English',
        'Wereld orientatie': 'World orientation'
    }, inplace=True)
    #lijnplot voor het geselecteerde niveau
    fig = px.line(combined_data, x='Groep', y=['Dutch language', 'Mathematics and Arithmetic', 'Grammar', 'English', 'World orientation'],
                  title = f'Scores of the method tests of grade 6,7 and 8',
                  labels = {'Groep': 'Grade'})
    fig.update_yaxes(range=[0, 6])  # Zorg ervoor dat de y-as geschikt is voor de scores

    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text=f"""The weighted average of 6th grade: {gewogen_gemiddelde_score_groep_6:.2f}
             The weighted average of 7th grade: {gewogen_gemiddelde_score_groep_7:.2f}
             The weighted average of 8th grade: {gewogen_gemiddelde_score_groep_8:.2f}""",
        showarrow=False,
        font=dict(
            family="Arial",
            size=18,
            color="black"
        )
    )

    return fig

@app.callback(
    Output('line-plot-3', 'figure'),
    [Input('niveau-dropdown', 'value')]
)
def update_graph(selected_niveau):
    werkhouding.rename(columns={
        'Werkhouding gemeten door leerling': 'Work ethic as measured by student',
        'Werkhouding gemeten door docent': 'Work ethic as measured by teachers'
    }, inplace=True)
    fig = px.line(werkhouding, x='Groep', y=["Work ethic as measured by student", "Work ethic as measured by teachers"],
                  title = f'Work ethic in grades 6,7 and 8',
                  labels={'Groep': 'Grade'})
    fig.update_yaxes(range=[0, 6])  # Zorg ervoor dat de y-as geschikt is voor de scores

    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text=f"""Average work ethic as measured by student: {werkhouding_leerling_mean:.2f}
               Average work ethic as measured by teachers: {werkhouding_docent_mean:.2f}""",
        showarrow=False,
        font=dict(
            family="Arial",
            size=18,
            color="black"
        )
    )


    return fig

#starting dash
if __name__ == '__main__':
    app.run_server(debug=True)



