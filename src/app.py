#Import packages
import pandas as pd
import streamlit as  st
import plotly.offline as pyoff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image
from prophet import Prophet
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from mlforecast import MLForecast
import numba
from numba import njit
from sklearn.ensemble import RandomForestRegressor
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
from mlforecast.target_transforms import Differences
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS

#---------------------------------------------------------------------------------------
#                                  Sidebar configuration 
#---------------------------------------------------------------------------------------
HOME_AIRPORTS=('LGW','LGW','LIS','LIS','SSA','NTE','LYS','PNH','POP','SCL')
PAIRED_AIRPORTS=('BCN','AMS','ORY','OPO','GRU','FUE','PIS','NGB','JFK','LHR')

# Chemin vers l'image de logo
# Création d'une mise en page en colonnes avec Streamlit
col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
with col1:
   image = Image.open('C:/Users/AYI KPADONOU Aldah/Pictures/pax.jpg')
   st.image(image)
# Affichage du titre dans la deuxième colonne
with col2:
    st.title('Trafic Forecaster')
#Import data
df=pd.read_parquet("D:/Documents/traffic_10lines.parquet")
df.info()
#sidebar configuration
with st.sidebar:
    home_airport=st.selectbox('Home Airport',HOME_AIRPORTS)
    paired_airport=st.selectbox('Paired Airport',PAIRED_AIRPORTS)
    nb_days=st.slider('Days of forecast',7,30,1)
st.write('Home Airport selected:',home_airport)
st.write('Paired Airport selected:',paired_airport)
st.write('Day selected:',nb_days)

#---------------------------------------------------------------------------------------
#                                     Plotly graph
#---------------------------------------------------------------------------------------
def draw_ts_multiple(df: pd.DataFrame, v1: str, v2: str=None, prediction: str=None, date: str='date',
              secondary_y=True, covid_zone=False, display=True):
  """Draw times series possibly on two y axis, with COVID period option.

  Args:
  - df (pd.DataFrame): time series dataframe (one line per date, series in columns)
  - v1 (str | list[str]): name or list of names of the series to plot on the first x axis
  - v2 (str): name of the serie to plot on the second y axis (default: None)
  - prediction (str): name of v1 hat (prediction) displayed with a dotted line (default: None)
  - date (str): name of date column for time (default: 'date')
  - secondary_y (bool): use a secondary y axis if v2 is used (default: True)
  - covid_zone (bool): highlight COVID-19 period with a grayed rectangle (default: False)
  - display (bool): display figure otherwise just return the figure (default: True)

  Returns:
  - fig (plotly.graph_objs._figure.Figure): Plotly figure generated

  Notes:
  Make sure to use the semi-colon trick if you don't want to have the figure displayed twice.
  Or use `display=False`.
  """
  if isinstance(v1, str):
    variables = [(v1, 'V1')]
  else:
    variables = [(v, 'V1.{}'.format(i)) for i, v in enumerate(v1)]
  title = '<br>'.join([n + ': '+ v for v, n in variables]) + ('<br>V2: ' + v2) if v2 else '<br>'.join([v + ': '+ n for v, n in variables])
  layout = dict(
    title=title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
  )
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.update_layout(layout)
  for v, name in variables:
    fig.add_trace(go.Scatter(x=df[date], y=df[v], name=name), secondary_y=False)
  if v2:
    fig.add_trace(go.Scatter(x=df[date], y=df[v2], name='V2'), secondary_y=secondary_y)
    fig['layout']['yaxis2']['showgrid'] = False
    fig.update_yaxes(rangemode='tozero')
    fig.update_layout(margin=dict(t=125 + 30 * (len(variables) - 1)))
  if prediction:
    fig.add_trace(go.Scatter(x=df[date], y=df[prediction], name='^V1', line={'dash': 'dot'}), secondary_y=False)

  if covid_zone:
    fig.add_vrect(
        x0=pd.Timestamp("2020-03-01"), x1=pd.Timestamp("2022-01-01"),
        fillcolor="Gray", opacity=0.5,
        layer="below", line_width=0,
    )
  
  return fig

Plot=draw_ts_multiple(
    (df
     .query("home_airport == @home_airport and paired_airport == @paired_airport")
     .groupby(['home_airport', 'paired_airport', 'date'])
     .agg(pax_total=('pax', 'sum'))
     .reset_index()
    ),
    'pax_total',
    covid_zone=True,
)
st.subheader("Time series in plotly format")
st.plotly_chart(Plot)


#---------------------------------------------------------------------------------------
#                              Prophet Model : Modele et Graph
#---------------------------------------------------------------------------------------

def generate_route_df(df:pd.DataFrame,homeAirport:str, pairedAirport:str)-> pd.DataFrame:
  _df=(df
       .query('home_airport== @home_airport and paired_airport==@paired_airport'.format(home=homeAirport,paired=pairedAirport))
       .groupby(['home_airport','paired_airport','date'])
       .agg(pax_total=('pax','sum'))
       .reset_index()
       )
  return _df
generate_route_df(df,"@home_airport","@paired_airport")

dt=generate_route_df(df, "@home_airport","@paired_airport").rename(columns={'date':'ds','pax_total':'y'})

def prediction(df):
    baseline_model=Prophet()
    baseline_model.fit(dt)
    
    # Effectuer la prédiction pour les prochains jours
    future = baseline_model.make_future_dataframe(periods=nb_days)
    forecast = baseline_model.predict(future)
    
    # Afficher le résultat de la prédiction
    st.write(forecast)
    

def main():
    # Charger les données (remplacez cette étape par vos propres données)
    data = dt # Chargez vos données ici
    
    st.subheader("Prophet model prediction")
    # Ajouter un bouton pour déclencher la prédiction
       
    if st.sidebar.button("Forecast_Prophet"):
        # Appeler la fonction prediction lorsque le bouton est cliqué
        st.subheader("Input data")
        st.write(data)
        st.subheader("The Prophet model")
        prediction(data)
        df_prophet = dt[['ds', 'y']].copy()
        df_prophet = df_prophet.rename(columns={'ds': 'ds', 'y': 'y'})
        Plot = df_prophet.rename(columns={'date': 'ds', 'pax_total': 'y'})
         # Afficher les données
        
# Création du modèle Prophet
        modele = Prophet()
        modele.fit(df_prophet)

# Préparation des dates pour la prédiction
        dates_prediction = modele.make_future_dataframe(periods=nb_days)  

# Prédiction avec Prophet
        forecast = modele.predict(dates_prediction)

# Création du graphique Plotly
        fig = go.Figure()

# Ajout des données historiques
        fig.add_trace(go.Scatter(
       x=Plot['ds'],
       y=Plot['y'],
       mode='lines',
       name='Données historiques',
       line=dict(color='blue')
))

# Ajout des données prédites
        fig.add_trace(go.Scatter(
       x=forecast['ds'],
       y=forecast['yhat'],
       mode='lines',
       name='Prédiction',
       line=dict(color='black', dash='dash')
))

# Personnalisation du graphique
        fig.update_layout(
       xaxis_title='Date',
       yaxis_title='Valeur',
)
        fig.layout = dict(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
  )
        covid_zone=True
        if covid_zone:
         fig.add_vrect(
         x0=pd.Timestamp("2020-03-01"), x1=pd.Timestamp("2022-01-01"),
         fillcolor="Gray", opacity=0.5,
         layer="below", line_width=0,
    )
        st.subheader("Graph of prediction results versus historical data")  
        st.plotly_chart(fig)
if __name__ == '__main__':
    main()

#---------------------------------------------------------------------------------------
#                                 Others models
#---------------------------------------------------------------------------------------

# Créer une instance de MLForecast avec les modèles souhaités

# Création des modèles
LGBM = lgb.LGBMRegressor()
XGB = xgb.XGBRegressor()
Random_Forest = RandomForestRegressor(random_state=0)

tested_models = [
    LGBM,
    XGB,
    Random_Forest,
]

# Interface utilisateur
with st.sidebar:
    model_option = st.selectbox(
        'Choose another model',
        ('MLForcast:LGBM,XGB,Random_Forest', 'Neural Forcast')
    )

@njit
def rolling_mean_28(x):
    return rolling_mean(x, window_size=28)

# Configuration de MLForecast
fcst = MLForecast(
    models=tested_models,
    freq='D',
    lags=[7, 14],
    lag_transforms={
        1: [expanding_mean],
        7: [rolling_mean_28]
    },
    date_features=['dayofweek'],
    target_transforms=[Differences([1])],
)

horizon = 90
models = [NBEATS(input_size=2 * horizon, h=horizon, max_epochs=50),
          NHITS(input_size=2 * horizon, h=horizon, max_epochs=50)]

def predict_model(df):
   if model_option== "MLForcast:LGBM,XGB,Random_Forest":
      fcst.fit(generate_route_df(df, "@home_airport", "@paired_airport").drop(columns=['paired_airport']),
      id_col='home_airport', time_col='date', target_col='pax_total')
      forecast_1 = fcst.predict(nb_days)
      return forecast_1
   else :
      nforecast = NeuralForecast(models=models, freq='D')
      nforecast.fit(generate_route_df(df, "@home_airport", "@paired_airport")
              .drop(columns=['paired_airport'])
              .rename(columns={'home_airport': 'unique_id', 'date': 'ds', 'pax_total': 'y'}))
      forecast_2 = nforecast.predict().reset_index()
      return forecast_2

def main2():
    # Chargement des données (remplacez cette étape par le chargement de vos propres données)
    df=pd.read_parquet("D:/Documents/traffic_10lines.parquet")  # Remplacez load_data() par votre fonction de chargement de données

    st.subheader("Prediction of other models")
    st.write('You have selected:', model_option)

    # Ajout d'un bouton pour déclencher la prédiction
    if st.sidebar.button("Prediction"):
        
        # Prédiction avec le modèle sélectionné
        forecast = predict_model(df)
        
        # Affichage des prédictions
        st.write(forecast)
        nforecast= NeuralForecast(models=models, freq='D')
    
        if model_option== "MLForcast:LGBM,XGB,Random_Forest":
            nixtla_model=fcst.fit(generate_route_df(df, "@home_airport", "@paired_airport").drop(columns=['paired_airport']),
            id_col='home_airport', time_col='date', target_col='pax_total')
            fig1= draw_ts_multiple((pd.concat([generate_route_df(df, '@home_airport', '@paired_airport').drop(columns=['paired_airport']),
                             nixtla_model.predict(7*10)])),
                 v1='pax_total', v2='LGBMRegressor');
            st.plotly_chart(fig1) 
        
        else:
           nforecast = NeuralForecast(models=models, freq='D')
           nforecast.fit(generate_route_df(df, "@home_airport", "@paired_airport").drop(columns=['paired_airport']).rename(columns={'home_airport': 'unique_id',
                                                                                                      'date': 'ds','pax_total': 'y'}))
           fig2= draw_ts_multiple((pd.concat([generate_route_df(df, '@home_airport', '@paired_airport').drop(columns=['paired_airport']),
                             nforecast.predict().reset_index()])),v1='pax_total', v2='NBEATS');
           st.plotly_chart(fig2)      
if __name__ == '__main__':
    main2()

