import streamlit as st
import pandas as pd
import streamlit as  st
import plotly
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
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
from mlforecast.target_transforms import Differences
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.ticker import MaxNLocator

Analyses=('Analyse_mensuelle','Statistique par site')
Entité=("Marseille","Montoir","Dunkerque","Rouen", "LE HAVRE")
Approches=("Clustering RFM", "Logit Binaire")

    # Ajoutez le contenu de la page de tableau de bord ici
    #Import packages

#---------------------------------------------------------------------------------------
#                                  Sidebar configuration 
#---------------------------------------------------------------------------------------
  
# Chemin vers l'image de logo
# Création d'une mise en page en colonnes avec Streamlit
def RESUME():
    st.title("")
    col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
    with col1:
        image = Image.open('logo_rdt.jpg')
        st.image(image)
# Affichage du titre dans la deuxième colonne
    with col2:
        st.title('SITUATION CONMERCIALE À FIN DECEMBRE 2023')
        st.subheader("ANALYSE DATA SUR 2023")
#RESUME
    st.subheader("RESUME")
    st.write("Les statistiques présentées dans ce rapport se réfèrent à l’activité maritime de la Rhodanienne de Transit en 2023."  "Les données considérées couvrent la période de janvier à fin septembre 2023." )
    st.write("Afin de mieux appréhender la situation de l’activité commerciale sur les 3 trimestres de 2023, les données mensuelles du volume, du chiffre d’affaires facturé et de la marge ont été analysées dans la première partie.")
    st.write("Les statistiques par agence ont également été présentées en partie 2. Pour mieux catégoriser les clients, une segmentation par clustering (construction de grappes où les clients sont attribués à des grappes en fonctionde leurs caractéristiques R, F et M les plus proches) a été réalisée.)")


#
def page_dashboard():
    st.title("")
    col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
    with col1:
        image = Image.open('logo_rdt.jpg')
        st.image(image)
# Affichage du titre dans la deuxième colonne
    with col2:
        st.title('SITUATION CONMERCIALE À FIN DECEMBRE 2023')
#Import data
    Evol_df=pd.read_excel("Evolution_mensuelle_2023.xlsx")
    Stat_mens=pd.read_excel("Analyse_maritime.xlsx")
#sidebar configuration
    with st.sidebar:
        Analyse_Exploratoire=st.selectbox('Statistiques mensuelles et globales', Analyses)
    if  Analyse_Exploratoire == 'Analyse_mensuelle': 
        Evol_df['DATE'] = pd.to_datetime(Evol_df['DATE'])
    # Grouper les données mensuellement
        monthly_data_grouped = Evol_df.resample('M',on='DATE').mean()
    # Créer un graphique Plotly Express
        #fig1
        imputer = KNNImputer(n_neighbors=2, weights="distance")
        Evol_df['VOLUME'] = imputer.fit_transform(Evol_df[['VOLUME']])
        fig1 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='VOLUME', title='Volume Monthly Evolution', markers=True)
        fig1.update_traces(texttemplate='%{y:.2f}', textposition='top center', mode='markers+lines+text')
        fig1.update_xaxes(
        dtick='M1',  # Marquer tous les mois
        tickformat='%b %Y',  # Format de l'étiquette (abrégé du mois et année)
        tickangle=45,  # Angle de rotation des étiquettes (facultatif)
    )
        fig1.update_layout(width=350, height=500, bargap=0.1, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        #fig2
        fig2 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='MONTANT', title='CA Monthly Evolution', markers=True)
        fig2.update_traces(texttemplate='%{y:.2f}', textposition='top center', mode='markers+lines+text')
        fig2.update_xaxes(
    dtick='M1',  # Marquer tous les mois
    tickformat='%b %Y',  # Format de l'étiquette (abrégé du mois et année)
    tickangle=45,  # Angle de rotation des étiquettes (facultatif)
)
        fig2.update_layout(width=450, height=500, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    # Afficher le graphique dans l'interface Streamlit
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1)

        with col2:
            st.plotly_chart(fig2)


# Créez une barre latérale pour la navigation entre les pages
page = st.sidebar.radio("Visualisation", ["Resumé","Analyse Exploratoire", "Techniques de Machine Learning"])
# Affichage conditionnel en fonction de la page sélectionnée

if page == "Resumé":
    RESUME()
elif page == "Analyse Exploratoire":
    page_dashboard()
elif page == "Techniques de Machine Learning":
    page_settings()


