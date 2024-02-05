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
    imputer = KNNImputer(n_neighbors=2, weights="distance")
    Evol_df['VOLUME'] = imputer.fit_transform(Evol_df[['VOLUME']])
#sidebar configuration
    with st.sidebar:
        Analyse_Exploratoire=st.selectbox('Statistiques mensuelles et globales', Analyses)
    if  Analyse_Exploratoire == 'Analyse_mensuelle': 
        Evol_df['DATE'] = pd.to_datetime(Evol_df['DATE'])
    # Grouper les données mensuellement
        monthly_data_grouped = Evol_df.resample('M',on='DATE').mean()
    # Créer un graphique Plotly Express
       
        mois_fr = {
    'January': 'janvier',
    'February': 'février',
    'March': 'mars',
    'April': 'avril',
    'May': 'mai',
    'June': 'juin',
    'July': 'juillet',
    'August': 'août',
    'September': 'septembre',
    'October': 'octobre',
    'November': 'novembre',
    'December': 'décembre'
}
         #fig0
        fig0 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='VOLUME', title='Volume Monthly Evolution', markers=True)
        fig0.update_traces(texttemplate='%{y:.2f}', textposition='top center', mode='markers+lines+text')
        fig0.update_xaxes(
        dtick='M1',  # Marquer tous les mois
        tickformat='%b %Y',  # Format de l'étiquette (abrégé du mois et année)
        tickangle=45,  # Angle de rotation des étiquettes (facultatif)
    )
        fig0.update_layout(width=700, height=500, bargap=0.1, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        monthly_data_grouped['Month'] = monthly_data_grouped.index.strftime('%B')
        # Trier le DataFrame par volume décroissant
        monthly_data_grouped2 = monthly_data_grouped.sort_values(by='VOLUME', ascending=False)
        # Sélectionner les mois avec les volumes les plus élevés 
        top_months = monthly_data_grouped2.head(3)
        top_months['Month'] = top_months['Month'].map(mois_fr)
        st.plotly_chart(fig0)
        st.write('Des pics de volumes moyens sont constatés au cours des mois de', ', '.join(top_months['Month']))
        
        #fig1
        fig1 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='MONTANT', title='CA Monthly Evolution', markers=True)
        fig1.update_traces(texttemplate='%{y:.2f}', textposition='top center', mode='markers+lines+text')
        fig1.update_xaxes(
        dtick='M1',  # Marquer tous les mois
        tickformat='%b %Y',  # Format de l'étiquette (abrégé du mois et année)
        tickangle=45,  # Angle de rotation des étiquettes (facultatif)
    )
        fig1.update_layout(width=370, height=500, bargap=0.1, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        monthly_data_grouped['Month'] = monthly_data_grouped.index.strftime('%B')
        # Trier le DataFrame par volume décroissant
        monthly_data_grouped2 = monthly_data_grouped.sort_values(by='MONTANT', ascending=False)
        # Sélectionner les mois avec les volumes les plus élevés 
        top_months1 = monthly_data_grouped2.head(3)
        top_months1['Month'] = top_months1['Month'].map(mois_fr)
        
        #fig2
        fig2 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='MARGE', title='Marge Monthly Evolution', markers=True)
        fig2.update_traces(texttemplate='%{y:.2f}', textposition='top center', mode='markers+lines+text')
        fig2.update_xaxes(
    dtick='M1',  # Marquer tous les mois
    tickformat='%b %Y',  # Format de l'étiquette (abrégé du mois et année)
    tickangle=45,  # Angle de rotation des étiquettes (facultatif)
)
        fig2.update_layout(width=410, height=500, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        monthly_data_grouped22 = monthly_data_grouped.sort_values(by='MARGE', ascending=False)
        top_months2 = monthly_data_grouped22.head(3)
        top_months2['Month'] = top_months2['Month'].map(mois_fr)
# Afficher les résultats
    # Afficher le graphique dans l'interface Streamlit
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1)
            st.write('Les CA moyens les plus élevées sont observées en ', ', '.join(top_months1['Month']))
        with col2:
            st.plotly_chart(fig2)
            st.write('Les marges moyennes les plus élevées sont enrégistrées en ', ', '.join(top_months2['Month']))
            
#Volumes par sites
        monthly_data_grouped = Evol_df.groupby(['ENTITE', pd.Grouper(key='DATE', freq='M')])['VOLUME'].sum().reset_index()

# Création du graphique à barres empilées avec des couleurs personnalisées
        fig3 = go.Figure()
        colors = ['chocolate','Peru','darkorange','deepskyblue','silver','lightyellow']  # Liste de couleurs personnalisées

        for i, entity in enumerate(monthly_data_grouped['ENTITE'].unique()):
              entity_data = monthly_data_grouped[monthly_data_grouped['ENTITE'] == entity]
              fig3.add_trace(go.Bar(
        x=entity_data['DATE'],
        y=entity_data['VOLUME'],
        name=entity,
        text=entity_data['DATE'].dt.strftime('%b %Y'),  # Format du texte (mois)
        textposition='inside',
        marker_color=colors[i % len(colors)]  # Choisissez une couleur de la liste en boucle
    ))
# Définir la date de début (janvier) et la date de fin (septembre)
        start_date = pd.to_datetime('2023-01-01')
        end_date = pd.to_datetime('2023-09-30')
        fig3.update_xaxes(
    range=[start_date, end_date],  # Plage de dates souhaitée
    dtick='M1',  # Marquer tous les mois
    tickformat='%b %Y',  # Format de l'étiquette (abrégé du mois et année)
    tickangle=45,  # Angle de rotation des étiquettes (facultatif)
)
# Personnaliser la mise en page pour enlever l'axe des abscisses
fig3.update_layout(
    barmode='stack',
    title='Cascade Bar Chart by Site',
    xaxis_title='DATE',
    yaxis_title='Volume',
    height=400,
    width=800,
    xaxis_showticklabels=False,  # Enlever les étiquettes de l'axe des abscisses
    xaxis_visible=False  # Rendre l'axe des abscisses invisible
)
# Créez une barre latérale pour la navigation entre les pages
page = st.sidebar.radio("Visualisation", ["Resumé","Analyse Exploratoire", "Techniques de Machine Learning"])
# Affichage conditionnel en fonction de la page sélectionnée

if page == "Resumé":
    RESUME()
elif page == "Analyse Exploratoire":
    page_dashboard()
elif page == "Techniques de Machine Learning":
    page_settings()


