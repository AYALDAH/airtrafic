                                                           #---------------------------------------------------------------------------------------
                                                           #                                  Sidebar configuration 
                                                           #---------------------------------------------------------------------------------------
#Importation des packages
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

                                                    #---------------------------------------------------------------------------------------
                                                    #                                  Sidebar configuration et Acceuil
                                                    #---------------------------------------------------------------------------------------
#Définir les variables 
Analyses=('Analyse Mensuelle','Analyse globale des sites','Analyse par sites')
ENTITE=("MARSEILLE","MONTOIR","DUNKERQUE","ROUEN", "LE HAVRE")
Approches=("Clustering RFM", "PROPHET")

# Chemin vers l'image de logo
# Création d'une mise en page en colonnes avec Streamlit
def ACCUEIL():
    st.title("")
    col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
    with col1:
        image = Image.open('logo_rdt.jpg')
        st.image(image)
# Affichage du titre dans la deuxième colonne
    with col2:
        st.title('ACCUEIL')
        #st.subheader("RESUME")
#RESUME
    st.subheader("RESUME")
    st.write("Les statistiques présentées dans ce rapport se réfèrent à l’activité maritime de la Rhodanienne de Transit en 2023. "  "Les données considérées couvrent la période de janvier à fin Décembre." )
    st.write("Afin de mieux appréhender la situation de l’activité commerciale sur les 4 trimestres de 2023, les données mensuelles du volume, du chiffre d’affaires facturé et de la marge ont été analysées dans la première partie.")
    st.write("Les statistiques par agence ont également été présentées et une section dédiée à la prédiction des ventes et des volumes sur les prochains mois a été mise en place.")
    # Pour mieux catégoriser les clients, une segmentation par clustering (construction de grappes où les clients sont attribués à des grappes en fonctionde leurs caractéristiques R, F et M les plus proches) a été réalisée.
#Mode D'emploi
    st.subheader("MODE D'EMPLOI")
    st.write("-Selectionnez la page à visualiser sur la barre latérale gauche" )
    st.write("-Naviguer sur les différentes pages en selectionnant les options" )
    st.write("I-Si vous choisissez la page Analyse Exploratoire, vous avez plusieurs options dans le menu déroulant.")
    st.write("La première option est Analyse Mensuelle, qui prend en compte les détails mensuels par site (Choisir le site dans le deuxième menu déroulant)")
    st.write("La deuxième option est Analyse globale des sites, qui présente de manière globale des statistiques sur les sites")
    st.write("La troisième option de cette page donne la possibilité de visualiser les statistiques de chaque individuellement en choisissant le bouton du site qui vous intéresse.")
  # Affichage des bâteaux
    cola, colb = st.columns(2)
    with cola:
        st.image("boat_new.gif")
    with colb:
        st.image("boat_new.gif")


    
                                                            #---------------------------------------------------------------------------------------
                                                            #                                  Analyse Mensuelle
                                                            #---------------------------------------------------------------------------------------
#Definir la page d'exploration des données
def page_dashboard():
    st.title("")
    col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
    with col1:
        image = Image.open('logo_rdt.jpg')
        st.image(image)
# Affichage du titre dans la deuxième colonne
    with col2:
        st.title('SITUATION CONMERCIALE EN 2023')

  #Importation et traitement des données mensuelles 
    Evol_df=pd.read_excel("Evolution_mensuelle_2023.xlsx")
    Stat_mens=pd.read_excel("Analyse_maritime.xlsx")
    imputer = KNNImputer(n_neighbors=5)
    Evol_df['VOLUME'] = imputer.fit_transform(Evol_df[['VOLUME']])
  
#Configuration de la barre pour le choix des analyses
    with st.sidebar:
        Analyse_Exploratoire=st.selectbox('Analyses mensuelles et par sites', Analyses)
    if  Analyse_Exploratoire == 'Analyse Mensuelle':
        st.write("**ANALYSE GLOBALE MENSUELLE**")
        Evol_df['DATE'] = pd.to_datetime(Evol_df['DATE'])
# Grouper les données mensuellement
        monthly_data_grouped = Evol_df.resample('M',on='DATE').mean()
# Créer un graphique Plotly Express
#Définir les Mois en Français
       
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
         #Evolution mensuelle du volume
      
        fig0 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='VOLUME', title='Evolution mensuelle du Volume', markers=True)
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
        #Commentaire
        st.write('Des pics de volumes moyens sont constatés au cours des mois de', ', '.join(top_months['Month']))
        
        #Evolution menuelle du CA
      
        fig1 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='MONTANT', title='Evolution mensuelle du CA', markers=True)
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
        
        #Evolution menuelle de la Marge
      
        fig2 = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='MARGE', title='Evolution mensuelle de la Marge', markers=True)
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

    # Afficher les en colonne
    # Afficher le graphique dans l'interface Streamlit
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1)
            st.write('Les CA moyens les plus élevées sont observées en ', ', '.join(top_months1['Month']))
        with col2:
            st.plotly_chart(fig2)
            st.write('Les marges moyennes les plus élevées sont enrégistrées en ', ', '.join(top_months2['Month']))

        
#Visualiser les volumes par site et par Mois
#Renommer les Entités/Sites   
        Evol_df["ENTITE"] = Evol_df["ENTITE"].replace(["RDT13"], value = "MARSEILLE")
        Evol_df["ENTITE"] = Evol_df["ENTITE"].replace(["RDT45"], value = "MONTOIR")
        Evol_df["ENTITE"] = Evol_df["ENTITE"].replace(["RDT59"], value = "DUNKERQUE")
        Evol_df["ENTITE"]= Evol_df["ENTITE"].replace(["RDT76"], value = "ROUEN")
        Evol_df["ENTITE"] = Evol_df["ENTITE"].replace(["RDT76LEH"], value = "LE HAVRE")

#Grouper les données de volumes par Mois
      
        monthly_data_grouped = Evol_df.groupby(['ENTITE', pd.Grouper(key='DATE', freq='M')])['VOLUME'].sum().reset_index()
       
# Création du graphique à barres empilées avec des couleurs personnalisées (Volumes par mois et par Sites)
        fig3 = go.Figure()
        #colors = ['chocolate','Peru','darkorange','deepskyblue','silver','lightyellow']  # Liste de couleurs personnalisées
        Colors=px.colors.sequential.Cividis
        for i, entity in enumerate(monthly_data_grouped['ENTITE'].unique()):
              entity_data = monthly_data_grouped[monthly_data_grouped['ENTITE'] == entity]
              fig3.add_trace(go.Bar(x=entity_data['DATE'],y=entity_data['VOLUME'],name=entity,text=entity_data['DATE'].dt.strftime('%b %Y'),  # Format du texte (mois)textposition='inside',
        marker_color=Colors[i % len(Colors)]  # Choisissez une couleur de la liste en boucle
    ))
# Définir la date de début (janvier) et la date de fin (Decembre)
        start_date = pd.to_datetime('2023-01-01')
        end_date = pd.to_datetime('2023-12-31')
        fig3.update_xaxes(range=[start_date, end_date], dtick='M1',tickformat='%b %Y',tickangle=45
)
# Personnaliser la mise en page pour enlever l'axe des abscisses
        fig3.update_layout(barmode='stack',title='Volume par site et par mois',xaxis_title='DATE',yaxis_title='Volume',height=400,width=800,xaxis_showticklabels=False,
    xaxis_visible=False  # Rendre l'axe des abscisses invisible
)
#Afficher le graphique      
        st.plotly_chart(fig3)  

#Commentaires du graphique
    
        # Trier le DataFrame par volume décroissant
        # Trier le DataFrame par entité décroissant 
        top_entities = [] 
        monthly_data_groupedP = Evol_df.resample('M',on='DATE').mean()
        monthly_data_groupedP['Month'] = monthly_data_groupedP.index.strftime('%B')
        top_months = monthly_data_groupedP.sort_values(by='VOLUME', ascending=False).head(3)
        top_months['Month'] = top_months['Month'].map(mois_fr)
        
        # Filtrer les données pour le mois actuel
        for month_index, month_data in top_months.iterrows():
            month_entities_data = Evol_df[Evol_df['MOIS'] == month_index.month]
            
        # Grouper les données par entité et calculer le volume total pour chaque entité
            entities_volume = month_entities_data.groupby('ENTITE')['VOLUME'].sum()
        # Trier les entités par volume total dans l'ordre décroissant et sélectionner les trois premières entités
            top_entities_in_month = entities_volume.nlargest(3)
       # Ajouter les entités à la liste des entités ayant le plus de volume parmi les mois ayant le plus de volume
            top_entities.append(top_entities_in_month)     
        for i, (month_index, _) in enumerate(top_months.iterrows()):
            entities_series = top_entities[i]
            if isinstance(entities_series, pd.Series) and isinstance(entities_series.index, pd.Index):
                mois_francais = mois_fr[month_index.strftime('%B')]
                st.write(f"Au cours du mois de {mois_francais}, les entités ayant les volumes les plus élevés sont : {', '.join(entities_series.index.tolist())}")
            else:
                st.write(f"Erreur: Au cours du mois de {month_index.strftime('%B')} ne sont pas disponibles.")
    
# Visualisation des détails par site pour chaque indicateur (VOLUME,CA,MARGE)( Graphique de variation)
#Choix des Sites
        with st.sidebar:
            st.write("**Pour plus de détails sur les variations:**")
            st.write("**Choisissez un Site**")
            selected_entity = st.selectbox('SITE',ENTITE)
            st.write("**Choisissez un indicateur**")

# Grouper les données par site et par mois,et calculer la somme des volume pour chque mois
        #Indicateur VOLUME
        if st.sidebar.button("VOLUME"):
            st.write("**VARIATION DES INDICATEURS PAR SITE**")
            st.write('Vous avez selectionné:', selected_entity)
            filtered_data = Evol_df[Evol_df['ENTITE'] == selected_entity]
            monthly_data_grouped = filtered_data.groupby([pd.Grouper(key='DATE', freq='M')])['VOLUME'].sum().reset_index()
            monthly_data_grouped['Change'] = monthly_data_grouped['VOLUME'].diff().fillna(0)

           # Créer un waterfall pour le volume
            fig_waterfall = px.bar(monthly_data_grouped, x='DATE', y='Change', title='Variation du volume moyen en 2023', barmode='overlay', labels={'DATE': 'Date', 'Change': 'Change in Volume'},color='Change',color_continuous_scale='RdBu',color_continuous_midpoint=0)

           # Mise en forme du graphique
            fig_waterfall.update_layout(height=400, width=800)
            fig_waterfall.update_layout(width=700, height=500, bargap=0.1,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_waterfall)
           
          #Commentaires Volume
            monthly_data_grouped = filtered_data.groupby([pd.Grouper(key='DATE', freq='M')])['VOLUME'].sum().reset_index()
            monthly_data_grouped = monthly_data_grouped.resample('M',on='DATE').mean()
            monthly_data_grouped['Change'] = monthly_data_grouped['VOLUME'].diff().fillna(0)
            monthly_data_grouped=monthly_data_grouped.sort_values(by='Change', ascending=True)
            monthly_data_grouped['Month'] = monthly_data_grouped.index.strftime('%B')
            top_months = monthly_data_grouped.head(3)
            top_months['Month'] = top_months['Month'].map(mois_fr)
            st.write('Sur le site de ', ''.join(selected_entity), 'les baisses les plus importantes de volume ont lieu en ',','.join(top_months['Month']))
        
         #Indicateur MONTANT
        if st.sidebar.button("MONTANT"):
            st.write("**VARIATION DES INDICATEURS PAR SITE**")
            st.write('Vous avez selectionné:', selected_entity)
            filtered_data = Evol_df[Evol_df['ENTITE'] == selected_entity]
            monthly_data_grouped = filtered_data.groupby([pd.Grouper(key='DATE', freq='M')])['MONTANT'].sum().reset_index()
            monthly_data_grouped['Change'] = monthly_data_grouped['MONTANT'].diff().fillna(0)

         # Créer un waterfall pour le CA
            fig_waterfall = px.bar(monthly_data_grouped, x='DATE', y='Change', title='Variation du montant moyen en 2023', barmode='overlay', labels={'DATE': 'Date', 'Change': 'Change in amount'},color='Change',color_continuous_scale='RdBu',color_continuous_midpoint=0)

         # Mise en forme du graphique
            fig_waterfall.update_layout(height=400, width=800)
            fig_waterfall.update_layout(width=700, height=500, bargap=0.1,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_waterfall)
          
          #Commentaires Montant
            monthly_data_grouped = filtered_data.groupby([pd.Grouper(key='DATE', freq='M')])['MONTANT'].sum().reset_index()
            monthly_data_grouped = monthly_data_grouped.resample('M',on='DATE').mean()
            monthly_data_grouped['Change'] = monthly_data_grouped['MONTANT'].diff().fillna(0)
            monthly_data_grouped=monthly_data_grouped.sort_values(by='Change', ascending=True)
            monthly_data_grouped['Month'] = monthly_data_grouped.index.strftime('%B')
            top_months = monthly_data_grouped.head(3)
            top_months['Month'] = top_months['Month'].map(mois_fr)
            st.write('A', ''.join(selected_entity), ', le CA a fortement baissé en ',' ,'.join(top_months['Month']))
            
        #Indicateur MARGE
        if st.sidebar.button("MARGE"):
            st.write("**VARIATION DES INDICATEURS PAR SITE**")
            st.write('Vous avez selectionné:', selected_entity)
            filtered_data = Evol_df[Evol_df['ENTITE'] == selected_entity]
            monthly_data_grouped = filtered_data.groupby([pd.Grouper(key='DATE', freq='M')])['MARGE'].sum().reset_index()
            monthly_data_grouped['Change'] = monthly_data_grouped['MARGE'].diff().fillna(0)

        # Créer un waterfall pour la MARGE
            fig_waterfall = px.bar(monthly_data_grouped, x='DATE', y='Change', title='Variation de la marge moyenne en 2023', barmode='overlay', labels={'DATE': 'Date', 'Change': 'Change in amount'},color='Change',color_continuous_scale='RdBu',color_continuous_midpoint=0)

        # Mise en forme du graphique
            fig_waterfall.update_layout(height=400, width=800)
            fig_waterfall.update_layout(width=700, height=500, bargap=0.1,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_waterfall)
        #Commentaires Marge
            monthly_data_grouped = filtered_data.groupby([pd.Grouper(key='DATE', freq='M')])['MARGE'].sum().reset_index()
            monthly_data_grouped = monthly_data_grouped.resample('M',on='DATE').mean()
            monthly_data_grouped['Change'] = monthly_data_grouped['MARGE'].diff().fillna(0)
            monthly_data_grouped=monthly_data_grouped.sort_values(by='Change', ascending=True)
            monthly_data_grouped['Month'] = monthly_data_grouped.index.strftime('%B')
            top_months = monthly_data_grouped.head(3)
            top_months['Month'] = top_months['Month'].map(mois_fr)
            st.write('La marge sur le site de', ''.join(selected_entity), ', a connu des changements négatifs au cours des mois de ',', '.join(top_months['Month']))


                                                                      #---------------------------------------------------------------------------------------
                                                                      #                                  Analyse des sites
                                                                      #---------------------------------------------------------------------------------------

  
  #---------------------------------------------------------------------------------------
  #                                  Analyse global des sites
  #---------------------------------------------------------------------------------------
  # Analyse globale des sites
    elif  Analyse_Exploratoire == 'Analyse globale des sites':   
    #Préparation des données
    #Importation des données
                  Maritime_df=pd.read_excel("Maritime_data.xlsx")
                 #Imputation variables quantitatives
                  imputer = KNNImputer(n_neighbors=5)
                  Maritime_df['VOLUME'] = imputer.fit_transform(Maritime_df[['VOLUME']])

                 #Imputation variables qualitatives
                 #Inputation des valeurs manquantes de la variable PAYS_CLIENT
                  imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                  Maritime_df['PAYS_CLIENT'] = imputer.fit_transform(Maritime_df[['PAYS_CLIENT']]) 

                 #Inputation des valeurs manquantes de la variable ARMATEUR
                  imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
                  Maritime_df['ARMATEUR'] = imputer.fit_transform(Maritime_df[['ARMATEUR']])

                 #Renommer les sites
                  Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT13"], value = "MARSEILLE")
                  Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT45"], value = "MONTOIR")
                  Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT59"], value = "DUNKERQUE")
                  Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT76"], value = "ROUEN")
                  Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT76LEH"], value = "LE HAVRE")
                  st.write("**VUE GENERALE SUR L'ENSEMBLE DES SITES**")
        
          #Répartition des sites par nombre de TEU
                  Maritime_df1 =Maritime_df.groupby(['ENTITE'])['TEU'].sum().reset_index()
                  Maritime_df1=Maritime_df1.sort_values(by='ENTITE').sort_values(by='TEU', ascending=True)
                  fig1 = px.bar(Maritime_df1, x='TEU', y="ENTITE", orientation='h')
                  fig1.update_layout(title = dict(text = "Graphique du TEU par Site"))
                  fig1.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=360, height=400, xaxis=dict(title="TEU"),  # Add x-axis label
                  yaxis=dict(title="Site"),)
                  fig1.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(255,69,0)')
                  
           #Répartition des sites par CA
                  Maritime_df1 =Maritime_df.groupby(['ENTITE'])['MONTANT_VENTES'].sum().reset_index()
                  Maritime_df1=Maritime_df1.sort_values(by='ENTITE').sort_values(by='MONTANT_VENTES', ascending=True)
                  fig2 = px.bar(Maritime_df1, x='MONTANT_VENTES', y="ENTITE", orientation='h')
                  fig2.update_layout(title = dict(text = "Graphique du TEU par Site"))
                  fig2.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=450, height=400, xaxis=dict(title="CA"),  # Add x-axis label
                  yaxis=dict(title="Site"),)
                  fig2.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(0,191,255)')
                  
            #Présentation en colonne
                  col1, col2 = st.columns(2)
                  with col1:
                    st.plotly_chart(fig1)
                    Maritime_df1 =Maritime_df.groupby(['ENTITE'])['TEU'].sum().reset_index()
                    Maritime_df11=Maritime_df1.sort_values(by='ENTITE').sort_values(by='TEU', ascending=False)
                    top_site_teu =Maritime_df11.head(3)
                    st.write('Les 03 sites ayant enrégistré le plus de ventes de TEU sont', ', '.join(top_site_teu['ENTITE']))
                  with col2:
                    st.plotly_chart(fig2)
                    Maritime_df1 =Maritime_df.groupby(['ENTITE'])['MONTANT_VENTES'].sum().reset_index()
                    Maritime_df11=Maritime_df1.sort_values(by='ENTITE').sort_values(by='MONTANT_VENTES', ascending=False)
                    top_site_teu =Maritime_df11.head(3)
                    st.write('En terme de CA facturé, les sites de', ', '.join(top_site_teu['ENTITE']),',arrivent en tête de liste')
            
              #Répartition des sites par Marge
                  Maritime_df1 =Maritime_df.groupby(['ENTITE'])['MARGE'].sum().reset_index()
                  Maritime_df1=Maritime_df1.sort_values(by='ENTITE').sort_values(by='MARGE', ascending=True)
                  fig3 = px.bar(Maritime_df1, x='MARGE', y="ENTITE", orientation='h')
                  fig3.update_layout(title = dict(text = "Graphique du TEU par Site"))
                  fig3.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=350, height=400, xaxis=dict(title="MARGE"),  # Add x-axis label
                  yaxis=dict(title="Site"),)
                  fig3.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(139,0,139)')
                 
              #Repartition des Taux Moyen de Marge
                  Maritime_df2 =Maritime_df.groupby(['ENTITE'])['Taux_Marge'].mean().reset_index()

                  colors = ['deepskyblue', 'salmon','violet', 'powderblue',"firebrick", "mediumslateblue"]
                  explode = [0.1, 0]
                  fig4 = go.Figure()

              # Creer le graphique
                  fig4.add_trace(go.Pie(labels=Maritime_df2["ENTITE"], values=Maritime_df2["Taux_Marge"],
                     marker=dict(colors=colors, line=dict(color='white', width=0)),
                     textinfo='percent+label', hole=0.3, sort=False,
                     pull=explode, textfont_size=12))  # Decrease the font size to 12

                # Mise en forme du graphique
                  fig4.update_layout(title=dict(text=""),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,1,1,0)',
                  showlegend=False,  # Optional: Remove the legend
                  width=400, height=400,
                  xaxis=dict(showline=False, showgrid=False), # Remove x-axis line and grid
                  yaxis=dict(showline=False, showgrid=False), # Remove y-axis line and grid
                  annotations=[dict(text='taux', x=0.50, y=0.50, font_size=20, showarrow=False)] )

                #Présentation en colonne
                  col3, col4 = st.columns(2)
                  with col3:
                    st.plotly_chart(fig3)
                    Maritime_df1 =Maritime_df.groupby(['ENTITE'])['MARGE'].sum().reset_index()
                    Maritime_df11=Maritime_df1.sort_values(by='ENTITE').sort_values(by='MARGE', ascending=False)
                    top_site_teu =Maritime_df11.head(3)
                    st.write('Les sites de', ', '.join(top_site_teu['ENTITE']),',ont réalisé les marges les plus importantes au cours de cette année')
                    
                  with col4:
                    st.plotly_chart(fig4)
                    Maritime_df2 =Maritime_df.groupby(['ENTITE'])['Taux_Marge'].mean().reset_index()
                    Maritime_df_tm=Maritime_df2.sort_values(by='ENTITE').sort_values(by='Taux_Marge', ascending=False)
                    top_site_teu = Maritime_df_tm.head(3)
                    st.write('Les sites de', ', '.join(top_site_teu['ENTITE']),',ont réalisé les taux marges les plus élevés au cours de cette année')           

  
  #---------------------------------------------------------------------------------------
  #                                  Analyse par sites
  #---------------------------------------------------------------------------------------
                 
    #Voir les détails par site
    elif  Analyse_Exploratoire == 'Analyse par sites':
        #Préparation des données
        Maritime_df=pd.read_excel("Maritime_data.xlsx")
        #Imputation variables quantitatives
        imputer = KNNImputer(n_neighbors=5)
        Maritime_df['VOLUME'] = imputer.fit_transform(Maritime_df[['VOLUME']])
        
        #Imputation variables qualitatives
        #Inputation des valeurs manquantes de la variable PAYS_CLIENT
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        Maritime_df['PAYS_CLIENT'] = imputer.fit_transform(Maritime_df[['PAYS_CLIENT']]) 

        #Inputation des valeurs manquantes de la variable ARMATEUR
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        Maritime_df['ARMATEUR'] = imputer.fit_transform(Maritime_df[['ARMATEUR']])

        #Renommer les sites
        Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT13"], value = "MARSEILLE")
        Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT45"], value = "MONTOIR")
        Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT59"], value = "DUNKERQUE")
        Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT76"], value = "ROUEN")
        Maritime_df["ENTITE"] = Maritime_df["ENTITE"].replace(["RDT76LEH"], value = "LE HAVRE")
      
        #Site de Marseille
        st.sidebar.write("**Choisissez un site pour découvrir les statistiques correspondantes**")
        
        if st.sidebar.button("MARSEILLE"):
            st.write("**Les statistiques du site de Marseille**")
            filtered_data = Maritime_df[Maritime_df['ENTITE'] == "MARSEILLE"]
                      
        #Secteur d'activité
            sector_counts = filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].value_counts()
            top_sectors = sector_counts[sector_counts >10].index
            filtered_data_top = filtered_data[filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#e14500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap1= px.treemap(filtered_data_top,path=["SECTEUR_ACTIVITE_PRINCIPAL"],title="",color_discrete_sequence=color)
            treemap1=treemap1.update_layout( width=400, height=450)
                      
        #Pays du client
            d_pays=pd.DataFrame(filtered_data["PAYS_CLIENT"].value_counts()).sort_values(by='PAYS_CLIENT', ascending=False)
            d_pays=d_pays.head(3)
            pays_bar = px.bar(d_pays, x='PAYS_CLIENT', y=d_pays.index, orientation='h')
            pays_bar.update_layout(title = dict(text = "Graphique du pourcentage par site"))
            pays_bar.update_layout(title=' PAYS_CLIENT et ACTIVITE PRINCIPALE', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=350, height=400, xaxis=dict(title="count"),  # Add x-axis label
            yaxis=dict(title=""),)
            pays_bar.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(225,69,0)')
          
        #Présentation en colonne
            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(pays_bar)
            with col6:
                st.plotly_chart(treemap1)
          #Armateur
            sector_counts = filtered_data['ARMATEUR'].value_counts()
            top_sectors = sector_counts[sector_counts > 10].index
            filtered_data_top = filtered_data[filtered_data['ARMATEUR'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#ff9896", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap2= px.treemap(filtered_data_top,path=["ARMATEUR"],title="",color_discrete_sequence=color)
            treemap2=treemap2.update_layout( width=400, height=450)

        #sens
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            fig_sens = go.Figure()
            sens = pd.DataFrame(filtered_data["SENS"].value_counts())
            fig_sens.add_trace(go.Pie(labels=sens.index, values=sens['SENS'],
            marker=dict(colors=colors, line=dict(color='white', width=0)),
            textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig_sens=fig_sens.update_layout( width=320, height=450,title = dict(text = "SENS et ARMATEUR "),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='sens', x=0.52, y=0.5, font_size=20, showarrow=False)])
            

        #Présentation en colonne
            col7, col8 = st.columns(2)
            with col7:
                st.plotly_chart(fig_sens)
            with col8:
                st.plotly_chart(treemap2)
        #Cloroplètre
                      #Depart
            palette_couleur = ["#FFD700", "#800080", "#FF0000"] 
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_DEPART_LO', 'code_iso_d'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_1= px.choropleth(somme_marges_par_pays, locations="code_iso_d", hover_name="PAYS_DEPART_LO", color="MARGE",projection="natural earth",hover_data=somme_marges_par_pays.columns,color_continuous_scale=palette_couleur)
            Chloroplètre_1=Chloroplètre_1.update_layout( width=800, height=800,title = dict(text = "Pays de Départ" ))
            st.plotly_chart(Chloroplètre_1)
                      #Arrivée
            palette_couleur = ["#8B4513", "#87CEEB", "#FF6347"]
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_ARRIVEE_Lo', 'code_iso_a'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_2= px.choropleth(somme_marges_par_pays, locations="code_iso_a", hover_name="PAYS_ARRIVEE_Lo", color="MARGE",hover_data=somme_marges_par_pays.columns,projection="natural earth",color_continuous_scale=palette_couleur)
            Chloroplètre_2=Chloroplètre_2.update_layout( width=800, height=800,title = dict(text = "Pays Arrivée"))
            st.plotly_chart(Chloroplètre_2)
          
        #Les TEU
                       # Update layout and appearance of the plot
                       #Type TEU
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            d_2 = pd.DataFrame(filtered_data["TYPE_DOSSIER"].value_counts())
            fig7 = go.Figure()
            fig7.add_trace(go.Pie(labels=d_2.index, values=d_2['TYPE_DOSSIER'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig7=fig7.update_layout(title=dict(text=" Type de Dossier, Nombre et Taille du TEU"),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=310, height=400,
            xaxis=dict(showline=False, showgrid=False),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='type', x=0.50, y=0.45, font_size=20, showarrow=False)] )
            colors = ['deepskyblue', 'salmon','khaki']
            explode = [0.1, 0]
          
            #Nombre de teu  et taille
            fig8 = go.Figure()
            d_3 = pd.DataFrame(filtered_data["TEU_2"].value_counts())
            d_4 = pd.DataFrame(filtered_data["TAILLE_TC"].value_counts())

            #nb_teu
            fig8=go.Figure()
            fig8.add_trace(go.Pie(labels=d_3.index, values=d_3['TEU_2'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12)) 
            fig8=fig8.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=290, height=400,
            xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='nb_teu', x=0.52, y=0.5, font_size=20, showarrow=False)])
            
          #taille_tc
            fig9 = go.Figure()
            fig9.add_trace(go.Pie(labels=d_4.index, values=d_4['TAILLE_TC'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig9=fig9.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False, width=250, height=400,xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='taille_tc', x=0.52, y=0.5, font_size=20, showarrow=False)])
            col9, col10,col11= st.columns(3)
            with col9:
                st.plotly_chart(fig7)
            with col10:
                st.plotly_chart(fig8)
            with col11:
                st.plotly_chart(fig9)

    #Site de Rouen
        if st.sidebar.button("ROUEN"):
            st.write("**Les statistiques du site de Rouen**")
            filtered_data = Maritime_df[Maritime_df['ENTITE'] == "ROUEN"]
                      
        #Secteur d'activité
            sector_counts = filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].value_counts()
            top_sectors = sector_counts[sector_counts >10].index
            filtered_data_top = filtered_data[filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#e14500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap1= px.treemap(filtered_data_top,path=["SECTEUR_ACTIVITE_PRINCIPAL"],title="",color_discrete_sequence=color)
            treemap1=treemap1.update_layout( width=400, height=450)
                      
        #Pays du client
            d_pays=pd.DataFrame(filtered_data["PAYS_CLIENT"].value_counts()).sort_values(by='PAYS_CLIENT', ascending=False)
            d_pays=d_pays.head(3)
            pays_bar = px.bar(d_pays, x='PAYS_CLIENT', y=d_pays.index, orientation='h')
            pays_bar.update_layout(title = dict(text = "Graphique du pourcentage par site"))
            pays_bar.update_layout(title=' PAYS_CLIENT et ACTIVITE PRINCIPALE', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=350, height=400, xaxis=dict(title="count"),  # Add x-axis label
            yaxis=dict(title=""),)
            pays_bar.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(225,69,0)')
        
          #Présentation en colonne
            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(pays_bar)
            with col6:
                st.plotly_chart(treemap1)
          #Armateur
            sector_counts = filtered_data['ARMATEUR'].value_counts()
            top_sectors = sector_counts[sector_counts > 10].index
            filtered_data_top = filtered_data[filtered_data['ARMATEUR'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#ff9896", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap2= px.treemap(filtered_data_top,path=["ARMATEUR"],title="",color_discrete_sequence=color)
            treemap2=treemap2.update_layout( width=400, height=450)

          #sens
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            fig_sens = go.Figure()
            sens = pd.DataFrame(filtered_data["SENS"].value_counts())
            fig_sens.add_trace(go.Pie(labels=sens.index, values=sens['SENS'],
            marker=dict(colors=colors, line=dict(color='white', width=0)),
            textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig_sens=fig_sens.update_layout( width=320, height=450,title = dict(text = "SENS et ARMATEUR "),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='sens', x=0.52, y=0.5, font_size=20, showarrow=False)])

          #Présentation en colonne
            col7, col8 = st.columns(2)
            with col7:
                st.plotly_chart(fig_sens)
            with col8:
                st.plotly_chart(treemap2)
            #Cloroplètre
                      #Depart
            palette_couleur = ["#FFD700", "#800080", "#FF0000"] 
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_DEPART_LO', 'code_iso_d'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_1= px.choropleth(somme_marges_par_pays, locations="code_iso_d", hover_name="PAYS_DEPART_LO", color="MARGE",projection="natural earth",hover_data=somme_marges_par_pays.columns,color_continuous_scale=palette_couleur)
            Chloroplètre_1=Chloroplètre_1.update_layout( width=800, height=800,title = dict(text = "Pays de Départ" ))
            st.plotly_chart(Chloroplètre_1)
                      #Arrivée
            palette_couleur = ["#8B4513", "#87CEEB", "#FF6347"]
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_ARRIVEE_Lo', 'code_iso_a'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_2= px.choropleth(somme_marges_par_pays, locations="code_iso_a", hover_name="PAYS_ARRIVEE_Lo", color="MARGE",hover_data=somme_marges_par_pays.columns,projection="natural earth",color_continuous_scale=palette_couleur)
            Chloroplètre_2=Chloroplètre_2.update_layout( width=800, height=800,title = dict(text = "Pays Arrivée"))
            st.plotly_chart(Chloroplètre_2)
            
          #Les TEU
                       # Update layout and appearance of the plot
                       #Type TEU
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            d_2 = pd.DataFrame(filtered_data["TYPE_DOSSIER"].value_counts())
            fig7 = go.Figure()
            fig7.add_trace(go.Pie(labels=d_2.index, values=d_2['TYPE_DOSSIER'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig7=fig7.update_layout(title=dict(text=" Type de Dossier, Nombre et Taille du TEU"),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=310, height=400,
            xaxis=dict(showline=False, showgrid=False),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='type', x=0.50, y=0.45, font_size=20, showarrow=False)] )
            colors = ['deepskyblue', 'salmon','khaki']
            explode = [0.1, 0]
          
            #Nombre de teu  et taille
            fig8 = go.Figure()
            d_3 = pd.DataFrame(filtered_data["TEU_2"].value_counts())
            d_4 = pd.DataFrame(filtered_data["TAILLE_TC"].value_counts())

            #nb_teu
            fig8=go.Figure()
            fig8.add_trace(go.Pie(labels=d_3.index, values=d_3['TEU_2'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12)) 
            fig8=fig8.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=290, height=400,
            xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='nb_teu', x=0.52, y=0.5, font_size=20, showarrow=False)])
            
            #taille_tc
            fig9 = go.Figure()
            fig9.add_trace(go.Pie(labels=d_4.index, values=d_4['TAILLE_TC'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig9=fig9.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False, width=250, height=400,xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='taille_tc', x=0.52, y=0.5, font_size=20, showarrow=False)])
            col9, col10,col11= st.columns(3)
            with col9:
                st.plotly_chart(fig7)
            with col10:
                st.plotly_chart(fig8)
            with col11:
                st.plotly_chart(fig9)
           
     #Site de Montoir
        if st.sidebar.button("MONTOIR"):
            st.write("**Les statistiques du site de Montoir**")
            filtered_data = Maritime_df[Maritime_df['ENTITE'] == "MONTOIR"]
                      
        #Secteur d'activité
            sector_counts = filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].value_counts()
            top_sectors = sector_counts[sector_counts >10].index
            filtered_data_top = filtered_data[filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#e14500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap1= px.treemap(filtered_data_top,path=["SECTEUR_ACTIVITE_PRINCIPAL"],title="",color_discrete_sequence=color)
            treemap1=treemap1.update_layout( width=400, height=450)
                      
        #Pays du client
            d_pays=pd.DataFrame(filtered_data["PAYS_CLIENT"].value_counts()).sort_values(by='PAYS_CLIENT', ascending=False)
            d_pays=d_pays.head(3)
            pays_bar = px.bar(d_pays, x='PAYS_CLIENT', y=d_pays.index, orientation='h')
            pays_bar.update_layout(title = dict(text = "Graphique du pourcentage par site"))
            pays_bar.update_layout(title=' PAYS_CLIENT et ACTIVITE PRINCIPALE', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=350, height=400, xaxis=dict(title="count"),  # Add x-axis label
            yaxis=dict(title=""),)
            pays_bar.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(225,69,0)')
       
        #Présentation en colonne
            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(pays_bar)
            with col6:
                st.plotly_chart(treemap1)
        #Armateur
            sector_counts = filtered_data['ARMATEUR'].value_counts()
            top_sectors = sector_counts[sector_counts > 10].index
            filtered_data_top = filtered_data[filtered_data['ARMATEUR'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#ff9896", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap2= px.treemap(filtered_data_top,path=["ARMATEUR"],title="",color_discrete_sequence=color)
            treemap2=treemap2.update_layout( width=400, height=450)

        #sens
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            fig_sens = go.Figure()
            sens = pd.DataFrame(filtered_data["SENS"].value_counts())
            fig_sens.add_trace(go.Pie(labels=sens.index, values=sens['SENS'],
            marker=dict(colors=colors, line=dict(color='white', width=0)),
            textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig_sens=fig_sens.update_layout( width=320, height=450,title = dict(text = "SENS et ARMATEUR "),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='sens', x=0.52, y=0.5, font_size=20, showarrow=False)])

        #Présentation en colonne
            col7, col8 = st.columns(2)
            with col7:
                st.plotly_chart(fig_sens)
            with col8:
                st.plotly_chart(treemap2)
            #Cloroplètre
                      #Depart
            palette_couleur = ["#FFD700", "#800080", "#FF0000"] 
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_DEPART_LO', 'code_iso_d'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_1= px.choropleth(somme_marges_par_pays, locations="code_iso_d", hover_name="PAYS_DEPART_LO", color="MARGE",projection="natural earth",hover_data=somme_marges_par_pays.columns,color_continuous_scale=palette_couleur)
            Chloroplètre_1=Chloroplètre_1.update_layout( width=800, height=800,title = dict(text = "Pays de Départ" ))
            st.plotly_chart(Chloroplètre_1)
                      #Arrivée
            palette_couleur = ["#8B4513", "#87CEEB", "#FF6347"]
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_ARRIVEE_Lo', 'code_iso_a'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_2= px.choropleth(somme_marges_par_pays, locations="code_iso_a", hover_name="PAYS_ARRIVEE_Lo", color="MARGE",hover_data=somme_marges_par_pays.columns,projection="natural earth",color_continuous_scale=palette_couleur)
            Chloroplètre_2=Chloroplètre_2.update_layout( width=800, height=800,title = dict(text = "Pays Arrivée"))
            st.plotly_chart(Chloroplètre_2)
            #Les TEU
                       # Update layout and appearance of the plot
                       #Type TEU
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            d_2 = pd.DataFrame(filtered_data["TYPE_DOSSIER"].value_counts())
            fig7 = go.Figure()
            fig7.add_trace(go.Pie(labels=d_2.index, values=d_2['TYPE_DOSSIER'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig7=fig7.update_layout(title=dict(text=" Type de Dossier, Nombre et Taille du TEU"),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=310, height=400,
            xaxis=dict(showline=False, showgrid=False),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='type', x=0.50, y=0.45, font_size=20, showarrow=False)] )
            colors = ['deepskyblue', 'salmon','khaki']
            explode = [0.1, 0]
          
            #Nombre de teu  et taille
            fig8 = go.Figure()
            d_3 = pd.DataFrame(filtered_data["TEU_2"].value_counts())
            d_4 = pd.DataFrame(filtered_data["TAILLE_TC"].value_counts())

            #nb_teu
            fig8=go.Figure()
            fig8.add_trace(go.Pie(labels=d_3.index, values=d_3['TEU_2'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12)) 
            fig8=fig8.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=290, height=400,
            xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='nb_teu', x=0.52, y=0.5, font_size=20, showarrow=False)])
          
            #taille_tc
            fig9 = go.Figure()
            fig9.add_trace(go.Pie(labels=d_4.index, values=d_4['TAILLE_TC'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig9=fig9.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False, width=250, height=400,xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='taille_tc', x=0.52, y=0.5, font_size=20, showarrow=False)])
            col9, col10,col11= st.columns(3)
            with col9:
                st.plotly_chart(fig7)
            with col10:
                st.plotly_chart(fig8)
            with col11:
                st.plotly_chart(fig9)
     #Site de Dunkerque
        if st.sidebar.button("DUNKERQUE"):
            st.write("**Les statistiques du site de Dunkerque**")
            filtered_data = Maritime_df[Maritime_df['ENTITE'] == "DUNKERQUE"]
                      
        #Secteur d'activité
            sector_counts = filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].value_counts()
            top_sectors = sector_counts[sector_counts >10].index
            filtered_data_top = filtered_data[filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#e14500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap1= px.treemap(filtered_data_top,path=["SECTEUR_ACTIVITE_PRINCIPAL"],title="",color_discrete_sequence=color)
            treemap1=treemap1.update_layout( width=400, height=450)
                      
        #Pays du client
            d_pays=pd.DataFrame(filtered_data["PAYS_CLIENT"].value_counts()).sort_values(by='PAYS_CLIENT', ascending=False)
            d_pays=d_pays.head(3)
            pays_bar = px.bar(d_pays, x='PAYS_CLIENT', y=d_pays.index, orientation='h')
            pays_bar.update_layout(title = dict(text = "Graphique du pourcentage par site"))
            pays_bar.update_layout(title=' PAYS_CLIENT et ACTIVITE PRINCIPALE', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=350, height=400, xaxis=dict(title="count"),  # Add x-axis label
            yaxis=dict(title=""),)
            pays_bar.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(225,69,0)')
        
          #Présentation en colonne
            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(pays_bar)
            with col6:
                st.plotly_chart(treemap1)
          #Armateur
            sector_counts = filtered_data['ARMATEUR'].value_counts()
            top_sectors = sector_counts[sector_counts > 10].index
            filtered_data_top = filtered_data[filtered_data['ARMATEUR'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#ff9896", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap2= px.treemap(filtered_data_top,path=["ARMATEUR"],title="",color_discrete_sequence=color)
            treemap2=treemap2.update_layout( width=400, height=450)

          #sens
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            fig_sens = go.Figure()
            sens = pd.DataFrame(filtered_data["SENS"].value_counts())
            fig_sens.add_trace(go.Pie(labels=sens.index, values=sens['SENS'],
            marker=dict(colors=colors, line=dict(color='white', width=0)),
            textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig_sens=fig_sens.update_layout( width=320, height=450,title = dict(text = "SENS et ARMATEUR "),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='sens', x=0.52, y=0.5, font_size=20, showarrow=False)])

          #Présentation en colonne
            col7, col8 = st.columns(2)
            with col7:
                st.plotly_chart(fig_sens)
            with col8:
                st.plotly_chart(treemap2)
            #Cloroplètre
                      #Depart
            palette_couleur = ["#FFD700", "#800080", "#FF0000"] 
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_DEPART_LO', 'code_iso_d'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_1= px.choropleth(somme_marges_par_pays, locations="code_iso_d", hover_name="PAYS_DEPART_LO", color="MARGE",projection="natural earth",hover_data=somme_marges_par_pays.columns,color_continuous_scale=palette_couleur)
            Chloroplètre_1=Chloroplètre_1.update_layout( width=800, height=800,title = dict(text = "Pays de Départ" ))
            st.plotly_chart(Chloroplètre_1)
                      #Arrivée
            palette_couleur = ["#8B4513", "#87CEEB", "#FF6347"]
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_ARRIVEE_Lo', 'code_iso_a'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_2= px.choropleth(somme_marges_par_pays, locations="code_iso_a", hover_name="PAYS_ARRIVEE_Lo", color="MARGE",hover_data=somme_marges_par_pays.columns,projection="natural earth",color_continuous_scale=palette_couleur)
            Chloroplètre_2=Chloroplètre_2.update_layout( width=800, height=800,title = dict(text = "Pays Arrivée"))
            st.plotly_chart(Chloroplètre_2)
          
            #Les TEU
                       # Update layout and appearance of the plot
                       #Type TEU
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            d_2 = pd.DataFrame(filtered_data["TYPE_DOSSIER"].value_counts())
            fig7 = go.Figure()
            fig7.add_trace(go.Pie(labels=d_2.index, values=d_2['TYPE_DOSSIER'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig7=fig7.update_layout(title=dict(text=" Type de Dossier, Nombre et Taille du TEU"),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=310, height=400,
            xaxis=dict(showline=False, showgrid=False),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='type', x=0.50, y=0.45, font_size=20, showarrow=False)] )
            colors = ['deepskyblue', 'salmon','khaki']
            explode = [0.1, 0]
          #Nombre de teu  et taille
            fig8 = go.Figure()
            d_3 = pd.DataFrame(filtered_data["TEU_2"].value_counts())
            d_4 = pd.DataFrame(filtered_data["TAILLE_TC"].value_counts())

          #nb_teu
            fig8=go.Figure()
            fig8.add_trace(go.Pie(labels=d_3.index, values=d_3['TEU_2'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12)) 
            fig8=fig8.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=290, height=400,
            xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='nb_teu', x=0.52, y=0.5, font_size=20, showarrow=False)])
          
          #taille_tc
            fig9 = go.Figure()
            fig9.add_trace(go.Pie(labels=d_4.index, values=d_4['TAILLE_TC'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig9=fig9.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False, width=250, height=400,xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='taille_tc', x=0.52, y=0.5, font_size=20, showarrow=False)])
            col9, col10,col11= st.columns(3)
            with col9:
                st.plotly_chart(fig7)
            with col10:
                st.plotly_chart(fig8)
            with col11:
                st.plotly_chart(fig9)
   
        #Site du Havre
        if st.sidebar.button("LE HAVRE"):
            st.write("**Les statistiques du site Le Havre**")
            filtered_data = Maritime_df[Maritime_df['ENTITE'] == "LE HAVRE"]
                      
        #Secteur d'activité
            sector_counts = filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].value_counts()
            top_sectors = sector_counts[sector_counts >10].index
            filtered_data_top = filtered_data[filtered_data['SECTEUR_ACTIVITE_PRINCIPAL'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#e14500", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap1= px.treemap(filtered_data_top,path=["SECTEUR_ACTIVITE_PRINCIPAL"],title="",color_discrete_sequence=color)
            treemap1=treemap1.update_layout( width=400, height=450)
                      
        #Pays du client
            d_pays=pd.DataFrame(filtered_data["PAYS_CLIENT"].value_counts()).sort_values(by='PAYS_CLIENT', ascending=False)
            d_pays=d_pays.head(3)
            pays_bar = px.bar(d_pays, x='PAYS_CLIENT', y=d_pays.index, orientation='h')
            pays_bar.update_layout(title = dict(text = "Graphique du pourcentage par site"))
            pays_bar.update_layout(title=' PAYS_CLIENT et ACTIVITE PRINCIPALE', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=350, height=400, xaxis=dict(title="count"),  # Add x-axis label
            yaxis=dict(title=""),)
            pays_bar.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(225,69,0)')
        #Présentation en colonne
            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(pays_bar)
            with col6:
                st.plotly_chart(treemap1)
                 #Armateur
            sector_counts = filtered_data['ARMATEUR'].value_counts()
            top_sectors = sector_counts[sector_counts > 10].index
            filtered_data_top = filtered_data[filtered_data['ARMATEUR'].isin(top_sectors)]
            color=custom_color_scale_without_green = ["#00bfff", "#1f77b4", "#ff9896", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",  "#c5b0d5", "#17becf","#aec7e8"]
            treemap2= px.treemap(filtered_data_top,path=["ARMATEUR"],title="",color_discrete_sequence=color)
            treemap2=treemap2.update_layout( width=400, height=450)

        #sens
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            fig_sens = go.Figure()
            sens = pd.DataFrame(filtered_data["SENS"].value_counts())
            fig_sens.add_trace(go.Pie(labels=sens.index, values=sens['SENS'],
            marker=dict(colors=colors, line=dict(color='white', width=0)),
            textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig_sens=fig_sens.update_layout( width=320, height=450,title = dict(text = "SENS et ARMATEUR "),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='sens', x=0.52, y=0.5, font_size=20, showarrow=False)])

        #Présentation en colonne
            col7, col8 = st.columns(2)
            with col7:
                st.plotly_chart(fig_sens)
            with col8:
                st.plotly_chart(treemap2)
            #Cloroplètre
                      #Depart
            palette_couleur = ["#FFD700", "#800080", "#FF0000"] 
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_DEPART_LO', 'code_iso_d'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_1= px.choropleth(somme_marges_par_pays, locations="code_iso_d", hover_name="PAYS_DEPART_LO", color="MARGE",projection="natural earth",hover_data=somme_marges_par_pays.columns,color_continuous_scale=palette_couleur)
            Chloroplètre_1=Chloroplètre_1.update_layout( width=800, height=800,title = dict(text = "Pays de Départ" ))
            st.plotly_chart(Chloroplètre_1)
                      #Arrivée
            palette_couleur = ["#8B4513", "#87CEEB", "#FF6347"]
            somme_marges_par_pays = filtered_data.groupby(['ENTITE','PAYS_ARRIVEE_Lo', 'code_iso_a'])['MARGE','MONTANT_VENTES','VOLUME','TEU'].sum().reset_index()
            Chloroplètre_2= px.choropleth(somme_marges_par_pays, locations="code_iso_a", hover_name="PAYS_ARRIVEE_Lo", color="MARGE",hover_data=somme_marges_par_pays.columns,projection="natural earth",color_continuous_scale=palette_couleur)
            Chloroplètre_2=Chloroplètre_2.update_layout( width=800, height=800,title = dict(text = "Pays Arrivée"))
            st.plotly_chart(Chloroplètre_2)
          
            #Les TEU
                       # Update layout and appearance of the plot
                       #Type TEU
            colors = ['deepskyblue', 'salmon']
            explode = [0.1, 0]
            d_2 = pd.DataFrame(filtered_data["TYPE_DOSSIER"].value_counts())
            fig7 = go.Figure()
            fig7.add_trace(go.Pie(labels=d_2.index, values=d_2['TYPE_DOSSIER'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig7=fig7.update_layout(title=dict(text=" Type de Dossier, Nombre et Taille du TEU"),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=310, height=400,
            xaxis=dict(showline=False, showgrid=False),yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='type', x=0.50, y=0.45, font_size=20, showarrow=False)] )
            colors = ['deepskyblue', 'salmon','khaki']
            explode = [0.1, 0]
          
            #Nombre de teu  et taille
            fig8 = go.Figure()
            d_3 = pd.DataFrame(filtered_data["TEU_2"].value_counts())
            d_4 = pd.DataFrame(filtered_data["TAILLE_TC"].value_counts())

            #nb_teu
            fig8=go.Figure()
            fig8.add_trace(go.Pie(labels=d_3.index, values=d_3['TEU_2'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,pull=explode, textfont_size=12)) 
            fig8=fig8.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False,width=290, height=400,
            xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='nb_teu', x=0.52, y=0.5, font_size=20, showarrow=False)])
          
            #taille_tc
            fig9 = go.Figure()
            fig9.add_trace(go.Pie(labels=d_4.index, values=d_4['TAILLE_TC'],marker=dict(colors=colors, line=dict(color='white', width=0)),textinfo='percent+label', hole=0.3, sort=False,
            pull=explode, textfont_size=12))  # Decrease the font size to 12
            fig9=fig9.update_layout(title=dict(text=""),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,1,1,0)',showlegend=False, width=250, height=400,xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
            yaxis=dict(showline=False, showgrid=False),annotations=[dict(text='taille_tc', x=0.52, y=0.5, font_size=20, showarrow=False)])
            col9, col10,col11= st.columns(3)
            with col9:
                st.plotly_chart(fig7)
            with col10:
                st.plotly_chart(fig8)
            with col11:
                st.plotly_chart(fig9)
def page_ML():
    st.title("")
    col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
    with col1:
        image = Image.open('logo_rdt.jpg')
        st.image(image)
# Affichage du titre dans la deuxième colonne
    with col2:
        st.title('MODELS DE PREDICTION')
        #st.subheader(" SEGMENTATION ET PREDICTION")        

st.sidebar.image("logo_rdt.jpg", width=250)
# Créez une barre latérale pour la navigation entre les pages
st.sidebar.subheader("NAVIGUER POUR DECOUVRIR 📋🔍")
page = st.sidebar.radio("Selectionnez la page qui vous intéresse", ["ACCUEIL 📝","Analyse Exploratoire📊", "Machine Learning 📈📉"])
# Affichage conditionnel en fonction de la page sélectionnée
          
if page == "ACCUEIL 📝":
    ACCUEIL()
elif page == "Analyse Exploratoire📊":
    page_dashboard()
elif page == "Machine Learning 📈📉":
    page_ML()


