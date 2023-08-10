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

Analyses=('Analyse_univariée','Analyse_bivariée','Analyse_mensuelle')
Entité=("Marseille","Montoir","Dunkerque","Rouen")
Approches=("Clustering RFM", "Logit Binaire")


def page_dashboard():
    st.title("")
    # Ajoutez le contenu de la page de tableau de bord ici
    #Import packages

#---------------------------------------------------------------------------------------
#                                  Sidebar configuration 
#---------------------------------------------------------------------------------------
  
# Chemin vers l'image de logo
# Création d'une mise en page en colonnes avec Streamlit
    col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
    with col1:
        image = Image.open('logo_rdt.jpg')
        st.image(image)
# Affichage du titre dans la deuxième colonne
    with col2:
        st.title('Attrition clients à la RDT')
        st.subheader("Attrition au trimestre 1 de 2023")
#Import data
    attrition_df=pd.read_excel("attrition_df.xlsx")
    attrition_df.info()
    attrition_long=pd.read_excel("Base _Aternance_evolut.xlsx")

#sidebar configuration
    with st.sidebar:
        Analyse_Exploratoire=st.selectbox('Exploration des données', Analyses)
        st.write(' Analyse_Exploratoire: ', Analyse_Exploratoire)
#---------------------------------------------------------------------------------------
#                                     Plotly graph Exploration des données
#---------------------------------------------------------------------------------------

    if  Analyse_Exploratoire == 'Analyse_univariée':
        d = pd.DataFrame(attrition_df["churn"].value_counts())
        fig1 = px.pie(d, values = "churn", names = ["Non", "Oui"], hole = 0.5, opacity = 0.8,
            labels = {"label" :"Potability","Potability":"Number of Samples"})
        fig1.update_layout(title = dict(text = "Attrition et activité principale"), width=360, height=400,annotations=[dict(text='attrition', x=0.50, y=0.5, font_size=20, showarrow=False)])
        fig1.update_traces(textposition = "outside", textinfo = "percent+label")
        fig1.update_layout(showlegend=False)
        plt.figure(figsize=(6, 6))
        sns.set_theme(style='white')
        sns.set(font_scale=2)
        ax2= px.treemap(attrition_df,path=["activite_prin"],title="")
        fig5=ax2.update_layout( width=400, height=410)
        col1, col2 = st.columns(2)
        with col1:
             st.plotly_chart(fig1)
        with col2:
            st.plotly_chart(fig5)
        d_site = pd.DataFrame(attrition_df["Site"].value_counts()).sort_values(by='Site', ascending=True)
        fig2 = px.bar(d_site, x='Site', y=d_site.index, orientation='h')
        fig2.update_layout(title = dict(text = "Graphique du pourcentage par site"))
        fig2.update_layout(title='Sites, Pays et Zone', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=250, height=350, xaxis=dict(title="count"),  # Add x-axis label
                  yaxis=dict(title="Site"),)
        fig2.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(92, 180, 174)')
        d_pays = pd.DataFrame(attrition_df["pays_cl"].value_counts()).sort_values(by='pays_cl', ascending=True)
        fig3 = px.bar(d_pays, x='pays_cl', y=d_pays.index, orientation='h')
        fig3.update_layout(title = dict(text = "Graphique du pourcentage par site"))
        fig3.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=250, height=350, xaxis=dict(title="count"),  # Add x-axis label
                  yaxis=dict(title="Pays_client"),)
        fig3.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(147,112,219)')
        d_zone_geo = pd.DataFrame(attrition_df["zone_geo_prin"].value_counts()).sort_values(by='zone_geo_prin', ascending=True)
        fig6 = px.bar(d_zone_geo, x='zone_geo_prin', y=d_zone_geo.index, orientation='h')
        fig6.update_layout(title = dict(text = "Graphique du pourcentage par site"))
        fig6.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,1,1,0)', width=300, height=350, xaxis=dict(title="count"),  # Add x-axis label
                  yaxis=dict(title="zone_geo"),)
        fig6.update_traces(marker_line_width=0, marker_opacity=0.7, marker_color='rgb(244,164,96)')
        col1, col2, col3 = st.columns(3)
        with col1:
             st.plotly_chart(fig2)
        with col2:
             st.plotly_chart(fig3)
        with col3:
            st.plotly_chart(fig6)
        colors = ['deepskyblue', 'salmon']
        explode = [0.1, 0]
        fig = go.Figure()
        d_2 = pd.DataFrame(attrition_df["sens"].value_counts())
        fig.add_trace(go.Pie(labels=d_2.index, values=d_2['sens'],
                     marker=dict(colors=colors, line=dict(color='white', width=0)),
                     textinfo='percent+label', hole=0.3, sort=False,
                     pull=explode, textfont_size=12))  # Decrease the font size to 12

# Update layout and appearance of the plot
        fig7=fig.update_layout(title=dict(text=" Sens, nombre et la taille du TEU"),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,1,1,0)',
                  showlegend=False,  # Optional: Remove the legend
                  width=310, height=400,
                  xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
                  yaxis=dict(showline=False, showgrid=False), # Remove y-axis line and grid
                  annotations=[dict(text='sens', x=0.50, y=0.45, font_size=20, showarrow=False)] )
 

        colors = ['deepskyblue', 'salmon','lightgreen']
        explode = [0.1, 0]
        fig8 = go.Figure()
   

        d_3 = pd.DataFrame(attrition_df["taille_tc"].value_counts())
        d_4 = pd.DataFrame(attrition_df["nb_teu"].value_counts())
# Create a pie chart
    # Decrease the font size to 12


        fig8.add_trace(go.Pie(labels=d_3.index, values=d_3['taille_tc'],
                     marker=dict(colors=colors, line=dict(color='white', width=0)),
                     textinfo='percent+label', hole=0.3, sort=False,
                     pull=explode, textfont_size=12))  # Decrease the font size to 12

# Update layout and appearance of the plot
        fig8=fig8.update_layout(title=dict(text=""),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,1,1,0)',
                  showlegend=False,  # Optional: Remove the legend
                  width=290, height=400,
                  xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
                  yaxis=dict(showline=False, showgrid=False)# Remove y-axis line and grid
                  ,annotations=[
                  dict(text='taille_tc', x=0.52, y=0.5, font_size=20, showarrow=False)])
   
        fig9 = go.Figure()
        fig9.add_trace(go.Pie(labels=d_4.index, values=d_4['nb_teu'],
                     marker=dict(colors=colors, line=dict(color='white', width=0)),
                     textinfo='percent+label', hole=0.3, sort=False,
                     pull=explode, textfont_size=12))  # Decrease the font size to 12
   
  
# Update layout and appearance of the plot
        fig9=fig9.update_layout(title=dict(text=""),
                  plot_bgcolor='rgba(0,0,0,0)',
                  paper_bgcolor='rgba(0,1,1,0)',
                  showlegend=False,  # Optional: Remove the legend
                  width=300, height=400,
                  xaxis=dict(showline=False, showgrid=False),  # Remove x-axis line and grid
                  yaxis=dict(showline=False, showgrid=False)# Remove y-axis line and grid
                  ,annotations=[
                  dict(text='nb_teu', x=0.5, y=0.5, font_size=20, showarrow=False)])
   


              
        col1, col2,col3= st.columns(3)
        with col1:
             st.plotly_chart(fig7)
        with col2:
             st.plotly_chart(fig8)
        with col3:
            st.plotly_chart(fig9)

    elif Analyse_Exploratoire== 'Analyse_bivariée':
        plt.figure(figsize=(10,8))
        fig11 = px.histogram(attrition_df, x="Site", color="churn", barmode="group", title="<b> Répartition des sites suivant Churn</b>")
        fig11.update_layout(width=400, height=400, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        color_map = {"0":'#c2c2f0', "1":  '#66b3ff'}
        plt.figure(figsize=(10,8))
        fig12 = px.histogram(attrition_df, x="taille_tc", color="churn", title="<b> Répartition des tailles de Teu suivant Churn</b>")
        fig12.update_layout(width=400, height=400, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig11)

        with col2:
            st.plotly_chart(fig12)
        fig13 = px.histogram(attrition_df, x="pays_cl", color="churn", title="<b>Répartition des pays du client selon Churn</b>")
        fig13.update_layout(width=350, height=400, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        fig14= px.histogram(attrition_df, x="activite_prin", color="churn", barmode="group", title="<b> Répartition des secteurs d'activités selon churn Churn</b>")
        fig14.update_layout(width=400, height=400, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')


        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig13)

        with col2:
            st.plotly_chart(fig14)

        attrition_df_n = attrition_df.drop(['pays_c','marge'], axis=1)
        corr_matrix = attrition_df_n.corr(method='spearman')

# Créer une heatmap Plotly pour la matrice de corrélation
        annotations = []
        for i, row in enumerate(corr_matrix.index):
            for j, value in enumerate(corr_matrix.iloc[i]):
                annotations.append(
                dict(
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                text=f'{value:.2f}',  # Formatage à deux décimales
                showarrow=False
            )
        )

# Créer la heatmap Plotly avec les valeurs directement affichées
            heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='blues',  # Choisir une échelle de couleurs
            hoverongaps=False,  # Afficher les espaces blancs pour les données manquantes
            text=corr_matrix.round(2)  # Utiliser les valeurs de la matrice comme texte
))

# Mise en forme de la heatmap
            heatmap.update_layout(
            title="Matrice de Corrélation (Spearman)",
            xaxis_title="",
            yaxis_title="",
            width=700,
            height=700,
            annotations=annotations  # Ajouter les annotations pour afficher les valeurs
) 
        st.plotly_chart(heatmap)   
# Application Streamlit


    elif Analyse_Exploratoire== 'Analyse_mensuelle':
         attrition_long['DATE_RENTA'] = pd.to_datetime(attrition_long['DATE_RENTA'])
         monthly_data_grouped = attrition_long.resample('M', on='DATE_RENTA').mean()
# Create the line plot using Plotly Express
         fig_men = px.line(monthly_data_grouped, x=monthly_data_grouped.index, y='VOLUME', title='Evolution mensuelle globale', markers=True)

         fig_men.update_layout(width=700, height=500, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
         st.plotly_chart(fig_men)

         with st.sidebar:
             selected_entity = st.selectbox('Entité', Entité)

         st.write('Entité sélectionnée:', selected_entity)
# Filter data based on the selected entity
         filtered_data = attrition_long[attrition_long['ENTITE'] == selected_entity]

# Group the data by month and site, and calculate the sum of volume for each month
         monthly_data_grouped = filtered_data.groupby([pd.Grouper(key='DATE_RENTA', freq='M')])['VOLUME'].sum().reset_index()

# Create the cascade bar chart using Plotly Express
         fig_det = px.bar(monthly_data_grouped, x='DATE_RENTA', y='VOLUME', title='Répartition mensuelle du volume moyen',
                 labels={'DATE_RENTA': 'Date', 'VOLUME': 'Volume'},
            
                 color_discrete_sequence=px.colors.qualitative.Plotly)

# Update layout and appearance of the plot
         fig_det.update_layout(height=400, width=710, barmode='relative', coloraxis_showscale=False)


# Display the Plotly Express chart using Streamlit
         st.plotly_chart(fig_det)

         monthly_data_grouped['Change'] = monthly_data_grouped['VOLUME'].diff().fillna(0)

# Create the waterfall chart using Plotly Express
         fig_waterfall = px.bar(monthly_data_grouped, x='DATE_RENTA', y='Change', title='Variation du volume moyen', barmode='overlay', labels={'DATE_RENTA': 'Date', 'Change': 'Change in Volume'},color_discrete_sequence=px.colors.qualitative.Plotly)

# Update layout and appearance of the plot
         fig_waterfall.update_layout(height=400, width=800)
         fig_waterfall.update_layout(width=700, height=500, bargap=0.1,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
         st.plotly_chart(fig_waterfall)



#---------------------------------------------------------------------------------------
#                              Clustering rfm 
#---------------------------------------------------------------------------------------

   
#---------------------------------------------------------------------------------------
#                                Logit Model
#---------------------------------------------------------------------------------------


# Création des modèles

def page_settings():
    st.title("")
    # Ajoutez le contenu de la page des paramètres ici
    with st.sidebar:
        Méthodes_ML=st.selectbox('Méthodes_Machine_learning', Approches)

    
     #attrition_rfm=attrition_df_n[['ca','recence','frequence']]
    col1, col2 = st.columns([1, 5])

# Affichage du logo dans la première colonne
    with col1:
        image = Image.open('logo_rdt.jpg')
        st.image(image)
# Affichage du titre dans la deuxième colonne
    with col2:
        st.title('Attrition clients à la RDT')
        st.subheader("Attrition au trimestre 1 de 2023")
    attrition_df=pd.read_excel("attrition_df.xlsx")
    attrition_rfm=attrition_df[['ca','recence','frequence']]
    s1=np.full((1,attrition_rfm.shape[0]-int(0.8*attrition_rfm.shape[0])),1)
    s2=np.full((1,int(0.2*attrition_rfm.shape[0])),2)
    s3=np.full((1,int(0.2*attrition_rfm.shape[0])),3)
    s4=np.full((1,int(0.2*attrition_rfm.shape[0])),4)
    s5=np.full((1,int(0.2*attrition_rfm.shape[0])),5)
    score=np.hstack((s1,s2,s3,s4,s5)).flatten()
    attrition_rfm=attrition_rfm.sort_values(by='recence',ascending=False)
    attrition_rfm['r_score']=score
    for i , j in zip (('frequence' ,'ca'),('f_score','m_score')):
        attrition_rfm= attrition_rfm.sort_values(by=i)
        attrition_rfm[j]=score

    if Méthodes_ML == "Clustering RFM":
        if st.sidebar.button("Inertie"):
            SSE=[]
            for k in range (0,10):
                kmeans=KMeans(n_clusters=k+1, random_state=128).fit(attrition_rfm.iloc[:,3:])
                SSE.append(kmeans.inertia_)
            plt.figure(figsize=(10,8))
            sns.pointplot(x=list(range(1,11)),y=SSE)
            sns.set_style("whitegrid")
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
            plt.title('Elbow Method for Optimal k', fontsize=12)
         # Sauvegarder le graphique Seaborn en tant qu'image temporaire
            plt.savefig("seaborn_plot.png")
    
    # Afficher l'image dans Streamlit
            st.image("seaborn_plot.png")

        if st.sidebar.button("Silhouette_visualizer"):
            image_silo = Image.open('silou_str.jpg')
            st.image(image_silo)
    
        st.subheader("Résultats du clustering RFM avec l'utilisation des scores")
    # Afficher l'image dans Streamlit
        model=KMeans(n_clusters=5, random_state=100).fit(attrition_rfm.iloc[:,3:])
        centers=model.cluster_centers_
        fig=plt.figure(figsize=(10,10))

        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(attrition_rfm.iloc[:,3],attrition_rfm.iloc[:,4],attrition_rfm.iloc[:,5],cmap="brg",c=model.predict(attrition_rfm.iloc[:,3:]))

        ax.set_xlabel('Recence', fontsize=12)
        ax.set_ylabel('Frequence', fontsize=12)
        ax.set_zlabel('CA', fontsize=11)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', marker='*', s=100)
# Set the limits for the x, y, and z axes to reduce the scale
        ax.set_xlim([min(attrition_rfm.iloc[:, 3]) , max(attrition_rfm.iloc[:, 3])])
        ax.set_ylim([min(attrition_rfm.iloc[:, 4]) , max(attrition_rfm.iloc[:, 4]) ])
        ax.set_zlim([min(attrition_rfm.iloc[:, 5]) , max(attrition_rfm.iloc[:, 5]) ])
# Add a colorbar to show the cluster color mapping

        num_ticks =10
        ax.xaxis.set_major_locator(MaxNLocator(num_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(num_ticks))
        ax.zaxis.set_major_locator(MaxNLocator(num_ticks))
        ax.set_facecolor('white')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)
        plt.savefig("clus_rfm_score.png")   
        st.image("clus_rfm_score.png")

        attrition_rfm['Cluster']=model.labels_
        melted_rfm_norm=pd.melt(attrition_rfm.reset_index(),
                        id_vars=['Cluster'],
                        value_vars=['r_score','f_score','m_score'],
                        var_name='Features',
                        value_name='Value')
        sns.lineplot(x='Features',y='Value', hue='Cluster', data=melted_rfm_norm)

        aggregated_data=attrition_rfm.groupby('Cluster').agg({'r_score':['mean','min','max'],
                            'f_score':['mean','min','max'],
                            'm_score':['mean','min','max','count']})
        formatted_data = aggregated_data.applymap(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)

# Afficher le tableau dans Streamlit
        st.dataframe(formatted_data)

    elif Méthodes_ML=='Logit Binaire':
        coefficients = {
    'const': -1.1431,
    'volume': -1.7502,
    'frequence': 1.7624,
    'ca':-0.0736,
    'nb_teu_6 et plus' : -1.0884,
    'taille_tc_taille_0': 1.3225,
    'activite_prin_Grande Distribution':-0.8571,
    'activite_prin_Transitaires et assimiles':-0.5986,
    'recence' : 0.4671,
    'activite_prin_Autres':0.8463,
    'nature_LCL':-1.5965,
    'taille_tc_taille_40': 0.6443,
    'nature_S':-1.9415,
    'pays_cl_RE':-0.9957,
    'zone_geo_prin_CROSS TRADE': 0.6866

}
        std_errors = {
    'const':  0.411,
    'volume': 0.853,
    'frequence': 0.641,
    'ca':0.776,
    'nb_teu_6 et plus' : 0.653,
    'taille_tc_taille_0': 0.842,
    'activite_prin_Grande Distribution': 0.667,
    'activite_prin_Transitaires et assimiles':0.395,
    'recence' : 0.416,
    'activite_prin_Autres': 0.337 ,
    'nature_LCL':0.850 ,
    'taille_tc_taille_40':0.346 ,
    'nature_S':0.828,
    'pays_cl_RE':0.564,
    'zone_geo_prin_CROSS TRADE': 0.531
}

        p_values = {
    'const': 0.005,
    'volume': 0.040,
    'frequence': 0.006,
    'ca':0.924,
    'nb_teu_6 et plus' : 0.095,
    'taille_tc_taille_0': 0.116 ,
    'activite_prin_Grande Distribution':0.199,
    'activite_prin_Transitaires et assimiles':0.130,
    'recence' : 0.262,
    'activite_prin_Autres':0.012,
    'nature_LCL': 0.060,
    'taille_tc_taille_40': 0.063,
    'nature_S':0.019,
    'pays_cl_RE':0.077,
    'zone_geo_prin_CROSS TRADE':  0.196

   
}
        
        odds_ratios = {
    'const':0.318817,
    'volume':0.173733,
    'frequence': 5.826480,
    'ca':0.929058,
    'nb_teu_6 et plus' :0.336754,
    'taille_tc_taille_0':  3.752755 ,
    'activite_prin_Grande Distribution':0.424397 ,
    'activite_prin_Transitaires et assimiles': 0.549592,
    'recence' : 1.595282,
    'activite_prin_Autres':2.331106,
    'nature_LCL':0.202604,
    'taille_tc_taille_40': 1.904692,
    'nature_S':0.143492,
    'pays_cl_RE':0.369462,
    'zone_geo_prin_CROSS TRADE': 1.987002
  
}
# Create a DataFrame to store results
        results_df = pd.DataFrame({'Coefficient': coefficients.values(),
                           'Standard Error': std_errors.values(),
                           'P-Value': p_values.values(),
                           'odds_ratios': odds_ratios.values()},
                          index=coefficients.keys())

        def highlight_cells(val):
            if isinstance(val, (int, float)):
                if abs(val) < 0.1:  # Example condition for highlighting coefficients
                    return f'background-color:lightgray'
                return ''

        styled_results_df = results_df.style.applymap(highlight_cells)

# Streamlit app
        st.title('Logistic Regression Results')

# Display the styled results table
        st.dataframe(styled_results_df)






# Créez une barre latérale pour la navigation entre les pages
page = st.sidebar.radio("Visualisation", [ "Analyse Exploratoire", "Techniques de Machine Learning"])

# Affichage conditionnel en fonction de la page sélectionnée

if page == "Analyse Exploratoire":
    page_dashboard()
elif page == "Techniques de Machine Learning":
    page_settings()



