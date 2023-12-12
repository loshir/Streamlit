import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from joblib import load
import plotly.graph_objects as go
import plotly.express as px
from streamlit_elements import elements, mui, html
from scipy.stats import pearsonr

st.set_page_config(
    page_title="Projet températures terrestre",
    layout="wide")

st.sidebar.title("Sommaire")
pages = ["Présentation projet", "Données et processing", "Dataviz", "Machine learning", "Conclusion projet"]
page = st.sidebar.radio("Navigation",pages)
with st.sidebar:
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.write("""Équipe :""")
    st.write(""" 
             \nIsmail Cissé :rain_cloud:
             \nAlexis Jover :seedling:
             \nGéraldine Sivilia :earth_asia:""")

# PAGE 1 INTRODUCTION
if page == pages[0]:
    st.title("Projet - Températures terrestre")
    st.divider()
    st.header("Objectifs")
    st.divider()
    st.write("""Les données mises à notre disposition faisant état de l'évolution des températures ainsi que des émissions de CO2 (avec plusieurs variables explicatives),
             nous avons déterminé comme objectif principal de constater, ou non, le réchauffement (ou dérèglement) climatique à différentes échelles. 
             Par la suite, après avoir établi un constat sur la situation que nous indique nos données, nous espérons réaliser un, ou des, modèle(s) de machine learning afin de prédire
             les températures ou d'autres variables cibles pertinentes.""")
    st.divider()
    st.write("Afin d'obtenir un constat solide, nous nous sommes posé les questions suivantes :")
    with st.container(border=True):
        st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Questions.png")
    
# PAGE 2 DONNEES ET PROCESSING
if page == pages[1]:
    st.title("Données et processing")
    st.divider()
    st.header("Jeux de données à disposition")
    st.write("""5 fichiers de la NASA ont été utilisés pour notre exploration :

1. Le fichier «GBL. Ts + dSST » : vision des anomalies de températures planétaire depuis 1880,
2. Le fichier « NH.Ts + dSST »: vision des anomalies de températures planétaire dans l’hémisphères
Nord depuis 1880,
3. Le fichier « SH.ts +dSST » : vision des anomalies de températures planétaire dans l’hémisphère
Sud depuis 1880,
4. Le fichier « ZonAnn.Ts » : vision des anomalies de températures planétaire depuis 1880 par
latitude et par longitude,
5. Le fichier « Owid CO2 »: évolution du CO2 annuel par habitant, gaz à effet de serre, mix
énergétique, etc.""")
    st.write("Les variables utilisées pour chaque jeu de données est affiché dans les tableaux ci-dessous :")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Capture d’écran 2023-12-09 à 23.16.23.png")
    with col2:
        with st.container(border=True):
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Capture d’écran 2023-12-09 à 23.16.35.png")
    st.divider()
    st.header("Traitement des données")
    st.write("""Le traitement des données a consisté en un processus en 7 étapes pour nettoyer et préparer les données à leur utilisation. Ces
actions ont été réalisées à l’identique pour tous les fichiers utilisés.""")
    with st.container(border=True):
        st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Capture d’écran 2023-12-09 à 23.19.42.png", caption = "Étapes du traitement des données")

# PAGE 3 DATAVIZ ------------------------------------------------------------------------------------------------------------------------------------------------------------
if page == pages[2]:
    st.title("Dataviz")
    tab1, tab2, tab3, tab4 = st.tabs(["Températures", "PIB", "CO2", "Corrélations"])
    # Températures ------------------------------------------------------------------------------------------------------------------------------------------------------------
    with tab1:
        st.divider()
        st.header("Le réchauffement climatique est-il une réalité actuelle?")
        st.divider()
        df1 = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/GLB.Ts+dSST.csv", skiprows = 1)
        df1 = df1.set_index("Year")
        df_courbes = df1[["DJF", "MAM", "JJA", "SON"]].stack().reset_index() # Préparation du dataframe pour avoir trois colonnes.
        df_courbes.columns = ["Année", "Saisons", "Températures"] # Definition des colonnes
        df_courbes["Températures"].replace(to_replace = "***", value = np.NaN, inplace = True)
        df_courbes = df_courbes.dropna()
        df_courbes = df_courbes.replace(to_replace = ["DJF", "MAM", "JJA", "SON"], value = ["Hiver", "Printemps", "Eté", "Automne"]) # Renommage
        df_courbes["Températures"] = df_courbes["Températures"].rolling(4).mean() # Afin de lisser la courbe

        df_zon = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/ZonAnn.Ts+dSST.csv", index_col = 0)

        # Plotly
        fig = px.line(df_courbes[(df_courbes.Saisons == "Eté") | (df_courbes.Saisons == "Hiver")], x='Année', y='Températures', color = "Saisons", line_shape="spline",
                      title = "Evolution des températures par saison à l'échelle mondiale",
                        labels=dict(Températures = "Anomalies de températures"),              color_discrete_map={'Printemps':'Green',
                                        'Eté':'red',
                                        'Automne':'orange',
                                        'Hiver':'blue'}, range_y = [-1, 1.5])
        

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, theme = "streamlit")
        st.write("""Nous constatons une élévation constante des températures depuis 1880 à l’échelle mondiale particulièrement marquée en Hiver et en Eté. Cette élévation semble plus prononcée à certaines dates clés comme 1920, 1941 et 1980. A partir de 1980, la hausse devient de plus en plus significative. 
                 \nCependant Pour le moment nous ne pouvons que constater une hausse des températures à l’échelle planétaire mais nous ne pouvons pas la corréler à l’activité humaine.""")
        df_plotly = df1.stack().reset_index() # Préparation DF pour avoir 3 colonnes
        df_plotly.columns = ['Year', 'Month', 'Temperature'] # renommage
        df_plotly["Temperature"].replace(to_replace = "***", value = np.NaN, inplace = True)
        df_plotly = df_plotly.dropna()
        df_plotly['Temperature'] = df_plotly['Temperature'].astype(float)


        #Plotly
        fig = px.scatter(df_plotly, x='Month', y='Temperature', color = "Year", range_y=[-1,1.5],
                        labels=dict(Temperature = "Anomalies de températures", Year = "Années", Month = "Mois"), color_continuous_scale="rainbow", title = "Cycle saisonnier depuis 1880 à l'échelle mondiale")

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.header("La hausse des températures est-elle circonscrite à une zone géographique?")
        col1, col2 = st.columns(2)
        with col1:
            df_NHGRAPH = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/df_NHGRAPH.csv")
            fig1 = px.line(df_NHGRAPH, x='Année', y='Températures', color = "Saisons", title="Evolution des températures dans l'hémisphère nord par saison", labels=dict(Températures = "Anomalies de températures"), color_discrete_map={'Eté':'red','Hiver':'blue'}, range_y = [-1, 2], line_shape="spline")
            with st.container(border=True):
                st.plotly_chart(fig1, use_container_width=True)
        with col2:
            df_HSGRAPH = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/df_HSGRAPH.csv")
            fig2 = px.line(df_HSGRAPH, x='Année', y='Températures', color = "Saisons", title="Évolution des températures dans l'hémisphère sud par saison",
                 labels=dict(Températures = "Anomalies de températures"),              color_discrete_map={
                                 'Eté':'red',
                                 'Hiver':'blue'}, range_y = [-1, 2], line_shape="spline")
            with st.container(border=True):
                st.plotly_chart(fig2, use_container_width=True)
        st.write("""Dans l’Hémisphères Nord, les températures semblent à la hausse en hiver comme en été, avec des pics d’anomalies dès 1900, puis dans les années 1920 et les année 1940. Aussi à partir des années 1980, l’élévation de la température terrestre grimpe significativement.
                \nDans l’Hémisphère Sud, le début de l’élévation de la température terrestre est décalé, plus tardif par rapport à l’Hémisphère Nord avec un pic qui s’amorce à compter de 1940. """)
        st.divider()
        df_ZonGRAPH = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/df_ZonGRAPH.csv")
        df_ZonGRAPH["Lattitudes"] = df_ZonGRAPH["Lattitudes"].replace(to_replace=["44N-64N","24N-44N","EQU-24N"], value=["Amérique du Nord, Europe, Asie (44N-64N)","Afrique du Nord, Moyen-Orient, Amérique centrale, Europe du Sud, Asie du Sud (24N-44N)","Afrique (EQU-24N)"])
        zones = list(df_ZonGRAPH.Lattitudes.unique())
        zone = st.selectbox(label = "Choix de la zone Géographique", options = zones)
        fig4 = px.line(df_ZonGRAPH[df_ZonGRAPH.Lattitudes == zone], x='Année', y='Températures',
                        labels=dict(Températures = "Anomalies de températures"), line_shape="spline")
        if zone == "Amérique du Nord, Europe, Asie (44N-64N)":
            fig4.update_layout(title_text = "Evolution des températures sur la zone Amérique du Nord, Europe, Asie (44N-64N)")
        elif zone == "Afrique du Nord, Moyen-Orient, Amérique centrale, Europe du Sud, Asie du Sud (24N-44N)":
            fig4.update_layout(title_text = "Evolution des températures sur la zone Afrique du Nord, Moyen-Orient, Amérique centrale, Europe du Sud, Asie du Sud (24N-44N)")
        else:
            fig4.update_layout(title_text = "Evolution des températures sur la zone Afrique (EQU-24N)")
        with st.container(border=True):
            st.plotly_chart(fig4, use_container_width=True)
        st.write("""Les territoires compris entre les latitudes 44N-64N affichent une tendance d’évolution des températures similaire à celle constatée dans l’Hémisphères Nord, soit des pics d’élévation dans les années 1920, les années 1940 puis à compter des années 1980. Date à laquelle la courbe grimpe à la hausse significativement. 
                \nL’élévation des températures dans l’Hémisphères Nord est liée aux zones géographiques à forte densité de population dès les années 1920. """)
    # PIB ------------------------------------------------------------------------------------------------------------------------------------------------------------
    with tab2:
        
        # PIB GLOBAL ---------------------------------------------------------------------
        st.header("Existe-t-il un lien entre la hausse des températures et le développement industriel?")
        st.write("Le PIB connaît également une hausse significative notamment à partir des années 1950. L’évolution du PIB suit la même tendance que celle de la température terrestre.")
        df_co2 = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/OWID.csv")
        gdp_global = df_co2.groupby("year")["gdp"].sum().reset_index() # Préparation DF
        fig = px.line(
            gdp_global,
            x = "year", y = "gdp",
            labels=dict(year = "Années", gdp = "PIB mesuré en dollars"), title = "Évolution du PIB à l'échelle mondiale")
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)   
        
        # PIB par continent ---------------------------------------------------------------------
        st.divider()
        st.header("Comparaison du PIB par continent")
        col1, col2 = st.columns(2)
        with col1:
            pib_continent = df_co2.groupby(["year", "continent"])["gdp"].sum().reset_index()
            choix = ["Europe", "Amérique du Nord", "Amérique du Sud", "Asie", "Afrique", "Océanie"]
            option = st.selectbox("Choix du continent A", choix, label_visibility = "hidden")
            if option == "Europe":
                fig = px.line(
                pib_continent[pib_continent.continent == "Europe"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Nord":
                fig = px.line(
                pib_continent[pib_continent.continent == "North America"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Sud":
                fig = px.line(
                pib_continent[pib_continent.continent == "South America"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Afrique":
                fig = px.line(
                pib_continent[pib_continent.continent == "Africa"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Asie":
                fig = px.line(
                pib_continent[pib_continent.continent == "Asia"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Océanie":
                fig = px.line(
                pib_continent[pib_continent.continent == "Oceania"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
        with col2:
            pib_continent = df_co2.groupby(["year", "continent"])["gdp"].sum().reset_index()
            choix2 = ["Europe", "Amérique du Nord", "Amérique du Sud", "Asie", "Afrique", "Océanie"]
            default = choix[0]
            option = st.selectbox("Choix du continent B", choix2, label_visibility = "hidden", index = 1)
            if option == "Europe":
                fig = px.line(
                pib_continent[pib_continent.continent == "Europe"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Nord":
                fig = px.line(
                pib_continent[pib_continent.continent == "North America"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Sud":
                fig = px.line(
                pib_continent[pib_continent.continent == "South America"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Afrique":
                fig = px.line(
                pib_continent[pib_continent.continent == "Africa"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Asie":
                fig = px.line(
                pib_continent[pib_continent.continent == "Asia"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Océanie":
                fig = px.line(
                pib_continent[pib_continent.continent == "Oceania"],
                x = "year", y = "gdp",
                labels=dict(year = "Années", gdp = "PIB mesuré en dollars")
                )
                fig.update_layout(yaxis_range=[0,50000000000000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
        st.write("""L’Europe représente le continent dont la courbe d’évolution du PIB est la plus significative et plus tôt, avec un pic dès 1950 correspondant à la période de reconstruction après la seconde guerre mondiale. Cette courbe d’évolution progresse de la même façon que la courbe d’évolution des températures dans l’Hémisphère Nord. L’Europe, l’Amérique du Nord et l’Asie sont les trois continents avec le PIB est le plus important et le plus croissant.""")
        # PIB PAR pays par continent ------------------------------------------------------------------
        st.divider()
        st.header("Contibution au PIB des pays dans leur continent")
        pib_continent_country = df_co2
        pib_continent_country["total_gdp_per_year"] = pib_continent_country.groupby(["year", "continent"])["gdp"].transform("sum")
        pib_continent_country["pourcent_gdp"] = (pib_continent_country.gdp / pib_continent_country.total_gdp_per_year) * 100
        pib_continent_country = pib_continent_country.dropna(axis = 0, subset="continent")
        
        continent_list = ["Europe", "Asia", "North America", "Africa", "South America", "Oceania"]
        continent = st.selectbox(label = "Choix du continent", options = continent_list)
        time_period = ["1930-1970", "1980-2000", "2005-2015"]
        period = st.selectbox(label = "Choix de la période", options = time_period)
        def label_countries(row):
            if row['pourcent_gdp'] < 5:
                return 'Other(< 5%)'
            else:
                return row['country']
        if period == time_period[0]:
            df_3070 = pib_continent_country[(pib_continent_country.year == 1930) |(pib_continent_country.year == 1950) |(pib_continent_country.year == 1970)]
            df_3070 = df_3070[df_3070.continent == continent]
            df_3070["country2"] = df_3070.apply(label_countries, axis=1)

            fig = px.pie(df_3070, values = "pourcent_gdp", names = "country2", facet_col = "year", facet_col_wrap=5, category_orders={"year" : [1930, 1950, 1970]}, width = 1500, height = 750, color_discrete_sequence=px.colors.qualitative.Pastel2)
            fig.update_traces(textposition='inside', textinfo = "percent+label", hole=.3, pull=[0.2])
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Pour n'afficher que l'année et pas "year = année"
            fig.update_layout(margin=dict(t=150))
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)    
        if period == time_period[1]:
            df_8000 = pib_continent_country[(pib_continent_country.year == 1980) |(pib_continent_country.year == 1990) |(pib_continent_country.year == 2000)]
            df_8000 = df_8000[df_8000.continent == continent]
            df_8000["country2"] = df_8000.apply(label_countries, axis=1) # Fonction label pays autre



            fig = px.pie(df_8000, values = "pourcent_gdp", names = "country2", facet_col = "year", facet_col_wrap=5, width = 1500, height = 750, color_discrete_sequence=px.colors.qualitative.Pastel2)
            fig.update_traces(textposition='inside', textinfo = "percent+label", hole=.3, pull=[0.2])
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))# Pour n'afficher que l'année et pas "year = année"
            fig.update_layout(margin=dict(t=150))
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)   
        if period == time_period[2]:
            df_0515 = pib_continent_country[(pib_continent_country.year == 2005) |(pib_continent_country.year == 2010) |(pib_continent_country.year == 2015)]
            df_0515 = df_0515[df_0515.continent == continent]
            df_0515["country2"] = df_0515.apply(label_countries, axis=1) # Fonction label pays autre



            fig = px.pie(df_0515, values = "pourcent_gdp", names = "country2", facet_col = "year", facet_col_wrap=5, width = 1500, height = 750, color_discrete_sequence=px.colors.qualitative.Pastel2)
            fig.update_traces(textposition='inside', textinfo = "percent+label",hole=.3, pull=[0.2])
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Pour n'afficher que l'année et pas "year = année"
            fig.update_layout(margin=dict(t=150))
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)
        st.write("""Focus sur l'Europe :
                 \n- Sur la période 1910-1950, nous pouvons observer que le Royaume-Unis,
                 l'Allemagne et la France représentent les trois plus gros contributeurs au PIB global de l'Europe.
                 \n- Sur la période 1960, il s'agit de l'Allemagne, le Royaume-Unis et la Russie. 
                  La France ne reprenant sa place sur le "podium", qu'à partir de 1980 au profit du Royaume-Unis.
                 \n- Sur la dernière période de 2005 à 2018, les trois plus gros contributeurs restent l'Allemagne, la Russie et la France.""")       
        # PIB par pays ---------------------------------------------------------------------
        st.divider()
        st.header("Comparaison du PIB entre deux pays")
        col1, col2 = st.columns(2)
        with col1:
            pib_country = df_co2.groupby(["year", "country"])["gdp"].sum().reset_index()
            country_list = list(pib_country["country"].unique())

            country = st.selectbox(label = "PIB Pays A", options = country_list, label_visibility = "hidden", index = 35)

            fig = px.line(pib_country[pib_country["country"] == country], x = "year", y = "gdp", labels=dict(year = "Années", gdp = "PIB mesuré en dollars"))
            if country == "United States" or country == "China":
                fig.update_layout(yaxis_range=[0,16000000000000], title_text = "Attention, échelle modifiée en ordonnée")
            else:
                fig.update_layout(yaxis_range=[0,4000000000000])
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)  
        with col2:
            pib_country = df_co2.groupby(["year", "country"])["gdp"].sum().reset_index()
            country_list = list(pib_country["country"].unique())

            country = st.selectbox(label = "PIB Pays B", options = country_list, label_visibility = "hidden", index = 37)

            fig = px.line(pib_country[pib_country["country"] == country], x = "year", y = "gdp", labels=dict(year = "Années", gdp = "PIB mesuré en dollars"))
            if country == "United States" or country == "China":
                fig.update_layout(yaxis_range=[0,16000000000000], title_text = "Attention, échelle modifiée en ordonnée")   
            else:
                fig.update_layout(yaxis_range=[0,4000000000000])
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)      
        st.write("""Focus sur l'Europe et en particulier sur l'exemple du Royaume-Unis et de la Pologne :
                 \nLa corrélation entre le PIB et les températures semble se cofirmer. En effet, la tendance d’évolution du PIB au Royaume-Unis reflète bien l’évolution des températures de l’Hémisphère Nord. A contrario, le PIB de la Pologne évolue plus lentement et affiche un retard par rapport au Royaume-Unis. """)
    # CO2 ------------------------------------------------------------------------------------------------------------------------------------------------------------
    with tab3:
        # CO2 GLOBAL --------------------------------------------------------------------------
        st.header("Existe-t-il un lien entre la hausse des températures et les émissions de CO2?")
        st.write("""Les émissions de CO2 semblent suivre la même tendance d’évolution que le PIB à l’échelle planétaire. 
                 \nEn effet, nous observons, là aussi, une hausse plus importante à partir des mêmes dates clés, soit les années 1920, 1940 et 1980.  Aussi la hausse des émissions de CO2 devient de plus en plus significative à mesure que les années progressent, comme pour les températures.""")
        co2_global = df_co2.groupby("year")["co2"].sum().reset_index()
        fig = px.line(
            co2_global,
            title = "Evolution des émissions de CO2 à l'échelle mondiale",
            x = "year", y = "co2",
            labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
            )
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True) 
        # CO2 PAR CONTINENT ---------------------------------------------------------------------------
        st.header("Comparaison des émissions de CO2 par continent")
        col1, col2 = st.columns(2)
        with col1:
            co2_continent = df_co2.groupby(["year", "continent"])["co2"].sum().reset_index()
            choix = ["Europe", "Amérique du Nord", "Amérique du Sud", "Asie", "Afrique", "Océanie"]
            option = st.selectbox("Continent A", choix, label_visibility="hidden")
            if option == "Europe":
                fig = px.line(
                co2_continent[co2_continent.continent == "Europe"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Nord":
                fig = px.line(
                co2_continent[co2_continent.continent == "North America"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Sud":
                fig = px.line(
                co2_continent[co2_continent.continent == "South America"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Afrique":
                fig = px.line(
                co2_continent[co2_continent.continent == "Africa"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Asie":
                fig = px.line(
                co2_continent[co2_continent.continent == "Asia"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Océanie":
                fig = px.line(
                co2_continent[co2_continent.continent == "Oceania"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
        with col2:
            co2_continent = df_co2.groupby(["year", "continent"])["co2"].sum().reset_index()
            choix2 = ["Europe", "Amérique du Nord", "Amérique du Sud", "Asie", "Afrique", "Océanie"]
            option = st.selectbox("Continent B", choix2, label_visibility="hidden", index = 1)
            if option == "Europe":
                fig = px.line(
                co2_continent[co2_continent.continent == "Europe"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Nord":
                fig = px.line(
                co2_continent[co2_continent.continent == "North America"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Amérique du Sud":
                fig = px.line(
                co2_continent[co2_continent.continent == "South America"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Afrique":
                fig = px.line(
                co2_continent[co2_continent.continent == "Africa"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Asie":
                fig = px.line(
                co2_continent[co2_continent.continent == "Asia"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
            if option == "Océanie":
                fig = px.line(
                co2_continent[co2_continent.continent == "Oceania"],
                x = "year", y = "co2",
                labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes")
                )
                fig.update_layout(yaxis_range=[0,15000])
                with st.container(border=True):
                    st.plotly_chart(fig, use_container_width=True)
        # CO2 PAR pays par continent ------------------------------------------------------------------
        st.divider()
        st.header("Contibution aux émissions de CO2 des pays dans leur continent")
        co2_continent_country = df_co2
        co2_continent_country["total_co2_per_year"] = co2_continent_country.groupby(["year", "continent"])["co2"].transform("sum")
        co2_continent_country["pourcent_co2"] = (co2_continent_country.co2 / co2_continent_country.total_co2_per_year) * 100
        co2_continent_country = co2_continent_country.dropna(axis = 0, subset="continent")
        
        continent_list = ["Europe", "Asia", "North America", "Africa", "South America", "Oceania"]
        continent = st.selectbox(label = "Continent", options = continent_list)
        time_period = ["1930-1970", "1980-2000", "2005-2015"]
        period = st.selectbox(label = "Période", options = time_period)
        def label_countries(row):
            if row['pourcent_co2'] < 5:
                return 'Other(< 5%)'
            else:
                return row['country']
        if period == time_period[0]:
            df_3070 = co2_continent_country[(co2_continent_country.year == 1930) |(co2_continent_country.year == 1950) |(co2_continent_country.year == 1970)]
            df_3070 = df_3070[df_3070.continent == continent]
            df_3070["country2"] = df_3070.apply(label_countries, axis=1)

            fig = px.pie(df_3070, values = "pourcent_co2", names = "country2", facet_col = "year", facet_col_wrap=5, category_orders={"year" : [1930, 1950, 1970]}, width = 1500, height = 750, color_discrete_sequence=px.colors.qualitative.Pastel2)
            fig.update_traces(textposition='inside', textinfo = "percent+label", hole=.3, pull=[0.2])
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Pour n'afficher que l'année et pas "year = année"
            fig.update_layout(margin=dict(t=150))
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)  
        if period == time_period[1]:
            df_8000 = co2_continent_country[(co2_continent_country.year == 1980) |(co2_continent_country.year == 1990) |(co2_continent_country.year == 2000)]
            df_8000 = df_8000[df_8000.continent == continent]
            df_8000["country2"] = df_8000.apply(label_countries, axis=1) # Fonction label pays autre



            fig = px.pie(df_8000, values = "pourcent_co2", names = "country2", facet_col = "year", facet_col_wrap=5, width = 1500, height = 750, color_discrete_sequence=px.colors.qualitative.Pastel2)
            fig.update_traces(textposition='inside', textinfo = "percent+label", hole=.3, pull=[0.2])
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))# Pour n'afficher que l'année et pas "year = année"
            fig.update_layout(margin=dict(t=150))
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)    
        if period == time_period[2]:
            df_0515 = co2_continent_country[(co2_continent_country.year == 2005) |(co2_continent_country.year == 2010) |(co2_continent_country.year == 2015)]
            df_0515 = df_0515[df_0515.continent == continent]
            df_0515["country2"] = df_0515.apply(label_countries, axis=1) # Fonction label pays autre



            fig = px.pie(df_0515, values = "pourcent_co2", names = "country2", facet_col = "year", facet_col_wrap=5, width = 1500, height = 750, color_discrete_sequence=px.colors.qualitative.Pastel2)
            fig.update_traces(textposition='inside', textinfo = "percent+label",hole=.3, pull=[0.2])
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) # Pour n'afficher que l'année et pas "year = année"
            fig.update_layout(margin=dict(t=150))
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)  
        # CO2 PAR PAYS ---------------------------------------------------------------------------
        st.header("Comparaison des émissions de CO2 par pays")
        col1, col2 = st.columns(2)
        with col1:
            co2_country = df_co2.groupby(["year", "country"])["co2"].sum().reset_index()
            country_list = list(co2_country["country"].unique())

            country = st.selectbox(label = "CO2 Pays A", options = country_list, label_visibility="hidden", index = 35)

            fig = px.line(co2_country[co2_country["country"] == country], x = "year", y = "co2", labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes"))
            if country == "United States" or country == "China":
                fig.update_layout(yaxis_range=[0,15000], title_text = "Attention, échelle modifiée en ordonnée")
            else:
                fig.update_layout(yaxis_range=[0,1500])
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)  
        with col2:
            co2_country = df_co2.groupby(["year", "country"])["co2"].sum().reset_index()
            country_list = list(co2_country["country"].unique())

            country = st.selectbox(label = "CO2 Pays B", options = country_list, label_visibility="hidden", index = 37)

            fig = px.line(co2_country[co2_country["country"] == country], x = "year", y = "co2", labels=dict(year = "Années", co2 = "CO2 mesuré en millions de tonnes"))
            if country == "United States" or country == "China":
                fig.update_layout(yaxis_range=[0,15000], title_text = "Attention, échelle modifiée en ordonnée")
            else:
                fig.update_layout(yaxis_range=[0,1500])
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)
    with tab4:
        st.header("L'élévation de la température terrestre est-elle corrélée à l'activité humaine?")
        choix = ["Températures - PIB", "PIB - CO2", "CO2 - Températures"]
        option = st.selectbox("Choix des variables", choix)
        if option == "Températures - PIB":
            st.divider()
            st.header("Corrélation entre les températures et le PIB")
            df_corr_temp_pib = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/corr_temp_pib")
            test_p = pearsonr(x = df_corr_temp_pib["gdp"], y = df_corr_temp_pib["temp"])
            fig = px.scatter(
                df_corr_temp_pib,
                x = "gdp",
                y = "temp",
                color = "year",
                width = 1500, height = 750,
                labels = {"temp": "Anomalie de températures en °C", "gdp" : "PIB mesuré en dollars", "year" : "Année"},
                color_continuous_scale=px.colors.sequential.Oranges
                )
            fig.update_traces(marker={'size': 8})
            fig.update_layout(margin=dict(t=150), title_text="Coefficient de corrélation Pearson : 0.92 <br> p-valeur =~ 0")
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True) 
        if option == "PIB - CO2":
            st.divider()
            st.header("Corrélation entre le PIB et le CO2")
            df_corr_co2_pib = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/corr_co2_PIB.csv")
            test_p = pearsonr(x = df_corr_co2_pib["co2"], y = df_corr_co2_pib["gdp"]) # Calcul Pearson
            fig = px.scatter(
                df_corr_co2_pib,
                x = "gdp",
                y = "co2",
                color = "year",
                width = 1500, height = 750,
                labels = {"gdp": "PIB mesuré en dollars", "co2" : "Emissions de CO2 mesurées en millions tonnes", "year" : "Année"},
                color_continuous_scale=px.colors.sequential.Oranges
                )
            fig.update_layout(margin=dict(t=150), title_text="Coefficient de corrélation Pearson : 0.93 <br> p-valeur : 0")
            fig.update_traces(marker={'size': 8})
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)            
        if option == "CO2 - Températures":
            st.divider()
            st.header("Corrélation entre les températures et le CO2")
            df_corr_co2_temp = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/corr_CO2_TEMP.csv")
            test_p = pearsonr(x = df_corr_co2_temp["co2"], y = df_corr_co2_temp["temp"]) # Calcul de corrélation Pearson.
            fig = px.scatter(
                df_corr_co2_temp,
                x = "co2",
                y = "temp",
                color = "year",
                width = 1500, height = 750,
                labels = {"temp": "Anomalie de températures en °C", "co2" : "Emissions de CO2 mesurées en millions tonnes", "year" : "Année"},
                color_continuous_scale=px.colors.sequential.Oranges
                )
            fig.update_layout(margin=dict(t=150), title_text="Coefficient de corrélation Pearson : 0.93 <br> p-valeur =~ 0 ")
            fig.update_traces(marker={'size': 8})
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True)  
# PAGE 4 MACHINE LEARNING ------------------------------------------------------------------------------------------------------------------------------------------------------------
if page == pages[3]:

    # TEMPERATURES
    regressor = load("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Regression_lineaire_temp.joblib")
    df_plotly = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/DF_PLOTLY.csv")
    df_ml = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/DF_ML_TEMP.csv")
    df_ml = df_ml.drop("Unnamed: 0", axis = 1)

    y_pred2 = regressor.predict(df_ml)

    df_plotly["pred"] = y_pred2 # ajout valeurs prédites


    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df_plotly["year"], y=df_plotly.pred_low,
        fill=None,
        mode="lines",
        line=dict(color = "orange", shape = "spline"),
        name="Limites intervalles de prédiction(*) <br><sup> * : Marge d'erreur de 0.07°C%."))
    fig_temp.add_trace(go.Scatter(
        x=df_plotly.year,
        y=df_plotly.pred_high,
        fill="tonexty",
        mode="lines", 
        line=dict(color = "orange", shape = "spline"),
        name="Intervalle de prédiction(*)"))
    fig_temp.add_trace(go.Scatter(
        x=df_plotly.year,
        y=df_plotly.temp,
        mode="lines", line_color="red", name="Valeurs réelles",
        line_shape="spline"))
    fig_temp.add_trace(go.Scatter(
        x=df_plotly.year,
        y=df_plotly.pred,
        line=dict(color="darkorange", dash="dot", shape="spline"),
        name="Valeurs prédites"))
    fig_temp.update_layout(yaxis=dict(range=[-1, 1.5]))
    fig_temp.update_layout(title="Observation de la différence entre les valeurs réelles de températures et les prédictions sur l'ensemble de nos données <br><sup> Modèle : Régression linéaire",
                
                    yaxis_title = "Anomalies de températures en °C",
                    xaxis_title = "Années", title_x=0.25)
    fig_temp.update_layout(
        autosize=True,
        width=1200,
        height=600)
    fig_temp.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    #------------------------------------------------------------

    #CO2 
    regressor_co2 = load("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Regression_lineaire_co2.joblib")
    df_plotly_co2 = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/DF_PLOTLY_CO2")
    df_ml_co2 = pd.read_csv("/workspaces/Streamlit/Projet températures terrestres/Streamlit/DF_ML_CO2")
    df_ml_co2 = df_ml_co2.drop("Unnamed: 0", axis = 1)
    y_pred_co2 = regressor_co2.predict(df_ml_co2)   

    df_plotly_co2["pred"] = y_pred_co2
    df_plotly_co2 = df_plotly_co2.groupby("year").sum().reset_index()
    def calculate_margin_of_error(df_plotly_co2):

        # Calcul de la marge d'erreur niveau de confiance 99%.  
        margin_of_error =  2.58 * np.std(df_plotly_co2["pred"]) / np.sqrt(len(df_plotly_co2))


        # ajout des colonnes au DF
        df_plotly_co2["pred_low"] = df_plotly_co2["pred"] - margin_of_error
        df_plotly_co2["pred_high"] = df_plotly_co2["pred"] + margin_of_error

        return df_plotly_co2
    calculate_margin_of_error(df_plotly_co2)
    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(
        x=df_plotly_co2.year, 
        y=df_plotly_co2.pred_low,
        fill=None,
        mode="lines",
        line=dict(color = "orange", 
                width = 0.5, 
                shape = "spline"),
        name="Limites intervalles de prédiction (*) <br><sup> * : Marge d'erreur de 3,5K millions de tonnes de Co2"))
    fig_co2.add_trace(go.Scatter(
        x=df_plotly_co2.year,
        y=df_plotly_co2.pred_high,
        fill="tonexty",
        mode="lines", 
        line=dict(color = "orange", 
                width = 0.5, 
                shape = "spline"),
        name="Intervalle de prédiction (*)"))
    fig_co2.add_trace(go.Scatter(
        x=df_plotly_co2.year,
        y=df_plotly_co2.co2,
        mode="lines", 
        line_color="red", 
        name="Valeurs réelles"))
    fig_co2.add_trace(go.Scatter(
        x=df_plotly_co2.year,
        y=df_plotly_co2.pred,
        line=dict(color="darkorange", 
                dash="dot"),
        name="Valeurs prédites"))

    fig_co2.update_layout(title="Observation de la différence entre les valeurs réelles d'émissions de co2 et les prédictions sur toutes nos données <br><sup> Modèle : Régression linéaire",
                
                    yaxis_title = "Co2 mesuré en millions de tonnes",
                    xaxis_title = "Années", title_x=0.25)
    fig_co2.update_layout(
        autosize=True,
        width=1200,
        height=600)
    fig_co2.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ))

    #------------------------------------------------------------
    st.title("Machine learning")
    tab1, tab2 = st.tabs(["Machine learning - Températures", "Machine learning - CO2"])

    with tab1:
        st.divider()
        st.header("Températures")
        st.divider()

        st.write("""Après avoir pu démontré une certaine corrélation entre la hausse des températures
                et le CO2, nous avons déterminé qu'il était possible de partir de l'hypothèse selon laquelle les températures 
                et les émissions CO2 augmentent à travers le temps et que ces hausses soient, 
                en partie, liées aux variables que nous avons dans notre jeu de données.""")

        st.write("""Nous avons choisi trois modèles pour essayer de prédire les températures :
                \n - La régression linéaire : Ce modèle nous permettrait en effet d'apprécier la linéarité de notre jeu de données basé sur une évolution par année et d'ainsi prédire cette évolution efficacement (aucun paramètre particulier choisi).
                \n - Le decision tree regressor : Ce modèle de régression est assez puissant et polyvalent. Nous avons donc décidé de l'inclure dans notre première approche (random_state = 42).
                \n - Le random forest regressor : De la même manière que le Decision Tree Regressor, ce modèle est puissant et polyvalent. Il est donc intéressant d'observer son utilisation sur notre jeu de données (random_state = 42).""")

        st.write("\n Résultats de nos premiers modèles :")
        with st.container(border=True):
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Resultats_1.png")
        st.write("""Les modèles ne semblent pas capables de prédire efficacement les températures avec le jeu de données à disposition. 
                Cependant, il est à noter que ces coefficients sont bons sur les valeurs d'entrainement.
                Dans un premier temps, nous avons attribué ce manque de précision à un jeu de données trop petit. 
                En effet, lors de la préparation des données, il a fallu grouper nos données par année afin d'avoir quelque 
                chose d'exploitable et interprétable ce qui a considérablement réduit le nombre de lignes.""")
        st.divider()
        st.write("""Après avoir essayé une méthode peu concluante consistant à augmenter le nombre de lignes dans notre jeu de données puis appliquer
                un bruit aux données, nous avons décidé d'essayer de focaliser nos modèles sur les paramètres importants :""")
        
        st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/variables importantes.png")
        st.divider()
        st.write("""Nous n'avons ainsi gardé que les variables "temperature_change_from_co2", "gas_co2", "population","cumulative_co2" et "gdp" et les résultats se sont révélés bien meilleurs : """)
        with st.container(border=True):
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Resultats_2.png")
        st.divider()
        st.write("""Enfin, nous pouvons mettre en forme nos résultats dans un graphique. Celui-ci représente la différence entre les valeurs observées et les valeurs réelles.
                On remarque que, malgré que le modèle semble capable de prédire correctement les valeurs communes, il semble être bien 
                moins performant sur les valeurs extrêmes. Ceci peut s'expliquer par la corrélation négative entre certaines variables et la températures dans notre modèle.""")
        with st.container(border=True):
            st.plotly_chart(fig_temp, use_container_width=True)
    with tab2:
        st.divider()
        st.header("CO2")
        st.divider()

        st.write("""Il est intéressant d'observer si les modèles seront plus capables de prédire les émissions de CO2. En effet,
                nous pouvons a priori penser que, notre jeu de données 'Owid' contenant beaucoup de données explicatives 
                sur les émissions de CO2, il serait plus facile de prédire cette valeur avec ces variables.""")

        st.write("""Nous avons choisi trois modèles pour essayer de prédire les émissions de CO2 :
                \n - La régression linéaire : Ce modèle nous permettrait en effet d'apprécier la linéarité de notre jeu de données basé sur une évolution par année et d'ainsi prédire cette évolution efficacement (aucun paramètre particulier choisi).
                \n - Le decision tree regressor : Ce modèle de régression est assez puissant et polyvalent. Nous avons donc décidé de l'inclure dans notre première approche (random_state = 42).
                \n - Le random forest regressor : De la même manière que le Decision Tree Regressor, ce modèle est puissant et polyvalent. Il est donc intéressant d'observer son utilisation sur notre jeu de données (random_state = 42).""")
        st.divider()
        st.write("Comme pour les températures, nous avons souhaité observé la différence de performance des modèles après avoir selectionné les variables les plus importantes. Cependant, la différence étant insignifiante, nous avons choisi de garder toutes nos variables :")
        col1, col2 = st.columns(2)
        with col1:
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Resultats_1co2.png", 
                    width = 800, 
                    caption = "Résultats de nos modèles avant sélection des variables les plus importantes")
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Metriques CO2.png", 
            width = 800, 
            caption = "Métriques pour nos modèles sans sélection des variables les plus importantes")
            
        with col2:
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/Resultats_2co2.png", 
                    width = 800, 
                    caption = "Résultats de nos modèles après sélection des variables les plus importantes")
            st.image("/workspaces/Streamlit/Projet températures terrestres/Streamlit/variables importantes co2.png", 
                    width = 800, 
                    caption = "Variables les plus importantes dans le modèle de régression linéaire")
        st.divider()
        st.write("""Enfin, nous pouvons mettre en forme nos résultats dans un graphique. Celui-ci représente la différence entre les valeurs observées et les valeurs réelles.
                On remarque que, malgré que le modèle semble capable de prédire correctement les émissions de CO2. En effet, nous n'observons pas de valeur réelle en dehors de la plage de prédiction.""")
        st.plotly_chart(fig_co2, use_container_width=True)

# PAGE 5 CONCLUSION PROJET
if page == pages[4]:
    st.title("Conclusion projet")
    st.divider()
    st.write("""Mener à bien ce projet a été riche en enseignements pour chacun d’entre nous. Nous avons vu notre gout du travail en équipe renforcé, et nous nous sommes mutuellement challengés afin de trouver les meilleurs solutions pour résoudre les problèmes rencontrés.
\nCe projet concret nous a permis d’aller plus loin dans l'exploration des données par rapport à notre formation en cours, en nous confrontons à des problématiques réelles. Ce qui nous a permis de comprendre que la manipulation et l’analyse de données propose un champ d'application bien plus large que le monde de l'entreprise.
\nAussi, cet exercice a confirmé et à renforcer notre vif intérêt pour l’exploration de la data.
\nNous aurions cependant souhaité aller plus loin dans ce projet en coconstruisant un modèle de prédiction des températures futures car nos explorations et modèles ont permis d'observer un lien entre la hausse des températures et l'activité humaine.""")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.header("Source des données")
        st.write("""
                - Dataset NASA :rocket: : https://data.giss.nasa.gov/gistemp/ 
                \n- Dataset OWID :office: : https://github.com/owid/co2-data""")
    st.divider()
    st.header("Remerciements")
    st.write("""
             \nNous tenons à remercier Alain Ferlac, notre mentor pour ce projet pour son implication et ses conseils.
             \nNous remercions également Yazid Msaadi, le chef de notre cohorte pour sa disponibilité.
             \nEnfin, nous remercions DataScientest pour le contenu de la formation qui nous a permis de mener, au mieux, ce projet.""")

