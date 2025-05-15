import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import pandapower as pp
import re
import ast
import nest_asyncio
import ee
from streamlit_folium import st_folium
from shapely.geometry import LineString
from datetime import datetime, timedelta
import random
import geemap
import numpy as np
import math
import traceback
from shapely.geometry import LineString, Point


# Set page configuration
st.set_page_config(
    page_title="Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Earth Engine authentication - for Streamlit Cloud deployment
@st.cache_resource
def initialize_ee():
    if 'EE_SERVICE_ACCOUNT_JSON' in st.secrets:
        service_account_info = st.secrets["EE_SERVICE_ACCOUNT_JSON"]
        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'], 
            key_data=service_account_info['private_key']
        )
        ee.Initialize(credentials)
    else:
        try:
            ee.Initialize(project='ee-hasan710')
        except Exception as e:
            st.error(f"Error initializing Earth Engine: {e}")
            st.error("Please make sure you have authenticated with Earth Engine locally or configured secrets for deployment.")

# Initialize Earth Engine
initialize_ee()


# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["About the App and Developers", "Network Initialization", "Weather Risk Visualisation Using Google Earth Engine", "Projected Operation - Under Current OPF", "Projected Operation - Under Weather Risk Aware OPF", "Data Analytics"]
selection = st.sidebar.radio("Go to", pages)

# Shared session state initialization
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "network_data" not in st.session_state:
    st.session_state.network_data = None
if "map_obj" not in st.session_state:
    st.session_state.map_obj = None
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None
if "weather_map_obj" not in st.session_state:
    st.session_state.weather_map_obj = None

# Apply nest_asyncio for async support in Streamlit (used in Network Initialization)
nest_asyncio.apply()

# Shared function: Add EE Layer to Folium Map (used in both pages)
def add_ee_layer(self, ee_object, vis_params, name):
    try:
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
    except Exception as e:
        st.error(f"Could not display {name}: {str(e)}")

# Attach the method to folium.Map
folium.Map.add_ee_layer = add_ee_layer



# Shared function: Create and display the map (used in Network Initialization)
def create_map(df_line):
    try:
        # Process geodata
        df_line["geodata"] = df_line["geodata"].apply(
            lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
        )
        line_geometries = [LineString(coords) for coords in df_line["geodata"]]
        gdf = gpd.GeoDataFrame(df_line, geometry=line_geometries, crs="EPSG:4326")

        # Create Folium map
        m = folium.Map(location=[30, 70], zoom_start=5, width=700, height=500)

        # Convert GeoDataFrame to EE FeatureCollection
        features = [ee.Feature(ee.Geometry.LineString(row["geodata"])) for _, row in df_line.iterrows()]
        line_fc = ee.FeatureCollection(features)

        # Add transmission lines to the map
        m.add_ee_layer(line_fc.style(**{'color': 'black', 'width': 2}), {}, "Transmission Lines")

        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)

        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None


# Page 1: Network Initialization
if selection == "Network Initialization":
    # Primary project title
    st.title("Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine")

    # Secondary page-specific title
    st.header("Network Initialization")

    # # File uploader for the Excel file
    # uploaded_file = st.file_uploader("Upload your network Excel file (e.g., Final_IEEE_9Bus_Parameters_only.xlsx)", type=["xlsx"], key="file_uploader")

    st.markdown(
    "Donot have an Excel File in our specified format? [Download the sample IEEE‑9 or 14 bus network parameters](https://drive.google.com/drive/folders/1oT10dY6hZiM0q3AYiFzEqe_GQ5vA-eEa?usp=sharing) "
    "from Google Drive.",
    unsafe_allow_html=True,
    )
    
    # File uploader for the Excel file
    uploaded_file = st.file_uploader(
        "Upload your network Excel file",
        type=["xlsx"],
        key="file_uploader",
        help="You can also use the template from Google Drive: "
             "[Sample Excel](https://drive.google.com/drive/folders/1oT10dY6hZiM0q3AYiFzEqe_GQ5vA-eEa?usp=sharing)",
    )
    
    # Check if a new file was uploaded
    if uploaded_file is not None and st.session_state.uploaded_file_key != uploaded_file.name:
        st.session_state.show_results = False
        st.session_state.network_data = None
        st.session_state.map_obj = None
        st.session_state.uploaded_file_key = uploaded_file.name
        st.session_state.uploaded_file = uploaded_file  # Store the file object

    if uploaded_file is not None and not st.session_state.show_results:
        # Create an empty pandapower network
        net = pp.create_empty_network()

        # --- Create Buses ---
        df_bus = pd.read_excel(uploaded_file, sheet_name="Bus Parameters", index_col=0)
        for idx, row in df_bus.iterrows():
            pp.create_bus(net,
                          name=row["name"],
                          vn_kv=row["vn_kv"],
                          zone=row["zone"],
                          in_service=row["in_service"],
                          max_vm_pu=row["max_vm_pu"],
                          min_vm_pu=row["min_vm_pu"])

        # --- Create Loads ---
        df_load = pd.read_excel(uploaded_file, sheet_name="Load Parameters", index_col=0)
        for idx, row in df_load.iterrows():
            pp.create_load(net,
                           bus=row["bus"],
                           p_mw=row["p_mw"],
                           q_mvar=row["q_mvar"],
                           in_service=row["in_service"])

        # --- Create Transformers (if sheet exists) ---
        df_trafo = None
        if "Transformer Parameters" in pd.ExcelFile(uploaded_file).sheet_names:
            df_trafo = pd.read_excel(uploaded_file, sheet_name="Transformer Parameters", index_col=0)
            for idx, row in df_trafo.iterrows():
                pp.create_transformer_from_parameters(net,
                                                      hv_bus=row["hv_bus"],
                                                      lv_bus=row["lv_bus"],
                                                      sn_mva=row["sn_mva"],
                                                      vn_hv_kv=row["vn_hv_kv"],
                                                      vn_lv_kv=row["vn_lv_kv"],
                                                      vk_percent=row["vk_percent"],
                                                      vkr_percent=row["vkr_percent"],
                                                      pfe_kw=row["pfe_kw"],
                                                      i0_percent=row["i0_percent"],
                                                      in_service=row["in_service"],
                                                      max_loading_percent=row["max_loading_percent"])

        # --- Create Generators/External Grid ---
        df_gen = pd.read_excel(uploaded_file, sheet_name="Generator Parameters", index_col=0)
        df_gen['in_service'] = df_gen['in_service'].astype(str).str.strip().str.upper().map({'TRUE': True, 'FALSE': False}).fillna(True)
        df_gen['controllable'] = df_gen['controllable'].astype(str).str.strip().str.upper().map({'TRUE': True, 'FALSE': False})
        for idx, row in df_gen.iterrows():
            if row["slack_weight"] == 1:
                ext_idx = pp.create_ext_grid(net,
                                             bus=row["bus"],
                                             vm_pu=row["vm_pu"],
                                             va_degree=0)
                pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                                    cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                    cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                    cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                    cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],
                                    cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                    cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])
            else:
                gen_idx = pp.create_gen(net,
                                        bus=row["bus"],
                                        p_mw=row["p_mw"],
                                        vm_pu=row["vm_pu"],
                                        min_q_mvar=row["min_q_mvar"],
                                        max_q_mvar=row["max_q_mvar"],
                                        scaling=row["scaling"],
                                        in_service=row["in_service"],
                                        slack_weight=row["slack_weight"],
                                        controllable=row["controllable"],
                                        max_p_mw=row["max_p_mw"],
                                        min_p_mw=row["min_p_mw"])
                pp.create_poly_cost(net, element=gen_idx, et="gen",
                                    cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                    cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                    cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                    cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],
                                    cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                    cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])

        # --- Create Lines ---
        df_line = pd.read_excel(uploaded_file, sheet_name="Line Parameters", index_col=0)
        for idx, row in df_line.iterrows():
            if isinstance(row["geodata"], str):
                geodata = ast.literal_eval(row["geodata"])
            else:
                geodata = row["geodata"]
            pp.create_line_from_parameters(net,
                                           from_bus=row["from_bus"],
                                           to_bus=row["to_bus"],
                                           length_km=row["length_km"],
                                           r_ohm_per_km=row["r_ohm_per_km"],
                                           x_ohm_per_km=row["x_ohm_per_km"],
                                           c_nf_per_km=row["c_nf_per_km"],
                                           max_i_ka=row["max_i_ka"],
                                           in_service=row["in_service"],
                                           max_loading_percent=row["max_loading_percent"],
                                           geodata=geodata)

        # --- Read Dynamic Profiles ---
        df_load_profile = pd.read_excel(uploaded_file, sheet_name="Load Profile")
        df_load_profile.columns = df_load_profile.columns.str.strip()

        df_gen_profile = pd.read_excel(uploaded_file, sheet_name="Generator Profile")
        df_gen_profile.columns = df_gen_profile.columns.str.strip()

        # --- Build Dictionaries for Dynamic Column Mapping ---
        load_dynamic = {}
        for col in df_load_profile.columns:
            m = re.match(r"p_mw_bus_(\d+)", col)
            if m:
                bus = int(m.group(1))
                q_col = f"q_mvar_bus_{bus}"
                if q_col in df_load_profile.columns:
                    load_dynamic[bus] = {"p": col, "q": q_col}

        gen_dynamic = {}
        for col in df_gen_profile.columns:
            if col.startswith("p_mw"):
                numbers = re.findall(r'\d+', col)
                if numbers:
                    bus = int(numbers[-1])
                    gen_dynamic[bus] = col

        # Store network data in session state
        st.session_state.network_data = {
            'df_bus': df_bus,
            'df_load': df_load,
            'df_gen': df_gen,
            'df_line': df_line,
            'df_load_profile': df_load_profile,
            'df_gen_profile': df_gen_profile,
            'df_trafo': df_trafo  # Add transformer data to session state
        }

    # --- Button to Display Results ---
    if st.button("Show Excel Network Parameters") and uploaded_file is not None:
        st.session_state.show_results = True
        # Generate map if not already generated
        if st.session_state.map_obj is None and st.session_state.network_data is not None:
            with st.spinner("Generating map..."):
                st.session_state.map_obj = create_map(st.session_state.network_data['df_line'])

    # --- Display Results ---
    if st.session_state.show_results and st.session_state.network_data is not None:
        st.subheader("Network Parameters")

        # Display Bus Parameters
        st.write("### Bus Parameters")
        st.dataframe(st.session_state.network_data['df_bus'])

        # Display Load Parameters
        st.write("### Load Parameters")
        st.dataframe(st.session_state.network_data['df_load'])

        # Display Generator Parameters
        st.write("### Generator Parameters")
        st.dataframe(st.session_state.network_data['df_gen'])

        # Display Transformer Parameters (if exists)
        if st.session_state.network_data['df_trafo'] is not None:
            st.write("### Transformer Parameters")
            st.dataframe(st.session_state.network_data['df_trafo'])

        # Display Line Parameters
        st.write("### Line Parameters")
        st.dataframe(st.session_state.network_data['df_line'])

        # Display Load Profile
        st.write("### Load Profile")
        st.dataframe(st.session_state.network_data['df_load_profile'])

        # Display Generator Profile
        st.write("### Generator Profile")
        st.dataframe(st.session_state.network_data['df_gen_profile'])

        # Display Map
        st.subheader("Transmission Network Map")
        if st.session_state.map_obj is not None:
            st_folium(st.session_state.map_obj, width=800, height=600, key="network_map")
            st.success("Network uploaded successfully!")
        else:
            st.warning("Map could not be generated.")

    # --- Clear Results Button ---
    if st.session_state.show_results and st.button("Clear Results"):
        st.session_state.show_results = False
        st.session_state.network_data = None
        st.session_state.map_obj = None
        st.session_state.uploaded_file_key = None
        st.experimental_rerun()

    if uploaded_file is None and not st.session_state.show_results:
        st.info("Please upload an Excel file to proceed.")
        
# Page 2: Weather Risk Visualisation Using GEE
elif selection == "Weather Risk Visualisation Using Google Earth Engine":
    st.title("Weather Risk Visualisation Using GEE")
    
    # Hardcoded output of page 2
    line_outages = [(3, 4, 15), (7, 1, 12)]

    # Set session state for page 3
    st.session_state.line_outage_data = {
        "lines": [(3, 4), (7, 1)],
        "hours": [15, 12],
        "risk_scores": [14, 16]  # Adjust these values based on your testing needs
    }
    
    # Display hardcoded data for verification
    st.write("**Using hardcoded data for testing purposes**")
    
    # Display line_outages as a table
    st.write("Hardcoded Line Outages:")
    df_outages = pd.DataFrame(line_outages, columns=["From Bus", "To Bus", "Outage Hour"])
    st.dataframe(df_outages)                

# Page 3: Projected future operations - Under Current OPF
elif selection == "Projected Operation - Under Current OPF":
    st.title("Projected Operation - Under Current OPF")
    
    
# ────────────────────────────────────────────────────────────────────────────
# Page 4 :  Weather‑Aware System
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Projected Operation - Under Weather Risk Aware OPF":
    st.title("Projected Operation - Under Weather Risk Aware OPF")

        
elif selection == "Data Analytics":
   
    st.title("Data Analytics")

   
    

# ────────────────────────────────────────────────────────────────────────────
# Page 0 :  About the App
# ────────────────────────────────────────────────────────────────────────────
elif selection == "About the App and Developers":
    st.title("Continuous Monitoring of Climate Risks to Electricity Grids using Google Earth Engine")

    st.markdown(
        """
        ### Overview  
        This web application gives an end to end decision support workflow for Grid Operators. It contains following five pages whose description is as follows:

        1. **Network Initialization** – This page ask user to input the Excel File containing Transmission Network Information.  
        2. **Weather‑Risk Visualisation** – This page ask user to set the Weather Analysis Parameters (see below for their details) and then utilize Google Earth Engine to analyze historic and forecasted weather data for day ahead.  
        3. **Projected Operation - Under Current OPF** – This page ask user to select contingency mode (see below for its details) and then yield 24 hourly electric grid operations along with the visualization on map for day ahead. This mode represents the usual operations of electric utilities where the generation does not account for historic weather data and projected extreme weather events that would cause transmissions lines to fail.  
        4. **Projected Operation - Under Weather Risk Aware OPF** – This page ask user to select contingency mode (see below for its details) and then yield 24 hourly operations along with the visualization on map for day ahead. This mode shows the vitality of our tool when it helps utilities to prepare the generation schedule for day ahead while incorporating historic and forecasted weather data and extreme weather risks to the electric grid. 
        5. **Data Analytics** – This page comprises of interactive comparative plots to show comparative analysis between the Projected Operations Under Current OPF vs Weather Risk Aware OPF in terms of cost, amount of load shedding, line loadings, estimated revenue loss under the “Projected Operation Under Current OPF” scenario and the hourly generation and load values. 
        
        The goal is to **quantify the technical and economic benefit** of risk aware dispatch decisions—highlighting potential lost revenue and critical load not served under various contingencies.

        **While an analysis is running, please remain on that page until it finishes. Once the process is complete, you’re free to navigate to any page and explore all options.**
        
        ---

        ### Want to learn more about our Web App?  
        * 📄 **Full Research Thesis** – [Google Drive (PDF)](https://drive.google.com/drive/folders/1mzGOuPhHn2UryrB2q5K4AZH2bPutvNhF?usp=drive_link)  
        * ▶️ **Video Walk‑Through / Tutorial** – [YouTube](https://youtu.be/your-tutorial-video)  

        ---

        ### Key Terminologies

        1)	Weather Analysis Parameters: These are the three parameters set by grid operators.
            *	Risk Tolerance (Low, Medium, and High)
            *	Study Period (Weekly, Monthly)
            *	Risk Score Threshold (6-18)
        2)	Projected Operation Under Current OPF and Projected Operation Under Weather Risk Aware OPF has following options.
            *  Contingency Mode Selection

        ### Risk Tolerance
        
        * Low: In the Low option, the following weather conditions are considered as thresholds beyond which the weather conditions would cause increased vulnerability to that specific region and a threat to electric network. The threshold values are:                                                                   Temperature > 35°C, Precipitation > 50 mm, Wind > 10 m/s.
        
        * Medium: In the Medium option, the following weather conditions are considered as thresholds beyond which the weather conditions would cause increased vulnerability to that specific region and a threat to electric network. The threshold values are: Temperature > 38°C, Precipitation > 100 mm, Wind > 15 m/s.
        
        * High: In the High option, the following weather conditions are considered as thresholds beyond which the weather conditions would cause increased vulnerability to that specific region and a threat to electric network. The threshold values are: Temperature > 41°C, Precipitation > 150 mm, Wind > 20 m/s.
        
        We can also say that these parameters would be based on how resilient an input network is. With Low means the network is least resilient and high means that network is strong against the extreme weather events.

        ### Study Period
        
        * Weekly: Under this option the tool will use weekly weather information (weekly aggregated data) for the historic weather analysis.
        
        * Monthly: Under this option the tool will use monthly weather information (monthly aggregated data) for the historic weather analysis.

        ### Risk Score Threshold
        
        * Risk Score can be chosen on a scale of 6-18 which is important for post weather data analysis. Using our novel Risk scoring Algorithm, when the risk scores are generated for each transmission lines for day ahead, this parameter decides which lines would fail on which projected hour during upcoming extreme weather event.

        ### Contingency Mode Selection:
        
        The Contingency Mode parameter allows the user to define the operational scope of the system’s vulnerability simulation by selecting between two distinct failure modeling strategies. This choice directly impacts the number of lines that would be down after risk scores have been computed for all transmission lines.
        
        * Capped Contingency Mode: This mode evaluates system stability under a constrained failure scenario, assuming that only 20% of the at-risk transmission lines (as identified by the risk score threshold) of the total lines will fail. Any additional forecasted failures beyond this cap are deprioritized, reflecting conservative grid planning under limited disruption assumptions.
        
        * Maximum Contingency Mode: In contrast, this mode simulates a worst-case scenario by assuming that all transmission lines flagged as high risk will fail. It supports comprehensive stress-testing of the network, providing insights into cascading failure risks, load redistribution behavior, and potential stability violations under extreme weather-induced conditions

         ---
         
        ### Key Features
        * **Google Earth Engine Integration** is utilized for having rich historic weather data as well as forecasted weather data. 
        * **Pandapower** is utilized for performing Optimal Power Flow (OPF) Analysis for finding optimized generation dispatch and calculation of lost load. 
        * **GEE, Folium based maps and Plotly analytics** are used for hourly visualization in both scenarios and interactive plots in comparative analysis.

        ---

        ### Usage Workflow
        1. Navigate left‑hand sidebar → **Network Initialization** and upload your Excel model.  
        2. Tune thresholds on **Weather Risk Visualisation** and press *Process*.  
        3. Run **Projected Operation - Under Current OPF** → then **Projected Operation Under Weather Risk Aware OPF**.  
        4. Explore comparative plots in **Data Analytics**.  

        *(You can re‑run any page; session‑state keeps everything consistent.)*

        ---

        ### Data Sources 

        This tool has utilized Google Earth Engine (GEE) which is a cloud-based platform designed for large scale analysis of geospatial data. Its three data sets have been utilized in this tool
        
        * ERA‑5 The following dataset is utilized for historic weather analysis [Link](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY)
        
        * ERA 5 Land reanalysis: The following dataset is utilized for historic weather analysis [Link](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY)

        * NOAA GFS forecasts: The following dataset is utilized to get hourly weather forecast [Link](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25)

        ---

        ### Authors & Contact  
        * **Muhammad Hasan Khan** – BSc Electrical Engineering, Habib University  
          * ✉️ iamhasan710@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](www.linkedin.com/in/hasankhan710)  
        * **Munim ul Haq** – BSc Electrical Engineering, Habib University  
          * ✉️ themunimulhaq24@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](https://www.linkedin.com/in/munim-ul-haq/) 
        * **Syed Muhammad Ammar Ali Jaffri** – BSc Electrical Engineering, Habib University  
          * ✉️ ammarjaffri6515@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](https://www.linkedin.com/in/ammarjaffri/) 

        ### Faculty Supervisor  
        * **Muhammad Umer Tariq** – Assistant Professor, Electrical and Computer Engineering at Habib University  
          * ✉️ umer.tariq@sse.habib.edu.pk  

        _We welcome feedback, and collaboration enquiries._
        """,
        unsafe_allow_html=True
    )









