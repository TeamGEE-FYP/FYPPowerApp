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

# ---------------------------------------------------------------------
# 1) Network INITIALISE  (was `Network_initialize` in Colab)
# ----------------------------------------------------------
def network_initialize(xls_file):
    """
    Re-creates a fresh pandapower network from the uploaded Excel *every* time
    Page-3 (baseline OPF) or Page-4 (weather-aware OPF) is run, so that both
    simulations start from the identical initial state.

    Parameters
    ----------
    xls_file : BytesIO or str
        The same object you get back from `st.file_uploader` (i.e. the upload
        stream).  A local file path also works, so existing unit-tests keep
        passing.

    Returns
    -------
    tuple
        [net, df_bus, df_slack, df_line, num_hours,
         load_dynamic, gen_dynamic,
         df_load_profile, df_gen_profile, *optional df_trafo*]
    """

    # --- 0. Fresh empty network
    net = pp.create_empty_network()

    # --- 1. Read all static sheets ------------------------------------------------
    df_bus   = pd.read_excel(xls_file, sheet_name="Bus Parameters",      index_col=0)
    df_load  = pd.read_excel(xls_file, sheet_name="Load Parameters",     index_col=0)
    df_slack = pd.read_excel(xls_file, sheet_name="Generator Parameters",index_col=0)
    df_line  = pd.read_excel(xls_file, sheet_name="Line Parameters",     index_col=0)

    # --- 2. Build static elements -------------------------------------------------
    for _, row in df_bus.iterrows():
        pp.create_bus(net,
                      name          = row["name"],
                      vn_kv         = row["vn_kv"],
                      zone          = row["zone"],
                      in_service    = row["in_service"],
                      max_vm_pu     = row["max_vm_pu"],
                      min_vm_pu     = row["min_vm_pu"])

    for _, row in df_load.iterrows():
        pp.create_load(net,
                       bus           = row["bus"],
                       p_mw          = row["p_mw"],
                       q_mvar        = row["q_mvar"],
                       in_service    = row["in_service"])

    for _, row in df_slack.iterrows():
        if row["slack_weight"] == 1:
            ext_idx = pp.create_ext_grid(net,
                                         bus        = row["bus"],
                                         vm_pu      = row["vm_pu"],
                                         va_degree  = 0)
            pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                                cp0_eur_per_mw = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])
        else:
            gen_idx = pp.create_gen(net,
                                    bus         = row["bus"],
                                    p_mw        = row["p_mw"],
                                    vm_pu       = row["vm_pu"],
                                    min_q_mvar  = row["min_q_mvar"],
                                    max_q_mvar  = row["max_q_mvar"],
                                    scaling     = row["scaling"],
                                    in_service  = row["in_service"],
                                    slack_weight= row["slack_weight"],
                                    controllable= row["controllable"],
                                    max_p_mw    = row["max_p_mw"],
                                    min_p_mw    = row["min_p_mw"])
            pp.create_poly_cost(net, element=gen_idx, et="gen",
                                cp0_eur_per_mw = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])

    for _, row in df_line.iterrows():
        if pd.isna(row["parallel"]):
            continue
        geodata = ast.literal_eval(row["geodata"]) if isinstance(row["geodata"], str) else row["geodata"]
        pp.create_line_from_parameters(net,
                                       from_bus             = row["from_bus"],
                                       to_bus               = row["to_bus"],
                                       length_km            = row["length_km"],
                                       r_ohm_per_km         = row["r_ohm_per_km"],
                                       x_ohm_per_km         = row["x_ohm_per_km"],
                                       c_nf_per_km          = row["c_nf_per_km"],
                                       max_i_ka             = row["max_i_ka"],
                                       in_service           = row["in_service"],
                                       max_loading_percent  = row["max_loading_percent"],
                                       geodata              = geodata)

    # --- 3. Optional transformers -------------------------------------------------
    xls_obj = pd.ExcelFile(xls_file)
    if "Transformer Parameters" in xls_obj.sheet_names:
        df_trafo = pd.read_excel(xls_file, sheet_name="Transformer Parameters", index_col=0)
        for _, row in df_trafo.iterrows():
            pp.create_transformer_from_parameters(net,
                 hv_bus    = row["hv_bus"],
                 lv_bus    = row["lv_bus"],
                 sn_mva    = row["sn_mva"],
                 vn_hv_kv  = row["vn_hv_kv"],
                 vn_lv_kv  = row["vn_lv_kv"],
                 vk_percent= row["vk_percent"],
                 vkr_percent=row["vkr_percent"],
                 pfe_kw    = row["pfe_kw"],
                 i0_percent= row["i0_percent"],
                 in_service=row["in_service"],
                 max_loading_percent=row["max_loading_percent"])

    # --- 4. Dynamic-profile helpers ----------------------------------------------
    df_load_profile = pd.read_excel(xls_file, sheet_name="Load Profile")
    df_load_profile.columns = df_load_profile.columns.str.strip()

    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus  = int(m.group(1))
            qcol = f"q_mvar_bus_{bus}"
            if qcol in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": qcol}

    df_gen_profile = pd.read_excel(xls_file, sheet_name="Generator Profile")
    df_gen_profile.columns = df_gen_profile.columns.str.strip()

    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            nums = re.findall(r"\d+", col)
            if nums:
                gen_dynamic[int(nums[-1])] = col

    num_hours = len(df_load_profile)

    # --- 5. Return exactly what Colab did -----------------------------------------
    if "Transformer Parameters" in xls_obj.sheet_names:
        return (net, df_bus, df_slack, df_line,
                num_hours, load_dynamic, gen_dynamic,
                df_load_profile, df_gen_profile, df_trafo)

    return (net, df_bus, df_slack, df_line,
            num_hours, load_dynamic, gen_dynamic,
            df_load_profile, df_gen_profile)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Colab-equivalent: rebuild + 24-h OPF loop on every call
# ---------------------------------------------------------------------------
def calculating_hourly_cost(xls_file):
    """
    Streamlit drop-in replacement for the Colab routine.

    Parameters
    ----------
    xls_file : BytesIO | str
        The file object returned by `st.file_uploader` *or* a filesystem path.

    Returns
    -------
    list
        Length == number of rows in “Load Profile”.
        Each element is net.res_cost (DataFrame) when OPF succeeded,
        otherwise the integer 0.
    """
    # 0) Prep
    xls = pd.ExcelFile(xls_file)
    hourly_cost_list = []
    net = pp.create_empty_network()

    # 1) Static sheets ----------------------------------------------------------
    df_bus   = pd.read_excel(xls_file, sheet_name="Bus Parameters",      index_col=0)
    df_load  = pd.read_excel(xls_file, sheet_name="Load Parameters",     index_col=0)
    df_slack = pd.read_excel(xls_file, sheet_name="Generator Parameters",index_col=0)
    df_line  = pd.read_excel(xls_file, sheet_name="Line Parameters",     index_col=0)

    # 2) Create buses -----------------------------------------------------------
    for _, row in df_bus.iterrows():
        pp.create_bus(net,
                      name        = row["name"],
                      vn_kv       = row["vn_kv"],
                      zone        = row["zone"],
                      in_service  = row["in_service"],
                      max_vm_pu   = row["max_vm_pu"],
                      min_vm_pu   = row["min_vm_pu"])

    # 3) Create loads -----------------------------------------------------------
    for _, row in df_load.iterrows():
        pp.create_load(net,
                       bus        = row["bus"],
                       p_mw       = row["p_mw"],
                       q_mvar     = row["q_mvar"],
                       in_service = row["in_service"])

    # 4) Create generators / ext-grid ------------------------------------------
    for _, row in df_slack.iterrows():
        if row["slack_weight"] == 1:
            ext_idx = pp.create_ext_grid(net,
                                         bus       = row["bus"],
                                         vm_pu     = row["vm_pu"],
                                         va_degree = 0)
            pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                                cp0_eur_per_mw   = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw   = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw   = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])
        else:
            gen_idx = pp.create_gen(net,
                                    bus          = row["bus"],
                                    p_mw         = row["p_mw"],     # overwritten hourly
                                    vm_pu        = row["vm_pu"],
                                    min_q_mvar   = row["min_q_mvar"],
                                    max_q_mvar   = row["max_q_mvar"],
                                    scaling      = row["scaling"],
                                    in_service   = row["in_service"],
                                    slack_weight = row["slack_weight"],
                                    controllable = row["controllable"],
                                    max_p_mw     = row["max_p_mw"],
                                    min_p_mw     = row["min_p_mw"])
            pp.create_poly_cost(net, element=gen_idx, et="gen",
                                cp0_eur_per_mw   = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw   = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw   = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])

    # 5) Create lines -----------------------------------------------------------
    for _, row in df_line.iterrows():
        if pd.isna(row["parallel"]):
            continue
        geodata = ast.literal_eval(row["geodata"]) if isinstance(row["geodata"], str) else row["geodata"]
        pp.create_line_from_parameters(net,
                                       from_bus            = row["from_bus"],
                                       to_bus              = row["to_bus"],
                                       length_km           = row["length_km"],
                                       r_ohm_per_km        = row["r_ohm_per_km"],
                                       x_ohm_per_km        = row["x_ohm_per_km"],
                                       c_nf_per_km         = row["c_nf_per_km"],
                                       max_i_ka            = row["max_i_ka"],
                                       in_service          = row["in_service"],
                                       max_loading_percent = row["max_loading_percent"],
                                       geodata             = geodata)

    # 6) Optional transformers ---------------------------------------------------
    if "Transformer Parameters" in xls.sheet_names:
        df_trafo = pd.read_excel(xls_file, sheet_name="Transformer Parameters", index_col=0)
        for _, row in df_trafo.iterrows():
            pp.create_transformer_from_parameters(net,
                 hv_bus             = row["hv_bus"],
                 lv_bus             = row["lv_bus"],
                 sn_mva             = row["sn_mva"],
                 vn_hv_kv           = row["vn_hv_kv"],
                 vn_lv_kv           = row["vn_lv_kv"],
                 vk_percent         = row["vk_percent"],
                 vkr_percent        = row["vkr_percent"],
                 pfe_kw             = row["pfe_kw"],
                 i0_percent         = row["i0_percent"],
                 in_service         = row["in_service"],
                 max_loading_percent= row["max_loading_percent"])

    # 7) Dynamic-profile helpers -------------------------------------------------
    df_load_profile = pd.read_excel(xls_file, sheet_name="Load Profile")
    df_load_profile.columns = df_load_profile.columns.str.strip()

    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus   = int(m.group(1))
            q_col = f"q_mvar_bus_{bus}"
            if q_col in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": q_col}

    df_gen_profile = pd.read_excel(xls_file, sheet_name="Generator Profile")
    df_gen_profile.columns = df_gen_profile.columns.str.strip()

    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            nums = re.findall(r"\d+", col)
            if nums:
                gen_dynamic[int(nums[-1])] = col

    num_hours = len(df_load_profile)

    # 8) Hour-by-hour OPF loop ---------------------------------------------------
    for hour in range(num_hours):

        # 8.1 update loads
        for bus_id, cols in load_dynamic.items():
            p_val = float(df_load_profile.at[hour, cols["p"]])
            q_val = float(df_load_profile.at[hour, cols["q"]])
            mask  = net.load.bus == bus_id
            net.load.loc[mask, "p_mw"]  = p_val
            net.load.loc[mask, "q_mvar"] = q_val

        # 8.2 update gens / ext grid
        for bus_id, col in gen_dynamic.items():
            p_val = float(df_gen_profile.at[hour, col])
            if bus_id in net.ext_grid.bus.values:
                net.ext_grid.loc[net.ext_grid.bus == bus_id, "p_mw"] = p_val
            else:
                net.gen.loc[net.gen.bus == bus_id, "p_mw"] = p_val

        # 8.3 run OPF
        try:
            pp.runopp(net)
            hourly_cost_list.append(net.res_cost)
        except Exception:
            hourly_cost_list.append(0)
            continue

    return hourly_cost_list

# ------------------------------------------------------------------
# 1)  all_real_numbers  — identical to your Colab version
# ------------------------------------------------------------------
def all_real_numbers(lst):
    invalid_count = 0
    for x in lst:
        # 1) Not numeric
        if not isinstance(x, (int, float)):
            invalid_count += 1
        # 2) NaN or infinite
        elif not math.isfinite(x):
            invalid_count += 1

    if invalid_count > len(line_outages):
        return False
    return True


# ------------------------------------------------------------------
# 2)  overloaded_lines  — identical to your Colab version
# ------------------------------------------------------------------
def overloaded_lines(net):
    overloaded = []
    # turn loading_percent Series into a list once
    loadings = transform_loading(net.res_line["loading_percent"])
    real_check = all_real_numbers(net.res_line["loading_percent"].tolist())

    for idx, (res, loading_val) in enumerate(zip(net.res_line.itertuples(), loadings)):
        # grab this line’s own max
        own_max = net.line.at[idx, "max_loading_percent"]
        # print(f"max loading capacity @ id {id} is {own_max}.")

        if not real_check:
            # any NaN/non-numeric or at-limit is overloaded
            if not isinstance(loading_val, (int, float)) or math.isnan(loading_val) or loading_val >= own_max:
                overloaded.append(idx)
        else:
            # only truly > its own max
            if loading_val is not None and not (isinstance(loading_val, float) and math.isnan(loading_val)) and loading_val > own_max:
                overloaded.append(idx)
    return overloaded

# -------------------------------------------------------------
# 1)  Does the (from_bus, to_bus) pair correspond to a trafo?
# -------------------------------------------------------------
def check_bus_pair(xls_file, bus_pair):
    """
    Parameters
    ----------
    xls_file : BytesIO | str
        Same object you get from st.file_uploader – or a path.
    bus_pair : tuple[int, int]
        (from_bus, to_bus) to look up.

    Returns
    -------
    True   → pair matches a transformer
    False  → pair matches a line
    None   → no match found
    """
    xls = pd.ExcelFile(xls_file)

    if "Transformer Parameters" in xls.sheet_names:
        transformer_df = pd.read_excel(xls_file, sheet_name="Transformer Parameters")
        line_df        = pd.read_excel(xls_file, sheet_name="Line Parameters")

        from_bus, to_bus = bus_pair

        transformer_match = (
            ((transformer_df["hv_bus"] == from_bus) & (transformer_df["lv_bus"] == to_bus)) |
            ((transformer_df["hv_bus"] == to_bus) & (transformer_df["lv_bus"] == from_bus))
        ).any()

        line_match = (
            ((line_df["from_bus"] == from_bus) & (line_df["to_bus"] == to_bus)) |
            ((line_df["from_bus"] == to_bus)  & (line_df["to_bus"] == from_bus))
        ).any()

        if transformer_match:
            return True
        if line_match:
            return False

    # nothing matched
    return None
# -------------------------------------------------------------


# -------------------------------------------------------------
# 2)  Normalise “loading_percent” fields so units are consistent
# -------------------------------------------------------------
def transform_loading(a):
    """
    Multiplies every value < 2.5 by 100 so that fractional %
    values (e.g. 0.95) become full percentages (95.0).
    Works for scalars or lists.  Returns the same “shape” back.
    """
    if a is None:
        return a

    # turn scalars into a list for uniform processing
    is_single = False
    if isinstance(a, (int, float)):
        a         = [a]
        is_single = True

    # decide whether conversion is needed
    flag = True
    for item in a:
        if isinstance(item, (int, float)) and item >= 2.5:
            flag = False

    if flag:
        a = [item * 100 if isinstance(item, (int, float)) else item for item in a]

    return a[0] if is_single else a
# -------------------------------------------------------------

# -------------------------------------------------------------
# 5)  Identify overloaded transformers (if any exist)
# -------------------------------------------------------------
def overloaded_transformer(net, xls_file, line_outages):
    """
    Same logic as Colab version but *xls_file* is explicit.
    Returns list of transformer indices exceeding their max loading.
    """
    overloaded = []

    xls = pd.ExcelFile(xls_file)
    if "Transformer Parameters" not in xls.sheet_names:
        return overloaded

    loadings   = transform_loading(net.res_trafo["loading_percent"])
    # real_check = all_real_numbers(net.res_trafo["loading_percent"].tolist(),
    #                               line_outages)
    real_check = all_real_numbers(net.res_trafo["loading_percent"].tolist())


    for idx, (_, loading_val) in enumerate(zip(net.res_trafo.itertuples(),
                                               loadings)):
        own_max = net.trafo.at[idx, "max_loading_percent"]

        if not real_check:
            if (loading_val is not None and
                not (isinstance(loading_val, float) and math.isnan(loading_val)) and
                loading_val >= own_max):
                overloaded.append(idx)
        else:
            if loading_val > own_max:
                overloaded.append(idx)
    return overloaded
# -------------------------------------------------------------
# ------------------------------------------------------------------
# Ensure `path` is available everywhere once the user has uploaded
# their Excel file on Page-1.
# ------------------------------------------------------------------
path = st.session_state.get("uploaded_file")   # BytesIO object


# ------------------------------------------------------------------
# generate_line_outages  – identical logic, NO print statements
# ------------------------------------------------------------------
# def generate_line_outages(outage_hours, line_down, risk_scores,
#                           capped_contingency_mode=False):
#     if not outage_hours or not line_down or not risk_scores:
#         return []

#     sheet_name = "Line Parameters"
#     df_line = pd.read_excel(path, sheet_name=sheet_name)
#     no_of_lines_in_network = len(df_line) - 1
#     capped_limit = math.floor(0.2 * no_of_lines_in_network)

#     # Combine line, outage_hour, and risk into tuples
#     combined = [
#         (line[0], line[1], hour, risk)
#         for line, hour, risk in zip(line_down, outage_hours, risk_scores)
#     ]

#     # Sort by risk score in descending order
#     sorted_combined = sorted(combined, key=lambda x: x[-1], reverse=True)

#     # Extract only (from_bus, to_bus, outage_hour)
#     line_outages = [(line[0], line[1], line[2]) for line in sorted_combined]

#     if capped_contingency_mode and len(line_outages) > capped_limit:
#         line_outages = line_outages[:capped_limit]

#     return line_outages

def generate_line_outages(outage_hours, line_down, risk_scores,
                          capped_contingency_mode=False):

    if not outage_hours or not line_down or not risk_scores:
        return []

    # ── 1  clean & align the risk-score list ───────────────────────────────
    needed = len(line_down)
    # keep only the numeric entries (positions 0,2,4,…) and take the first
    # *needed* of them.  This restores the Colab behaviour.
    cleaned  = [r for r in risk_scores if isinstance(r, (int, float))][:needed]

    for r in risk_scores:
        # convert dict → its numeric score
        if isinstance(r, dict):
            r = r.get("risk_score", 0)

        # accept only numeric values
        if isinstance(r, (int, float)):
            cleaned.append(r)

        if len(cleaned) == needed:      # stop when we have enough
            break

    # if still short, pad with zeros (shouldn't normally happen)
    cleaned += [0] * (needed - len(cleaned))
    # ----------------------------------------------------------------------

    # ── 2  build the tuples (from_bus, to_bus, hour, score) ────────────────
    combined = [
        (line[0], line[1], hour, score)
        for line, hour, score in zip(line_down, outage_hours, cleaned)
    ]

    # ── 3  sort by risk, apply contingency cap, return (fbus, tbus, hour) ──
    combined.sort(key=lambda x: x[-1], reverse=True)

    sheet_name = "Line Parameters"
    df_line = pd.read_excel(path, sheet_name=sheet_name)
    capped_limit = math.floor(0.2 * (len(df_line) - 1))

    if capped_contingency_mode:
        if len(combined) > capped_limit:
            combined = combined[:capped_limit]

    return [(f, t, hr) for f, t, hr, _ in combined]



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

    # Create columns for dropdown menus
    col1, col2, col3 = st.columns(3)

    with col1:
        # Risk Tolerance dropdown
        intensity_options = ["Low", "Medium", "High"]
        intensity = st.selectbox(
            "Risk Tolerance",
            options=intensity_options,
            help="Low: Temperature > 35°C, Precipitation > 50mm, Wind > 10m/s, Medium: Temperature > 38°C, Precipitation > 100mm, Wind > 15m/s, High: Temperature > 41°C, Precipitation > 150mm, Wind > 20m/s"
        )

    with col2:
        # Study Period dropdown
        period_options = ["Weekly", "Monthly"]
        study_period = st.selectbox(
            "Study Period",
            options=period_options,
            help="Weekly: Daily aggregated data, Monthly: Monthly aggregated data"
        )

    with col3:
        # Risk Score Threshold slider
        risk_score = st.slider(
            "Risk Score Threshold",
            min_value=6,
            max_value=18,
            value=14,
            help="Higher threshold means higher risk tolerance. Range: 6-18"
        )

    # Check if network data is available
    if "network_data" not in st.session_state or st.session_state.network_data is None:
        st.warning("Please upload and initialize network data on the Network Initialization page first.")
    else:
        # Button to process and show results
        if st.button("Process Weather Risk Data"):
            with st.spinner("Processing weather risk data (Estimated Time 5-15 minutes)..."):
                try:
                    # Initialize Earth Engine if not already done
                    try:
                        initialize_ee()
                    except Exception as e:
                        st.error(f"Error initializing Earth Engine: {str(e)}")

                    # Process temperature and generate results
                    def process_temperature(intensity, time_period, risk_score_threshold, df_line):
                        # Temperature thresholds for intensity levels
                        thresholds = {"Low": 35, "Medium": 38, "High": 41}
                        thresholds_p = {"Low": 50, "Medium": 100, "High": 150}
                        thresholds_w = {"Low": 10, "Medium": 15, "High": 20}

                        if intensity not in thresholds or time_period not in ["Monthly", "Weekly"]:
                            raise ValueError("Invalid intensity or time period")

                        # Use the transmission line data from session state
                        df = df_line.copy()

                        from_buses = df["from_bus"].tolist()
                        to_buses = df["to_bus"].tolist()
                        all_lines = list(df[["from_bus", "to_bus"]].itertuples(index=False, name=None))

                        df["geodata"] = df["geodata"].apply(lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x)
                        line_geometries = [LineString(coords) for coords in df["geodata"]]
                        gdf = gpd.GeoDataFrame(df, geometry=line_geometries, crs="EPSG:4326")

                        # Create Folium map (instead of geemap.Map)
                        m = folium.Map(location=[30, 70], zoom_start=5, width=800, height=600)

                        # Define date range (last 10 years)
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365 * 10)

                        # Select dataset based on time period
                        dataset_name = "ECMWF/ERA5/MONTHLY" if time_period == "Monthly" else "ECMWF/ERA5_LAND/DAILY_AGGR"
                        dataset = ee.ImageCollection(dataset_name).filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                        dataset_forecast = ee.ImageCollection("NOAA/GFS0P25")
                        d = dataset_forecast.first()

                        # Create land mask
                        land_mask = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                        land_mask = land_mask.map(lambda feature: feature.set("dummy", 1))
                        land_image = land_mask.reduceToImage(["dummy"], ee.Reducer.first()).gt(0)

                        # Select the correct band
                        temp_band = "temperature_2m" if time_period == "Weekly" else "mean_2m_air_temperature"
                        precip_band = "total_precipitation_sum" if time_period == "Weekly" else "total_precipitation"
                        u_wind_band = "u_component_of_wind_10m" if time_period == "Weekly" else "u_component_of_wind_10m"
                        v_wind_band = "v_component_of_wind_10m" if time_period == "Weekly" else "v_component_of_wind_10m"

                        temp_forecast = "temperature_2m_above_ground"
                        u_forecast = "u_component_of_wind_10m_above_ground"
                        v_forecast = "v_component_of_wind_10m_above_ground"
                        precip_forecast = "precipitation_rate"

                        # Ensure dataset contains required bands
                        first_img = dataset.first()
                        band_names = first_img.bandNames().getInfo()
                        required_bands = [temp_band, precip_band, u_wind_band, v_wind_band]
                        for band in required_bands:
                            if band not in band_names:
                                raise ValueError(f"Dataset does not contain band: {band}. Available bands: {band_names}")

                        # Convert temperature from Kelvin to Celsius and filter occurrences above threshold
                        dataset1 = dataset.map(lambda img: img.select(temp_band).subtract(273.15).rename("temp_C"))
                        filtered_dataset1 = dataset1.map(lambda img: img.gt(thresholds[intensity]))

                        dataset2 = dataset.map(lambda img: img.select(precip_band).multiply(1000).rename("preci_mm"))
                        filtered_dataset2 = dataset2.map(lambda img: img.gt(thresholds_p[intensity]))

                        dataset3 = dataset.map(lambda img: img.select(u_wind_band).rename("u_wind"))
                        dataset4 = dataset.map(lambda img: img.select(v_wind_band).rename("v_wind"))

                        wind_magnitude = dataset.map(lambda img: img.expression(
                            "sqrt(pow(u, 2) + pow(v, 2))",
                            {
                                "u": img.select(u_wind_band),
                                "v": img.select(v_wind_band)
                            }
                        ).rename("wind_magnitude"))

                        filtered_wind = wind_magnitude.map(lambda img: img.gt(thresholds_w[intensity]))

                        # Sum occurrences where thresholds were exceeded
                        occurrence_count_t = filtered_dataset1.sum()
                        occurrence_count_p = filtered_dataset2.sum()
                        occurrence_count_w = filtered_wind.sum()

                        # Convert transmission lines to FeatureCollection
                        features = [
                            ee.Feature(ee.Geometry.LineString(row["geodata"]), {
                                "line_id": i,
                                "geodata": str(row["geodata"])
                            }) for i, row in df.iterrows()
                        ]
                        line_fc = ee.FeatureCollection(features)

                        bounding_box = line_fc.geometry().bounds()

                        masked_occurrences_t = occurrence_count_t.clip(bounding_box)
                        masked_occurrences_p = occurrence_count_p.clip(bounding_box)
                        masked_occurrences_w = occurrence_count_w.clip(bounding_box)

                        masked_occurrences_t = masked_occurrences_t.updateMask(land_image)
                        masked_occurrences_p = masked_occurrences_p.updateMask(land_image)
                        masked_occurrences_w = masked_occurrences_w.updateMask(land_image)

                        # Computing occurrence statistics
                        stats_t = masked_occurrences_t.reduceRegion(
                            reducer=ee.Reducer.max(),
                            geometry=bounding_box,
                            scale=1000,
                            bestEffort=True,
                            maxPixels=1e13
                        )

                        stats_p = masked_occurrences_p.reduceRegion(
                            reducer=ee.Reducer.max(),
                            geometry=bounding_box,
                            scale=1000,
                            bestEffort=True,
                            maxPixels=1e13
                        )

                        stats_w = masked_occurrences_w.reduceRegion(
                            reducer=ee.Reducer.max(),
                            geometry=bounding_box,
                            scale=1000,
                            bestEffort=True,
                            maxPixels=1e13
                        )

                        stats_dict_t = stats_t.getInfo()
                        stats_dict_p = stats_p.getInfo()
                        stats_dict_w = stats_w.getInfo()

                        if stats_dict_t:
                            max_occurrence_key_t = list(stats_dict_t.keys())[0]
                            max_occurrence_t = ee.Number(stats_t.get(max_occurrence_key_t)).getInfo()
                        else:
                            max_occurrence_t = 1

                        if stats_dict_p:
                            max_occurrence_key_p = list(stats_dict_p.keys())[0]
                            max_occurrence_p = ee.Number(stats_p.get(max_occurrence_key_p)).getInfo()
                        else:
                            max_occurrence_p = 1

                        if stats_dict_w:
                            max_occurrence_key_w = list(stats_dict_w.keys())[0]
                            max_occurrence_w = ee.Number(stats_w.get(max_occurrence_key_w)).getInfo()
                        else:
                            max_occurrence_w = 1

                        mid1_t = max_occurrence_t / 3
                        mid2_t = 2 * (max_occurrence_t / 3)

                        mid1_p = max_occurrence_p / 3
                        mid2_p = 2 * (max_occurrence_p / 3)

                        mid1_w = max_occurrence_w / 3
                        mid2_w = 2 * (max_occurrence_w / 3)

                        mid1_ft = thresholds[intensity] * (1 - 10/100)
                        mid2_ft = thresholds[intensity] * (1 - 20/100)

                        mid1_fw = thresholds_w[intensity] * (1 - 10/100)
                        mid2_fw = thresholds_w[intensity] * (1 - 20/100)

                        # Get current time and forecast time
                        now = datetime.utcnow()
                        nearest_gfs_time = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
                        future = nearest_gfs_time + timedelta(hours=24)
                        now_str = nearest_gfs_time.strftime('%Y-%m-%dT%H:%M:%S')
                        future_str = future.strftime('%Y-%m-%dT%H:%M:%S')

                        latest_image = dataset_forecast.sort("system:time_start", False).first()
                        latest_timestamp = latest_image.date().format().getInfo()

                        # Classify occurrences
                        classified_occurrences_t = masked_occurrences_t.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": masked_occurrences_t,
                                "mid1": mid1_t,
                                "mid2": mid2_t,
                            }
                        )

                        classified_occurrences_w = masked_occurrences_w.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": masked_occurrences_w,
                                "mid1": mid1_w,
                                "mid2": mid2_w,
                            }
                        )

                        classified_occurrences_p = masked_occurrences_p.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": masked_occurrences_p,
                                "mid1": mid1_p,
                                "mid2": mid2_p,
                            }
                        )

                        classified_viz = {
                            'min': 1,
                            'max': 3,
                            'palette': ['green', 'yellow', 'red']
                        }

                        classified_t = classified_occurrences_t.clip(bounding_box)
                        classified_p = classified_occurrences_p.clip(bounding_box)
                        classified_w = classified_occurrences_w.clip(bounding_box)

                        classified_t = classified_t.updateMask(land_image)
                        classified_p = classified_p.updateMask(land_image)
                        classified_w = classified_w.updateMask(land_image)

                        combined_layer = classified_t.add(classified_p).add(classified_w)
                        combined_layer = combined_layer.clip(bounding_box)
                        combined_layer = combined_layer.updateMask(land_image)

                        vis_params = {
                            'min': 3,
                            'max': 9,
                            'palette': ['lightgreen', 'green', 'yellow', 'orange', 'red', 'crimson', 'darkred']
                        }

                        combined_viz = {
                            'min': 6,
                            'max': 18,
                            'palette': [
                                '#32CD32',  # 6 - Lime Green
                                '#50C878',  # 7 - Medium Sea Green
                                '#66CC66',  # 8 - Soft Green
                                '#B2B200',  # 9 - Olive Green
                                '#CCCC00',  # 10 - Yellow-Green
                                '#E6B800',  # 11 - Mustard Yellow
                                '#FFD700',  # 12 - Golden Yellow
                                '#FFCC00',  # 13 - Deep Yellow
                                '#FFA500',  # 14 - Orange
                                '#FF9933',  # 15 - Dark Orange
                                '#FF6600',  # 16 - Reddish Orange
                                '#FF0000'   # 18 - Red
                            ]
                        }

                        # Add weather layers
                        m.add_ee_layer(classified_t, classified_viz, f"Temperature Occurrence Classification ({time_period})")
                        m.add_ee_layer(classified_p, classified_viz, f"Precipitation Occurrence Classification ({time_period})")
                        m.add_ee_layer(classified_w, classified_viz, f"Wind Occurrence Classification ({time_period})")
                        m.add_ee_layer(combined_layer, vis_params, "Combined Historic Classification")

                        fut = [latest_timestamp]
                        daily_dfs = {}
                        results_per_day = []
                        max_times = []
                        risk_scores = []  # Add this line to initialize risk_scores

                        # Process forecast for next 24 hours
                        future = nearest_gfs_time + timedelta(hours=24)
                        future_str = future.strftime('%Y-%m-%dT%H:%M:%S')
                        fut.append(future_str)

                        forecast_24h = dataset_forecast.filterDate(latest_timestamp, future_str)

                        forecast_temp = forecast_24h.select(temp_forecast).max().rename("forecast_temp_C_day_1")
                        forecast_u = forecast_24h.select(u_forecast).max().rename("forecast_u_day_1")
                        forecast_v = forecast_24h.select(v_forecast).max().rename("forecast_v_day_1")
                        forecast_pre = forecast_24h.select(precip_forecast).max().multiply(86400).rename("forecast_prec_day_1")

                        forecast_wind_magnitude = forecast_u.expression(
                            "sqrt(pow(u, 2) + pow(v, 2))",
                            {"u": forecast_u, "v": forecast_v}
                        ).rename("forecast_wind_magnitude_day_1")

                        mid1_ft = thresholds[intensity] * 0.90
                        mid2_ft = thresholds[intensity] * 0.80
                        mid1_fw = thresholds_w[intensity] * 0.90
                        mid2_fw = thresholds_w[intensity] * 0.80
                        mid1_fp = thresholds_p[intensity] * 0.90
                        mid2_fp = thresholds_p[intensity] * 0.80

                        classified_forecast_t = forecast_temp.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": forecast_temp,
                                "mid1": mid1_ft,
                                "mid2": mid2_ft
                            }
                        ).clip(bounding_box)

                        classified_forecast_w = forecast_wind_magnitude.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": forecast_wind_magnitude,
                                "mid1": mid1_fw,
                                "mid2": mid2_fw
                            }
                        ).clip(bounding_box)

                        classified_forecast_p = forecast_pre.expression(
                            "(b(0) <= mid1) ? 1 : (b(0) <= mid2) ? 2 : 3",
                            {
                                "b(0)": forecast_pre,
                                "mid1": mid1_fp,
                                "mid2": mid2_fp
                            }
                        ).clip(bounding_box)

                        combined_forecast = classified_forecast_t.add(classified_forecast_w).add(classified_forecast_p).clip(bounding_box)
                        combined_forecast = combined_forecast.add(combined_layer)
                        combined_forecast = combined_forecast.updateMask(land_image)

                        # Add combined forecast layer
                        m.add_ee_layer(combined_forecast, combined_viz, "Day Ahead - Risk Score")

                        # Add transmission lines last with thicker styling
                        m.add_ee_layer(line_fc.style(**{'color': 'blue', 'width': 4}), {}, "Transmission Lines")

                        # Add layer control
                        folium.LayerControl(collapsed=False).add_to(m)

                        # Reduce regions to get risk scores per line
                        reduced = combined_forecast.reduceRegions(
                            collection=line_fc,
                            reducer=ee.Reducer.max(),
                            scale=1000
                        )

                        results = reduced.getInfo()

                        data = []
                        daily_results = []
                        risk_scores = []

                        for feature in results["features"]:
                            props = feature["properties"]
                            line_id = props["line_id"]
                            max_risk = props.get("max", 0)
                            from_bus = df.loc[line_id, "from_bus"]
                            to_bus = df.loc[line_id, "to_bus"]
                            daily_results.append((int(from_bus), int(to_bus), int(max_risk)))
                            risk_scores.append(int(max_risk))  # Add this line to collect risk scores
                            # if max_risk >= risk_score_threshold:
                            #     risk_scores.append(int(max_risk))
                            
                    
                            data.append({
                                "line_id": props["line_id"],
                                "from_bus": int(from_bus),
                                "to_bus": int(to_bus),
                                "risk_score": int(max_risk)
                            })
                    
                            risk_scores.append({
                                "line_id": int(line_id),
                                "from_bus": int(from_bus),
                                "to_bus": int(to_bus),
                                "risk_score": int(max_risk)
                            })
                        results_per_day.append(daily_results)
                        daily_dfs["Day_1"] = pd.DataFrame(data)

                        # Filter lines with risk score >= threshold
                        day_1_results = results_per_day[0]
                        filtered_lines_day1 = [(from_bus, to_bus) for from_bus, to_bus, score in day_1_results if score >= risk_score_threshold]
                        length_lines = len(filtered_lines_day1)
                        outage_hour_day = [random.randint(11, 15) for _ in range(length_lines)]

                        # Create structured output for lines and outage hours
                        line_outages = [{"from_bus": from_bus, "to_bus": to_bus} for from_bus, to_bus in filtered_lines_day1]
                        outage_data = [{"line": f"From Bus {line[0]} to Bus {line[1]}", "outage_hours": hours, "risk_score": score}
                                      for line, hours, score in zip(filtered_lines_day1, outage_hour_day, [score for _, _, score in day_1_results if score >= risk_score_threshold])]

                        # Store in a format that can be used by other pages
                        line_outage_data = {
                            "lines": filtered_lines_day1,
                            "hours": outage_hour_day,
                            "risk_scores": risk_scores
                        }

                        return m, daily_dfs["Day_1"], line_outage_data, outage_data, max_occurrence_t, max_occurrence_p, max_occurrence_w, risk_scores  # Update this line

                    # Call the function with selected parameters
                    weather_map, risk_df, line_outage_data, outage_data, max_occurrence_t, max_occurrence_p, max_occurrence_w, risk_scores = process_temperature(
                    intensity,
                    study_period,
                    risk_score,
                    st.session_state.network_data['df_line']
                    )
                    # Store the map and data in session state
                    st.session_state.weather_map_obj = weather_map
                    st.session_state.line_outage_data = line_outage_data
                    st.session_state["outage_hours"] = line_outage_data["hours"]
                    st.session_state["line_down"]    = line_outage_data["lines"]
                    st.session_state["risk_scores"]  = line_outage_data["risk_scores"]
                    st.session_state.risk_df = risk_df
                    st.session_state.outage_data = outage_data
                    st.session_state.risk_score = risk_score
                    st.session_state.max_occurrences = {
                        "temperature": max_occurrence_t,
                        "precipitation": max_occurrence_p,
                        "wind": max_occurrence_w
                    }

                    # Display the results
                    st.subheader("Day Ahead Risk Assessment")

                    # Display the map
                    st.write("### Geographic Risk Visualization")
                    if st.session_state.weather_map_obj:
                        st_folium(st.session_state.weather_map_obj, width=800, height=600, key="weather_map")

                    # Display legends
                    st.write("### Risk Visualization Legends")
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Patch

                    # Define legend data
                    final_risk_score = {
                        "title": "Final Risk Score (6-18)",
                        "colors": [('#32CD32', '6'), ('#50C878', '7'), ('#66CC66', '8'), ('#B2B200', '9'),
                                   ('#CCCC00', '10'), ('#E6B800', '11'), ('#FFD700', '12'), ('#FFCC00', '13'),
                                   ('#FFA500', '14'), ('#FF9933', '15'), ('#FF6600', '16'), ('#FF0000', '18')]
                    }

                    historical_classification = {
                        "title": "Historical Risk Classification (1-3)",
                        "colors": [('green', '1'), ('yellow', '2'), ('red', '3')]
                    }

                    historical_score = {
                        "title": "Historical Risk Score (3-9)",
                        "colors": [('lightgreen', '3'), ('green', '4'), ('yellow', '5'), ('orange', '6'),
                                   ('red', '7'), ('crimson', '8'), ('darkred', '9')]
                    }

                    # Create figure and grid layout
                    fig = plt.figure(figsize=(5, 3))
                    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.8])

                    # Final Risk Score - vertical on left
                    ax1 = fig.add_subplot(gs[:, 0])
                    ax1.axis('off')
                    ax1.set_title(final_risk_score["title"], fontsize=9, fontweight='bold', color='white', loc='left')
                    handles1 = [Patch(color=color, label=label) for color, label in final_risk_score["colors"]]
                    ax1.legend(handles=handles1, loc='center left', frameon=False,
                               handleheight=1.2, handlelength=8, fontsize=9, labelcolor='white')

                    # Historical Risk Classification - top right
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.axis('off')
                    ax2.set_title(historical_classification["title"], fontsize=9, fontweight='bold', color='white')
                    handles2 = [Patch(color=color, label=label) for color, label in historical_classification["colors"]]
                    ax2.legend(handles=handles2, loc='center', ncol=3, frameon=False,
                               handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                    # Historical Risk Score - bottom right
                    ax3 = fig.add_subplot(gs[1, 1])
                    ax3.axis('off')
                    ax3.set_title(historical_score["title"], fontsize=9, fontweight='bold', color='white')
                    handles3 = [Patch(color=color, label=label) for color, label in historical_score["colors"]]
                    ax3.legend(handles=handles3, loc='center', ncol=3, frameon=False,
                               handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                    fig.patch.set_facecolor('black')
                    plt.tight_layout(pad=1)
                    st.pyplot(fig)

                    # Display risk scores for all lines
                    st.write("### Risk Scores for All Transmission Lines")
                    risk_df_display = risk_df[["line_id", "from_bus", "to_bus", "risk_score"]].sort_values(by="risk_score", ascending=False)
                    risk_df_display.columns = ["Line ID", "From Bus", "To Bus", "Risk Score"]
                    st.dataframe(risk_df_display, use_container_width=True)

                    # Display lines expected to face outage based on threshold
                    if outage_data:
                        st.write(f"### Lines Expected to Face Outage (Risk Score ≥ {risk_score})")
                        outage_df = pd.DataFrame(outage_data)
                        outage_df.columns = ["Transmission Line", "Expected Outage Hours", "Risk Score"]
                        st.dataframe(outage_df, use_container_width=True)

                        # Summary statistics
                        st.write("### Outage Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Number of Lines at Risk", len(outage_data))
                        with col2:
                            st.metric("Max Temperature Occurrences", int(max_occurrence_t))
                        with col3:
                            st.metric("Max Precipitation Occurrences", int(max_occurrence_p))
                        with col4:
                            st.metric("Max Wind Occurrences", int(max_occurrence_w))
                    else:
                        st.success(f"No transmission lines are expected to face outage at the selected risk threshold ({risk_score}).")
                        # Still display max occurrences even if no outages
                        st.write("### Historical Max Occurrences")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Max Temperature Occurrences", int(max_occurrence_t))
                        with col2:
                            st.metric("Max Precipitation Occurrences", int(max_occurrence_p))
                        with col3:
                            st.metric("Max Wind Occurrences", int(max_occurrence_w))

                except Exception as e:
                    st.error(f"Error processing weather risk data: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        else:
            # Display cached results if available
            if st.session_state.weather_map_obj and "risk_df" in st.session_state and "outage_data" in st.session_state:
                st.subheader("Day Ahead Risk Assessment")

                # Display the map
                st.write("### Geographic Risk Visualization")
                st_folium(st.session_state.weather_map_obj, width=800, height=600, key="weather_map_cached")

                # Display legends
                st.write("### Risk Visualization Legends")
                import matplotlib.pyplot as plt
                from matplotlib.patches import Patch

                # Define legend data
                final_risk_score = {
                    "title": "Final Risk Score (6-18)",
                    "colors": [('#32CD32', '6'), ('#50C878', '7'), ('#66CC66', '8'), ('#B2B200', '9'),
                               ('#CCCC00', '10'), ('#E6B800', '11'), ('#FFD700', '12'), ('#FFCC00', '13'),
                               ('#FFA500', '14'), ('#FF9933', '15'), ('#FF6600', '16'), ('#FF0000', '18')]
                }

                historical_classification = {
                    "title": "Historical Risk Classification (1-3)",
                    "colors": [('green', '1'), ('yellow', '2'), ('red', '3')]
                }

                historical_score = {
                    "title": "Historical Risk Score (3-9)",
                    "colors": [('lightgreen', '3'), ('green', '4'), ('yellow', '5'), ('orange', '6'),
                               ('red', '7'), ('crimson', '8'), ('darkred', '9')]
                }

                # Create figure and grid layout
                fig = plt.figure(figsize=(5, 3))
                gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.8])

                # Final Risk Score - vertical on left
                ax1 = fig.add_subplot(gs[:, 0])
                ax1.axis('off')
                ax1.set_title(final_risk_score["title"], fontsize=9, fontweight='bold', color='white', loc='left')
                handles1 = [Patch(color=color, label=label) for color, label in final_risk_score["colors"]]
                ax1.legend(handles=handles1, loc='center left', frameon=False,
                           handleheight=1.2, handlelength=8, fontsize=9, labelcolor='white')

                # Historical Risk Classification - top right
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.axis('off')
                ax2.set_title(historical_classification["title"], fontsize=9, fontweight='bold', color='white')
                handles2 = [Patch(color=color, label=label) for color, label in historical_classification["colors"]]
                ax2.legend(handles=handles2, loc='center', ncol=3, frameon=False,
                           handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                # Historical Risk Score - bottom right
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.axis('off')
                ax3.set_title(historical_score["title"], fontsize=9, fontweight='bold', color='white')
                handles3 = [Patch(color=color, label=label) for color, label in historical_score["colors"]]
                ax3.legend(handles=handles3, loc='center', ncol=3, frameon=False,
                           handleheight=1.2, handlelength=5, fontsize=9, labelcolor='white')

                fig.patch.set_facecolor('black')
                plt.tight_layout(pad=1)
                st.pyplot(fig)

                # Display risk scores for all lines
                st.write("### Risk Scores for All Transmission Lines")
                risk_df_display = st.session_state.risk_df[["line_id", "from_bus", "to_bus", "risk_score"]].sort_values(by="risk_score", ascending=False)
                risk_df_display.columns = ["Line ID", "From Bus", "To Bus", "Risk Score"]
                st.dataframe(risk_df_display, use_container_width=True)

                # Display lines expected to face outage based on threshold
                if st.session_state.outage_data:
                    st.write(f"### Lines Expected to Face Outage (Risk Score ≥ {st.session_state.risk_score})")
                    outage_df = pd.DataFrame(st.session_state.outage_data)
                    outage_df.columns = ["Transmission Line", "Expected Outage Hours", "Risk Score"]
                    st.dataframe(outage_df, use_container_width=True)

                    # Summary statistics
                    st.write("### Outage Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Number of Lines at Risk", len(st.session_state.outage_data))
                    with col2:
                        st.metric("Max Temperature Occurrences", int(st.session_state.max_occurrences["temperature"]))
                    with col3:
                        st.metric("Max Precipitation Occurrences", int(st.session_state.max_occurrences["precipitation"]))
                    with col4:
                        st.metric("Max Wind Occurrences", int(st.session_state.max_occurrences["wind"]))
                else:
                    st.success(f"No transmission lines are expected to face outage at the selected risk threshold ({st.session_state.risk_score}).")
                    # Still display max occurrences even if no outages
                    st.write("### Historical Max Occurrences")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Temperature Occurrences", int(st.session_state.max_occurrences["temperature"]))
                    with col2:
                        st.metric("Max Precipitation Occurrences", int(st.session_state.max_occurrences["precipitation"]))
                    with col3:
                        st.metric("Max Wind Occurrences", int(st.session_state.max_occurrences["wind"]))
            else:
                st.info("Select parameters and click 'Process Weather Risk Data' to analyze weather risks to the electricity grid.")
                

# Page 3: Projected future operations - Under Current OPF
elif selection == "Projected Operation - Under Current OPF":
    st.title("Projected Operation - Under Current OPF")
    # ── PERSISTENT STORAGE (add right after st.title(...)) ────────────────────
    for k, v in (
        ("bau_ready",                          False),
        ("bau_day_end_df",                     None),
        ("bau_hourly_cost_df",                 None),
        ("bau_results",                        None),
        ("line_outages",                       None),
        ("line_idx_map",                       None),
        ("trafo_idx_map",                      None),
        ("max_loading_capacity",               None),
        ("max_loading_capacity_transformer",   None),
        ("bau_hour",                           0),        # hour to draw
    ):
        st.session_state.setdefault(k, v)
    # ──────────────────────────────────────────────────────────────────────────

    
    mode = st.selectbox("Select Contingency Mode",
                    ["Capped Contingency Mode",
                     "Maximum Contingency Mode"])
    cap_flag = (mode == "Capped Contingency Mode")
    
    if st.button("Run Analysis"):

         # ---------------------------------------------------------------------------
        # Aliases so the Colab names still resolve
        # ---------------------------------------------------------------------------
        def Network_initialize():
            return network_initialize(path)          # <— your global helper
        
        def overloaded_transformer_colab(net):
            # keep original single-arg call signature
            return overloaded_transformer(net, path, line_outages)
        # ---------------------------------------------------------------------------
        
        
        # ---------------------------------------------------------------------------
        # MAIN FUNCTION – unchanged numerical logic, no prints, returns DataFrames
        # ---------------------------------------------------------------------------
        def current_opf(line_outages):
            df_trafo = []
            
            path = st.session_state.get("uploaded_file")   # BytesIO object
        
            xls = pd.ExcelFile(path)                       # same object Colab used
        
            # ----------------------------------------------------------------------
            # 1. Build fresh network + get helper objects
            # ----------------------------------------------------------------------
            if "Transformer Parameters" in xls.sheet_names:
                [net, df_bus, df_slack, df_line, num_hours,
                 load_dynamic, gen_dynamic,
                 df_load_profile, df_gen_profile, df_trafo] = Network_initialize()
            else:
                [net, df_bus, df_slack, df_line, num_hours,
                 load_dynamic, gen_dynamic,
                 df_load_profile, df_gen_profile]          = Network_initialize()
        
            business_as_usuall_cost = calculating_hourly_cost(path)
        
            # ----------------------------------------------------------------------
            # 2. Set up spatial helpers (identical to Colab)
            # ----------------------------------------------------------------------
            df_lines = df_line.copy()
            df_lines["geodata"] = df_lines["geodata"].apply(
                lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x)
        
            gdf = gpd.GeoDataFrame(
                df_lines,
                geometry=[LineString(coords) for coords in df_lines["geodata"]],
                crs="EPSG:4326")
        
            load_df = pd.read_excel(path, sheet_name="Load Parameters")
            load_df["coordinates"] = load_df["load_coordinates"].apply(ast.literal_eval)
        
            # line / trafo index maps
            line_idx_map = {(r["from_bus"], r["to_bus"]): idx for idx, r in net.line.iterrows()}
            line_idx_map.update({(r["to_bus"], r["from_bus"]): idx for idx, r in net.line.iterrows()})
        
            trafo_idx_map = {}
            if "Transformer Parameters" in xls.sheet_names:
                trafo_idx_map = {(r["hv_bus"], r["lv_bus"]): idx for idx, r in net.trafo.iterrows()}
                trafo_idx_map.update({(r["lv_bus"], r["hv_bus"]): idx for idx, r in net.trafo.iterrows()})
        
            # ----------------------------------------------------------------------
            # 3. Book-keeping dicts  (unchanged)
            # ----------------------------------------------------------------------
            net.load["bus"] = net.load["bus"].astype(int)
            cumulative_load_shedding = {bus: {"p_mw": 0.0, "q_mvar": 0.0}
                                        for bus in net.load["bus"].unique()}
        
            total_demand_per_bus = {}
            p_cols = [c for c in df_load_profile.columns if c.startswith("p_mw_bus_")]
            q_cols = [c for c in df_load_profile.columns if c.startswith("q_mvar_bus_")]
            bus_ids = set(int(col.rsplit("_", 1)[1]) for col in p_cols)
            for bus in bus_ids:
                p_col, q_col = f"p_mw_bus_{bus}", f"q_mvar_bus_{bus}"
                total_demand_per_bus[bus] = {"p_mw": float(df_load_profile[p_col].sum()),
                                             "q_mvar": float(df_load_profile[q_col].sum())}
        
            # ----------------------------------------------------------------------
            # 4. Fixed 20 % shed fractions (same logic)
            # ----------------------------------------------------------------------
            initial_load_p = {}   # real power
            initial_load_q = {}   # reactive
            initial_load_p = {int(net.load.at[i, "bus"]): net.load.at[i, "p_mw"]
                              for i in net.load.index}
            initial_load_q = {int(net.load.at[i, "bus"]): net.load.at[i, "q_mvar"]
                              for i in net.load.index}
            shed_pct = 0.20
            fixed_shed_p = {b: shed_pct * p for b, p in initial_load_p.items()}
            fixed_shed_q = {b: shed_pct * q for b, q in initial_load_q.items()}
        
            # ----------------------------------------------------------------------
            # 5. Storage for hour-by-hour results
            # ----------------------------------------------------------------------
            hourly_shed_bau     = [0] * num_hours
            loading_records     = []
            loading_percent_bau = []
            served_load_per_hour= []
            gen_per_hour_bau    = []
            slack_per_hour_bau  = []
            shedding_buses      = []
            seen_buses          = set()
        
            # ----------------------------------------------------------------------
            # 6. ====  HOURLY LOOP  =================================================
            # ----------------------------------------------------------------------
            for hour in range(num_hours):
                # print(f"========== HOUR {hour} ==========")
        
                # 6-a) Apply scheduled outages
                for (fbus, tbus, start_hr) in line_outages:
                    if hour < start_hr:
                        continue
                    is_trafo = check_bus_pair(path, (fbus, tbus))
                    if is_trafo == True:
                        mask_tf = (((net.trafo.hv_bus == fbus) & (net.trafo.lv_bus == tbus)) |
                                   ((net.trafo.hv_bus == tbus) & (net.trafo.lv_bus == fbus)))    
                        if not mask_tf.any():
                            pass
                        else:
                            for tf_idx in net.trafo[mask_tf].index:
                                net.trafo.at[tf_idx, "in_service"] = False
                    else:
                        idx = line_idx_map.get((fbus, tbus))
                        if idx is not None:
                            net.line.at[idx, "in_service"] = False
        
                # 6-b) Update hourly load & gen profiles
                for idx in net.load.index:
                    bus = net.load.at[idx, "bus"]
                    if bus in load_dynamic:
                        net.load.at[idx, "p_mw"] = df_load_profile.at[hour, load_dynamic[bus]["p"]]
                        net.load.at[idx, "q_mvar"]= df_load_profile.at[hour, load_dynamic[bus]["q"]]
                for idx in net.gen.index:
                    bus = net.gen.at[idx, "bus"]
                    if bus in gen_dynamic:
                        net.gen.at[idx, "p_mw"] = df_gen_profile.at[hour, gen_dynamic[bus]]
        
                # 6-c) Re-read criticality each hour (kept identical)
                df_load_params = pd.read_excel(path, sheet_name="Load Parameters", index_col=0)
                crit_map = dict(zip(df_load_params["bus"], df_load_params["criticality"]))
                net.load["bus"] = net.load["bus"].astype(int)
                net.load["criticality"] = net.load["bus"].map(crit_map)
        
                # 6-d) Initial power-flow try
                flag_initial_fail = False
                try:
                    pp.runpp(net)
                except:
                    flag_initial_fail = True
        
                if flag_initial_fail == False:
                    inter = transform_loading(net.res_line["loading_percent"])
                    if "Transformer Parameters" in xls.sheet_names:
                        inter.extend(transform_loading(net.res_trafo["loading_percent"].tolist()))
                    loading_records.append(inter)
                    loading_percent_bau.append(inter.copy())
                else:
                    loading_records.append([0]*(len(net.res_line)+len(df_trafo)))
                    loading_percent_bau.append([0]*(len(net.res_line)+len(df_trafo)))
        
                # 6-e) Check overloads and shed if needed
                overloads       = overloaded_lines(net)
                overloads_trafo = overloaded_transformer_colab(net)
                all_loads_zero_flag = False
        
                if (overloads == []) and (overloads_trafo == []) and (all_real_numbers(loading_records[-1])):
                    slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]))
                    # served_load_per_hour.append(net.load["p_mw"].tolist())
                    # gen_per_hour_bau.append(net.res_gen["p_mw"].tolist())

                    if net.load["p_mw"].isnull().any():
                        served_load_per_hour.append([None] * len(net.load))
                    else:
                        hourly_loads = net.load["p_mw"].tolist()
                        served_load_per_hour.append(hourly_loads)
        
                    if net.res_gen["p_mw"].isnull().any():
                        gen_per_hour_bau.append([None] * len(net.res_gen))
                    else:
                        hourly_gen = net.res_gen["p_mw"].tolist()
                        gen_per_hour_bau.append(hourly_gen)
                    continue
                else:
                    while ((overloaded_lines(net) or overloaded_transformer_colab(net)) and
                           not all_loads_zero_flag):
        
                        for crit in sorted(net.load["criticality"].dropna().unique(), reverse=True):
                            for ld_idx in net.load[net.load["criticality"] == crit].index:
                                if (not overloaded_lines(net)) and (not overloaded_transformer_colab(net)):
                                    break
        
                                bus = net.load.at[ld_idx, "bus"]
                                dp, dq = fixed_shed_p[bus], fixed_shed_q[bus]
                                net.load.at[ld_idx, "p_mw"]  -= dp
                                net.load.at[ld_idx, "q_mvar"]-= dq
        
                                shedding_buses.append((hour, int(bus)))
                                cumulative_load_shedding[bus]["p_mw"]  += dp
                                cumulative_load_shedding[bus]["q_mvar"]+= dq
                                hourly_shed_bau[hour]                  += dp
        
                                try:
                                    pp.runopp(net)
                                    business_as_usuall_cost[hour] = net.res_cost if net.OPF_converged else business_as_usuall_cost[hour]
                                    if net.OPF_converged:
                                        pf_loading = transform_loading(net.res_line["loading_percent"])
                                        if "Transformer Parameters" in xls.sheet_names:
                                            pf_loading.extend(transform_loading(net.res_trafo["loading_percent"]))
                                        if all_real_numbers(pf_loading):
                                            all_loads_zero_flag = True
                                    business_as_usuall_cost[hour] = net.res_cost       
                                except:
                                    pp.runpp(net)
                                
                                # if this load has now gone negative, slam to zero
                                if net.load.at[ld_idx, "p_mw"] - dp < 0:
                                    all_loads_zero_flag = True
                                    business_as_usuall_cost[hour] = 0
                                    
                                    remaining_p = net.load.loc[net.load["bus"] == bus, "p_mw"].sum()
                                    remaining_q = net.load.loc[net.load["bus"] == bus, "q_mvar"].sum()
                                    cumulative_load_shedding[bus]["p_mw"]  += remaining_p
                                    cumulative_load_shedding[bus]["q_mvar"]+= remaining_q
                                    hourly_shed_bau[hour] += sum(net.load["p_mw"])
                                    
                                    for i in range(len(net.load)):
                                        net.load.at[i, 'p_mw'] = 0
                                        net.load.at[i, 'q_mvar'] = 0
                                    break

                    # record final served, gen, slack
                    if net.load["p_mw"].isnull().any():
                        served_load_per_hour.append([None] * len(net.load))
                    else:
                        hourly_loads = net.load["p_mw"].tolist()
                        served_load_per_hour.append(hourly_loads)

                    
                    if (net.res_gen["p_mw"].isnull().any()) or (business_as_usuall_cost[hour] == 0):
                        gen_per_hour_bau.append([None]*len(net.res_gen))
                        slack_per_hour_bau.append(None)
                    else:
                        hourly_gen = net.res_gen["p_mw"].tolist()
                        gen_per_hour_bau.append(net.res_gen["p_mw"].tolist())
                        slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]))
        
            # ----------------------------------------------------------------------
            # 7. Build Day-End Summary table  (instead of prints)
            # ----------------------------------------------------------------------
            summary_rows = []
            for bus, shed in cumulative_load_shedding.items():
                total = total_demand_per_bus.get(bus, {"p_mw": 0.0, "q_mvar": 0.0})
                summary_rows.append({
                    "bus": bus,
                    "load shedding (MWh)":  shed["p_mw"],
                    "load shedding (MVARh)":shed["q_mvar"],
                    "total demand (MWh)":   total["p_mw"],
                    "total demand (MVARh)": total["q_mvar"]
                })
            day_end_df = pd.DataFrame(summary_rows)
        
            # ----------------------------------------------------------------------
            # 8. Build Hourly Generation Cost table
            # ----------------------------------------------------------------------
            hourly_cost_df = pd.DataFrame({
                "hour": list(range(len(business_as_usuall_cost))),
                "generation_cost (PKR)": business_as_usuall_cost
            })
        
            # ----------------------------------------------------------------------
            # 9. Return everything Colab returned *plus* the two DataFrames
            # ----------------------------------------------------------------------
            return (loading_percent_bau, served_load_per_hour, gen_per_hour_bau,
                    slack_per_hour_bau, loading_records, business_as_usuall_cost,
                    hourly_shed_bau, seen_buses, shedding_buses, df_lines, df_trafo,
                    load_df, line_idx_map, trafo_idx_map, gdf,
                    day_end_df, hourly_cost_df)

    
        # build the outage list first
        line_outages = generate_line_outages(
            outage_hours   = st.session_state["outage_hours"],
            line_down      = st.session_state["line_down"],
            risk_scores    = st.session_state["risk_scores"],
            capped_contingency_mode = cap_flag
        )
        st.session_state.line_outages = line_outages 
    
        # store globally for helper functions
        globals()["line_outages"] = line_outages
    
        with st.spinner("Running OPF …"):
            (_lp_bau, _served, _gen, _slack, _rec, _cost,
             _shed, _seen, _shed_buses, _df_lines, _df_trafo,
             _load_df, _line_idx_map, _trafo_idx_map, _gdf,
             day_end_df, hourly_cost_df) = current_opf(line_outages)

        # -----------------------------------------------------------------
        # CACHE RESULTS so they persist across page switches
        # -----------------------------------------------------------------
        # 2-C · WRITE ALL RESULTS TO SESSION STATE  ←── only here!
        st.session_state.update({
            "bau_ready":                        True,
            "bau_day_end_df":                   day_end_df,
            "bau_hourly_cost_df":               hourly_cost_df,
            "bau_results": {
                "loading_percent_bau": _lp_bau,
                "shedding_buses":      _shed_buses,
            },
            "line_idx_map":                     _line_idx_map,
            "trafo_idx_map":                    _trafo_idx_map,
            "max_loading_capacity":             _df_lines["max_loading_percent"].max(),
        })
        if _df_trafo is not None and not _df_trafo.empty:
            st.session_state.max_loading_capacity_transformer = (
                _df_trafo["max_loading_percent"].max()
            )
        # -----------------------------------------------------------------
    
        # st.subheader("Day-End Summary")
        # st.dataframe(day_end_df, use_container_width=True)
    
        # st.subheader("Hourly Generation Cost")
        # st.dataframe(hourly_cost_df, use_container_width=True)

        # ────────────────────────────────────────────────────────────────
        # Show cached tables even after you left the page
        # ────────────────────────────────────────────────────────────────
      # ░░ 3 · ALWAYS-VISIBLE OUTPUT (tables + map) ░░
    if st.session_state.bau_ready:

        # 3-A · Summary tables
        st.subheader("Day-End Summary")
        st.dataframe(st.session_state.bau_day_end_df, use_container_width=True)

        st.subheader("Hourly Generation Cost")
        st.dataframe(st.session_state.bau_hourly_cost_df, use_container_width=True)

        # 3-B · Hour picker (updates st.session_state.bau_hour automatically)
        num_hours   = len(st.session_state.network_data['df_load_profile'])
        st.selectbox(
            "Select Hour to Visualize",
            [f"Hour {i}" for i in range(num_hours)],
            index=st.session_state.bau_hour,
            key="bau_hour",
            help="Choose any hour; the map refreshes automatically.",
        )

        # 3-C · Build the map for that hour
        hr           = st.session_state.bau_hour
        df_line      = st.session_state.network_data['df_line'].copy()
        df_load      = st.session_state.network_data['df_load'].copy()
        df_trafo     = st.session_state.network_data.get('df_trafo')
        loading_rec  = st.session_state.bau_results['loading_percent_bau'][hr]
        shed_buses   = st.session_state.bau_results['shedding_buses']
        line_idx_map = st.session_state.line_idx_map
        trafo_idx_map= st.session_state.trafo_idx_map
        outages      = st.session_state.line_outages

        # ── helper colour fns (same logic) ─────────────────────
        def get_color(pct, max_cap):
            if pct is None:                return '#FF0000'
            if pct == 0:                   return '#000000'
            if pct <= 0.75*max_cap:        return '#00FF00'
            if pct <= 0.90*max_cap:        return '#FFFF00'
            if pct <  max_cap:             return '#FFA500'
            return '#FF0000'
        get_color_trafo = get_color

        # distinguish line vs trafo
        def check_bus_pair_df(df_line, df_trafo, pair):
            fbus, tbus = pair
            if df_trafo is not None:
                if (((df_trafo["hv_bus"] == fbus) & (df_trafo["lv_bus"] == tbus)) |
                    ((df_trafo["hv_bus"] == tbus) & (df_trafo["lv_bus"] == fbus))).any():
                    return True
            if (((df_line["from_bus"] == fbus) & (df_line["to_bus"] == tbus)) |
                ((df_line["from_bus"] == tbus) & (df_line["to_bus"] == fbus))).any():
                return False
            return None

        # GeoDataFrame for lines
        df_line["geodata"] = df_line["geodata"].apply(
            lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
        )
        gdf = gpd.GeoDataFrame(
            df_line,
            geometry=[LineString(c) for c in df_line["geodata"]],
            crs="EPSG:4326",
        )
        gdf["idx"]     = gdf.index
        gdf["loading"] = gdf["idx"].map(lambda i: loading_rec[i] if i < len(loading_rec) else 0.0)

        # mark weather-down equipment
        weather_down = set()
        for fbus, tbus, start_hr in outages:
            if hr >= start_hr:
                is_tf = check_bus_pair_df(df_line, df_trafo, (fbus, tbus))
                if is_tf:
                    idx = trafo_idx_map.get((fbus, tbus))
                    if idx is not None:
                        weather_down.add(idx + len(df_line))
                else:
                    idx = line_idx_map.get((fbus, tbus))
                    if idx is not None:
                        weather_down.add(idx)
        gdf["down_weather"] = gdf["idx"].isin(weather_down)

        # Folium map
        m = folium.Map(location=[27, 66.5], zoom_start=6, width=800, height=600)
        max_line_cap = st.session_state.max_loading_capacity
        max_trf_cap  = st.session_state.get("max_loading_capacity_transformer", max_line_cap)
        no_of_lines  = len(df_line)

        def style_fn(feat):
            p = feat["properties"]
            if p.get("down_weather", False):
                return {"color": "#000000", "weight": 3}
            pct = p.get("loading", 0.0)
            colour = (get_color_trafo(pct, max_trf_cap)
                      if df_trafo is not None and p["idx"] >= no_of_lines
                      else get_color(pct, max_line_cap))
            return {"color": colour, "weight": 3}

        folium.GeoJson(gdf.__geo_interface__, style_function=style_fn,
                       name=f"Transmission Net – Hour {hr}").add_to(m)

        # load circles
        shed_now = [b for (h, b) in shed_buses if h == hr]
        for _, row in df_load.iterrows():
            bus = row["bus"]
            lat, lon = ast.literal_eval(row["load_coordinates"])
            col = "red" if bus in shed_now else "green"
            folium.Circle((lat, lon), radius=20000,
                          color=col, fill_color=col, fill_opacity=0.5).add_to(m)

        # legend + title  (same HTML you used)
        legend_html = """<style>.legend-entry{display:flex;align-items:center;margin-bottom:4px;}
        .legend-swatch{width:12px;height:12px;margin-right:6px;display:inline-block;}
        .circle{border-radius:70%;}</style>
        <div style="background:#fff;padding:8px;border:1px solid #ccc;">
        <strong>Line Load Level (% of Max) & Load Status</strong>
        <div class='legend-entry'><span class='legend-swatch' style='background:#00FF00'></span>Below 75%</div>
        <div class='legend-entry'><span class='legend-swatch' style='background:#FFFF00'></span>75–90%</div>
        <div class='legend-entry'><span class='legend-swatch' style='background:#FFA500'></span>90–100%</div>
        <div class='legend-entry'><span class='legend-swatch' style='background:#FF0000'></span>Overloaded &gt;100%</div>
        <div class='legend-entry'><span class='legend-swatch' style='background:#000000'></span>Weather-Impacted</div>
        <div class='legend-entry'><span class='legend-swatch circle' style='background:#008000'></span>Fully Served</div>
        <div class='legend-entry'><span class='legend-swatch circle' style='background:#FF0000'></span>Not Fully Served</div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        title_html = f"""
        <div style='font-size:18px;font-weight:bold;background:rgba(255,255,255,0.8);padding:4px;'>
        Projected Operation – Current OPF · Hour {hr}</div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        folium.LayerControl(collapsed=False).add_to(m)

        st.write(f"### Network Loading Visualization – Hour {hr}")
        st_folium(m, width=800, height=600, key=f"bau_map_{hr}")
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









