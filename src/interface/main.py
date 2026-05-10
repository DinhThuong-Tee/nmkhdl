import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

from utils.geo import vn2000_to_latlon
from utils.forecast import predict_for_station
from utils.hsi import compute_hsi

st.set_page_config(layout="centered")

st.title("🌊 Dự báo môi trường nước cho Cá giò và Hàu khu vực biển Quảng Ninh")

# ==============================================================================
# --- PHẦN 1: LOAD DỮ LIỆU  ---
# ==============================================================================

@st.cache_data
def load_data():
    df = pd.read_csv('data/data_quang_ninh/qn_env_clean_ready.csv')
    if 'Quarter' in df.columns:
        df['Quarter'] = pd.to_datetime(df['Quarter'], errors='coerce')
    coords = df[['X', 'Y']].drop_duplicates()
    coords['lat'] = None
    coords['lon'] = None
    for idx, row in coords.iterrows():
        lat, lon = vn2000_to_latlon(row['X'], row['Y'])
        coords.at[idx, 'lat'] = lat
        coords.at[idx, 'lon'] = lon
    df = df.merge(coords[['X', 'Y', 'lat', 'lon']], on=['X', 'Y'], how='left')
    return df

@st.cache_data
def load_radius_data(species):
    try:
        filename = f'data/data_quang_ninh/R_{species}.csv'
        df_radius = pd.read_csv(filename)
        return df_radius
    except FileNotFoundError:
        st.warning(f"Không tìm thấy file {filename}")
        return None

@st.cache_data
def calculate_hsi_for_all_stations(species, year, quarter, station_list):
    import concurrent.futures
    def calculate_single_station(station_row):
        try:
            forecast_df = predict_for_station(species=species, x=station_row['X'], y=station_row['Y'],
                start_year=year, start_quarter=quarter, n_quarters=1)
            forecast_with_hsi = compute_hsi(forecast_df, species=species)
            if len(forecast_with_hsi) > 0:
                return (station_row['Station'], {
                    'HSI': forecast_with_hsi.iloc[0]['HSI'],
                    'HSI_Level': forecast_with_hsi.iloc[0]['HSI_Level']
                })
        except: pass
        return None
    hsi_results = {}
    stations_list = station_list.to_dict('records')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(calculate_single_station, stations_list)
    for result in results:
        if result: hsi_results[result[0]] = result[1]
    return hsi_results

df = load_data()
stations = df[['Station', 'Station_Name', 'lat', 'lon']].drop_duplicates()

# ==============================================================================
# PHẦN 2: BẢN ĐỒ CÁC TRẠM QUAN TRẮC 
# ==============================================================================
st.header("🗺 Bản đồ các trạm quan trắc môi trường")

# --- Cài đặt hiển thị bản đồ ---
st.subheader("⚙️ Cài đặt hiển thị bản đồ")
col_map1, col_map2, col_map3, col_map4 = st.columns(4)

with col_map1:
    # Chuyển chọn Loài lên đây vì bản đồ cần thông tin loài để hiển thị HSI và bán kính
    species_display = st.selectbox("Loài", options=["Cá giò", "Hàu"], index=0, key="main_species_select")
    species = "cobia" if species_display == "Cá giò" else "oyster"

with col_map2:
    map_year = st.number_input("Năm hiển thị", min_value=2026, max_value=2030, value=2026, step=1, key="map_year")

with col_map3:
    map_quarter = st.selectbox("Quý hiển thị", options=[1, 2, 3, 4], index=0, key="map_quarter")

with col_map4:
    st.markdown("<div style='padding-top: 35px;'></div>", unsafe_allow_html=True)
    show_hsi = st.checkbox("Hiển thị HSI", value=True, help="Tính toán và hiển thị HSI cho tất cả các trạm")

# Load radius data
df_radius = load_radius_data(species)

# Calculate HSI for map markers
hsi_data = {}
if show_hsi:
    with st.spinner('Đang tính toán HSI cho các trạm trên bản đồ...'):
        stations_unique = df[['Station', 'X', 'Y']].drop_duplicates()
        hsi_data = calculate_hsi_for_all_stations(species, map_year, map_quarter, stations_unique)

st.info("💡 **Hướng dẫn:** Click vào các điểm đỏ trên bản đồ để chọn trạm và xem chi tiết. Vòng tròn màu xanh biểu thị vùng áp dụng kết quả dự báo cho Q{}/{}. Hover chuột để xem thông tin nhanh.".format(map_quarter, map_year))

# Tạo Folium map
center_lat, center_lon = stations['lat'].mean(), stations['lon'].mean()
m = folium.Map(
    location=[center_lat, center_lon], zoom_start=10,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery'
)

# Thêm vòng tròn bán kính
if df_radius is not None:
    radius_filtered = df_radius[(df_radius['year'] == map_year) & (df_radius['quarter'] == map_quarter)].copy()
    radius_filtered = radius_filtered.merge(stations[['Station', 'lat', 'lon']], left_on='station', right_on='Station', how='left')
    for idx, row in radius_filtered.iterrows():
        if pd.notna(row['lat']) and pd.notna(row['lon']) and pd.notna(row['R_km']):
            folium.Circle(
                location=[row['lat'], row['lon']], radius=row['R_km'] * 1000,
                color='#2E86AB', fill=True, fillColor='#2E86AB', fillOpacity=0.15, weight=2, opacity=0.5,
                popup=folium.Popup(f"<b>{row['station']}</b><br>Bán kính: {row['R_km']} km", max_width=200),
                tooltip=f"{row['station']}: R = {row['R_km']} km"
            ).add_to(m)

# Thêm Marker trạm
for idx, row in stations.iterrows():
    marker_color = '#C81E1E'
    hsi_tooltip = ""
    if row['Station'] in hsi_data:
        hsi_val = hsi_data[row['Station']]['HSI']
        hsi_lvl = hsi_data[row['Station']]['HSI_Level']
        hsi_tooltip = f" | HSI: {hsi_val:.3f} ({hsi_lvl})"
        if hsi_val >= 0.85: marker_color = '#28a745'
        elif hsi_val >= 0.75: marker_color = '#ffc107'
        elif hsi_val >= 0.5: marker_color = '#fd7e14'
        else: marker_color = '#dc3545'

    popup_html = f"""<div style="font-family: Arial; width: 240px;"><h4 style="color: #2E86AB; margin: 0 0 10px 0;">{row['Station']}</h4>
    <p><b>Tên:</b> {row['Station_Name']}</p>{f"<p><b>HSI:</b> {hsi_val:.3f}</p><p><b>Đánh giá:</b> {hsi_lvl}</p>" if row['Station'] in hsi_data else ""}</div>"""
    
    folium.CircleMarker(
        location=[row['lat'], row['lon']], radius=8,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{row['Station']} - {row['Station_Name']}{hsi_tooltip}",
        color=marker_color, fill=True, fillColor=marker_color, fillOpacity=0.7, weight=2
    ).add_to(m)

# Legend
if show_hsi:
    legend_html = """<div style="position: fixed; bottom: 50px; right: 50px; width: 200px; height: auto; background-color: white; z-index:9999; font-size:14px; border:2px solid grey; border-radius: 5px; padding: 10px">
    <p><b>Chỉ số HSI:</b></p>
    <p><span style="color: #28a745;">●</span> Rất phù hợp (≥0.85)</p><p><span style="color: #ffc107;">●</span> Phù hợp (≥0.75)</p>
    <p><span style="color: #fd7e14;">●</span> Ít phù hợp (≥0.5)</p><p><span style="color: #dc3545;">●</span> Không phù hợp (<0.5)</p></div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

# Initialize session state
if 'selected_station' not in st.session_state:
    st.session_state.selected_station = None

# Display map
map_data = st_folium(m, width=None, height=500, returned_objects=["last_object_clicked"], key="map_top")

# Handle Click
if map_data and map_data.get("last_object_clicked"):
    c_lat, c_lon = map_data["last_object_clicked"]["lat"], map_data["last_object_clicked"]["lng"]
    stations_copy = stations.copy()
    stations_copy['distance'] = ((stations_copy['lat'] - c_lat)**2 + (stations_copy['lon'] - c_lon)**2)**0.5
    closest = stations_copy.loc[stations_copy['distance'].idxmin(), 'Station']
    if st.session_state.selected_station != closest:
        st.session_state.selected_station = closest
        st.rerun()

st.divider()

# ==============================================================================
# PHẦN 3: THAM SỐ DỰ BÁO CHI TIẾT
# ==============================================================================
st.header("🔮 Tham số dự báo chi tiết")

col1, col2, col3 = st.columns(3)
with col1:
    start_year = st.number_input("Năm bắt đầu dự báo", min_value=2026, max_value=2030, value=map_year, step=1)
with col2:
    start_quarter = st.selectbox("Quý bắt đầu dự báo", options=[1, 2, 3, 4], index=map_quarter-1)
with col3:
    n_quarters = st.number_input("Số quý dự báo", min_value=1, max_value=20, value=4, step=1)

# --- Chọn trạm để tính toán ---
st.subheader("🎯 Tính toán chỉ số HSI chi tiết cho trạm")
stations_sorted = stations.copy()
stations_sorted['sort_key'] = stations_sorted['Station'].str.extract('(\d+)').astype(int)
stations_sorted = stations_sorted.sort_values('sort_key')

col_select1, col_select2 = st.columns([3, 1])
with col_select1:
    default_index = 0
    if st.session_state.selected_station in stations_sorted['Station'].values:
        default_index = stations_sorted['Station'].tolist().index(st.session_state.selected_station)
    
    selected_station = st.selectbox("Chọn trạm:", options=stations_sorted['Station'].tolist(),
        format_func=lambda x: f"{x} - {stations_sorted[stations_sorted['Station']==x]['Station_Name'].values[0]}",
        index=default_index, key="station_selector")
    st.session_state.selected_station = selected_station

with col_select2:
    st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
    calculate_btn = st.button("📊 Tính HSI chi tiết", type="primary", use_container_width=True)

# --- Hiển thị Kết quả dự báo (Tab view) ---
if selected_station:
    st.session_state.last_station = selected_station
    station_data = df[df['Station'] == selected_station][['X', 'Y', 'Station_Name']].iloc[0]
    
    with st.spinner(f'Đang tính toán HSI cho trạm {selected_station}...'):
        try:
            forecast_df = predict_for_station(species=species, x=station_data['X'], y=station_data['Y'],
                start_year=start_year, start_quarter=start_quarter, n_quarters=n_quarters)
            forecast_with_hsi = compute_hsi(forecast_df, species=species)
            
            # Merge radius info
            if df_radius is not None:
                r_list = []
                for _, r in forecast_with_hsi.iterrows():
                    sr = df_radius[(df_radius['station']==selected_station) & (df_radius['year']==int(r['year'])) & (df_radius['quarter']==int(r['quarter']))]
                    r_list.append(sr.iloc[0]['R_km'] if len(sr)>0 else None)
                forecast_with_hsi['R_km'] = r_list

            hsi_results = []
            for _, row in forecast_with_hsi.iterrows():
                res = {'Thời gian': f"Q{int(row['quarter'])}/{int(row['year'])}", 'HSI': round(row['HSI'], 3), 'Đánh giá': row['HSI_Level']}
                if 'R_km' in row and pd.notna(row['R_km']): res['Bán kính (km)'] = row['R_km']
                hsi_results.append(res)
            hsi_df = pd.DataFrame(hsi_results)

            st.success(f"✅ Kết quả HSI cho trạm **{selected_station}** - {station_data['Station_Name']}")
            
            tab1, tab2, tab3 = st.tabs(["📈 Biểu đồ HSI", "🌡️ Biểu đồ các thông số môi trường", "📋 Bảng dữ liệu"])
            
            with tab1:
                chart_data = hsi_df.copy()
                chart_data['HSI_numeric'] = pd.to_numeric(chart_data['HSI'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_data['Thời gian'], y=chart_data['HSI_numeric'], mode='lines+markers', name='HSI', line=dict(color='#2E86AB', width=3)))
                fig.add_hline(y=0.85, line_dash="dash", line_color="green", annotation_text="Rất phù hợp")
                fig.add_hline(y=0.75, line_dash="dash", line_color="orange", annotation_text="Phù hợp")
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Ít phù hợp")
                fig.update_layout(title=f"Xu hướng HSI - {species_display}", xaxis_title="Thời gian", yaxis_title="HSI", yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                c_s1, c_s2, c_s3 = st.columns(3)
                c_s1.metric("HSI trung bình", f"{chart_data['HSI_numeric'].mean():.3f}")
                c_s2.metric("HSI thấp nhất", f"{chart_data['HSI_numeric'].min():.3f}")
                c_s3.metric("HSI cao nhất", f"{chart_data['HSI_numeric'].max():.3f}")

            with tab2:
                param_names = {'temp': 'Nhiệt độ (°C)', 'salinity': 'Độ mặn (‰)', 'DO': 'Oxy hòa tan (mg/L)', 'pH': 'pH', 'PO4': 'Phosphat (mg/L)'}
                avail = [c for c in forecast_with_hsi.columns if c in param_names.keys()]
                sel_params = st.multiselect("Chọn thông số hiển thị:", options=avail, default=avail[:3], format_func=lambda x: param_names.get(x, x))
                if sel_params:
                    from plotly.subplots import make_subplots
                    rows = (len(sel_params) + 1) // 2
                    fig_env = make_subplots(rows=rows, cols=2, subplot_titles=[param_names.get(p, p) for p in sel_params])
                    for i, p in enumerate(sel_params):
                        r, c = (i // 2) + 1, (i % 2) + 1
                        fig_env.add_trace(go.Scatter(x=hsi_df['Thời gian'], y=forecast_with_hsi[p], mode='lines+markers', name=param_names.get(p, p)), row=r, col=c)
                    fig_env.update_layout(height=300 * rows, showlegend=False)
                    st.plotly_chart(fig_env, use_container_width=True)

            with tab3:
                st.dataframe(hsi_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

st.divider()

# --- PHẦN 4: THỐNG KÊ DỮ LIỆU (CUỐI TRANG) ---
st.subheader("📊 Thông tin dữ liệu")
cf1, cf2, cf3 = st.columns(3)
cf1.metric("Số trạm quan trắc", len(stations))
cf2.metric("Tổng số mẫu", len(df))
cf3.metric("Số năm dữ liệu", df['Quarter'].dt.year.nunique() if 'Quarter' in df.columns else 'N/A')

with st.expander("📋 Xem danh sách các trạm quan trắc"):
    st.dataframe(stations.sort_values('Station')[['Station', 'Station_Name', 'lat', 'lon']], use_container_width=True, hide_index=True)