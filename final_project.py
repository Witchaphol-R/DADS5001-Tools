from streamlit_folium import st_folium
from shapely.geometry import Polygon
import typing_extensions as typing
import google.generativeai as genai
import streamlit as st
import plotly.express as px
import geopandas as gpd
import pandas as pd
import folium as fl
import toml, geohash, ast

config = toml.load('credential.toml')
api_key = config['api']['key']
genai.configure(api_key=api_key)

if 'chat' not in st.session_state:
  st.session_state.model = genai.GenerativeModel('gemini-1.5-flash')
  st.session_state.chat = st.session_state.model.start_chat(history=[])
if 'prediction' not in st.session_state:
  st.session_state.prediction = None
if 'trends' not in st.session_state:
  st.session_state.trends = None
if 'recommendations' not in st.session_state:
  st.session_state.recommendations = None
if 'last_option' not in st.session_state:
  st.session_state.last_option = None

def predict_success(location, cuisine_type, data_sample):
  custom_prompt = f"Predict the success of a restaurant at {location} with cuisine {cuisine_type}. Provide one sentence for prediction and another sentence for reasoning."
  data_str = f"Here is the data: {data_sample.to_dict()}"
  response = st.session_state.model.generate_content([custom_prompt, data_str])
  content = response.candidates[0].content.parts[0].text
  return content

def identify_trends(data_sample):
  custom_prompt = f"You are an app that help analyze the cuisine trends based on the given data. Your job is to provide the top cuisines, taste profile based on the top cuisines, and suggest some food that might sell well. We are aware of the bias and assumption, please answer to the best of your ability and make sure to not include technical method such as script, as you will be explaining to business users."
  data_str = f"Here is the data: {data_sample.to_dict()}"
  response = st.session_state.model.generate_content([custom_prompt, data_str])
  content = response.candidates[0].content.parts[0].text
  return content

def recommend_locations(cuisine_type, data_sample):
  custom_prompt = f"Recommend potential locations for a restaurant with cuisine/dish: {cuisine_type} based on the following data. Infer potential locations where people might like this cuisine. Select the top 10 geohash5 from the given sample and return them in a list format."
  data_str = f"Here is the data: {data_sample.to_dict()}"
  class geohashes(typing.TypedDict):
    geohash: list[str]
  response = st.session_state.model.generate_content(
    [custom_prompt, data_str]
    , generation_config=genai.GenerationConfig(response_mime_type="application/json", response_schema=list[geohashes])
  )
  content = response.candidates[0].content.parts[0].text
  geohash_list = ast.literal_eval(content)
  recommended_geohashes = geohash_list[0]['geohash']
  return recommended_geohashes

def add_geohash5_column(data):
  data['geohash5'] = data['geohash'].apply(lambda x: x[:5])
  return data

def geohash_to_polygon(geohash_code):
  lat, lon, lat_err, lon_err = geohash.decode_exactly(geohash_code)
  swlat = lat - lat_err
  swlng = lon - lon_err
  nelat = lat + lat_err
  nelng = lon + lon_err
  return Polygon([
    (swlng, swlat),
    (nelng, swlat),
    (nelng, nelat),
    (swlng, nelat),
    (swlng, swlat)
  ])

st.set_page_config(layout="wide")
st.title("Restaurant Insights and Recommendations")
st.markdown("<style> .stMainBlockContainer {padding-top: 3rem;} </style>", unsafe_allow_html=True)

with st.sidebar:
  uploaded_file = st.file_uploader(label="uploaded_file", type="csv", label_visibility="hidden")
  if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = add_geohash5_column(data)
    with st.expander("Data Preview"):
      st.write(data.head())
    geohash_counts = data['geohash5'].value_counts().head(20).reset_index()
    geohash_counts.columns = ['geohash5', 'count']
    fig = px.bar(
      geohash_counts
      , x='count'
      , y='geohash5'
      , orientation='h'
      , title='Top Geohashes'
      , labels={'geohash5': 'Geohash', 'count': 'Data Points'}
      , color_discrete_sequence=['purple']
    )
    fig.update_layout(
      yaxis=dict(categoryorder='total ascending', title=None)
      , xaxis=dict(showgrid=False, zeroline=False, visible=False) 
      , height=400
      , margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

if uploaded_file is not None:
  col1, col2 = st.columns(2)
  with col1:
    option = st.selectbox("Choose an option", ["Predict Success", "Cuisine Analysis", "Recommend Locations"], key='option_select')
  with col2:
    city = st.selectbox("City", data['city'].unique(), key='city_select')
  city_data = data[data['city'] == city]

  if st.session_state.option_select != st.session_state.last_option:
    st.session_state.prediction = None
    st.session_state.trends = None
    st.session_state.recommendations = None
    st.session_state.last_option = st.session_state.option_select

  col1, col2 = st.columns(2)
  with col1:
    if option != "Recommend Locations":
      st.header("Geohash Map")
      st.markdown(
        """
        <div style="background-color: #26272f; color: #24ff74; padding: 10px; border-radius: 5px;">
          Select the geohash boundary from the interactive map to filter the area of interest.
        </div>
        """, unsafe_allow_html=True)
      map_center = [city_data['latitude'].mean(), city_data['longitude'].mean()]
      m = fl.Map(location=map_center, zoom_start=10)
      fl.TileLayer('cartodbpositron').add_to(m)

      geohash_polygons = [geohash_to_polygon(gh) for gh in city_data['geohash5'].unique()]
      gdf = gpd.GeoDataFrame({'geohash5': city_data['geohash5'].unique(), 'geometry': geohash_polygons})
      gdf.set_crs(epsg=4326, inplace=True)
      geojson = fl.GeoJson(
        gdf,
        style_function=lambda x: {
          "fillColor": "orange",
          "color": "orange",
          "weight": 2,
          "fillOpacity": 0.5
        },
        tooltip=fl.GeoJsonTooltip(fields=['geohash5'], aliases=['Geohash:'])
      ).add_to(m)

      click_js = """
      <script>
      function handleMapClick(e) {
        var geohash = encodeGeohash(e.latlng.lat, e.latlng.lng, 5);
        var geohash_input = window.parent.document.querySelector('input[data-testid="stTextInput"][aria-label="Selected Geohash"]');
        geohash_input.value = geohash;
        geohash_input.dispatchEvent(new Event('input', { bubbles: true }));
      }
      function encodeGeohash(latitude, longitude, precision) {
        var BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz";
        var BITS = [16, 8, 4, 2, 1];
        var is_even = 1;
        var i = 0;
        var lat = [];
        var lon = [];
        var bit = 0;
        var ch = 0;
        var geohash = "";

        lat[0] = -90.0;
        lat[1] = 90.0;
        lon[0] = -180.0;
        lon[1] = 180.0;

        while (geohash.length < precision) {
          if (is_even) {
            mid = (lon[0] + lon[1]) / 2;
            if (longitude > mid) {
              ch |= BITS[bit];
              lon[0] = mid;
            } else {
              lon[1] = mid;
            }
          } else {
            mid = (lat[0] + lat[1]) / 2;
            if (latitude > mid) {
              ch |= BITS[bit];
              lat[0] = mid;
            } else {
              lat[1] = mid;
            }
          }

          is_even = !is_even;
          if (bit < 4) {
            bit++;
          } else {
            geohash += BASE32[ch];
            bit = 0;
            ch = 0;
          }
        }
        return geohash;
      }
      var map = {{this.get_name()}};
      map.on('click', handleMapClick);
      </script>
      """
      m.get_root().html.add_child(fl.Element(click_js))
      map_data = st_folium(m, height=250, width=700)

      if map_data.get("last_clicked"):
        lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        selected_geohash = geohash.encode(lat, lng, precision=5)
        st.text_input("Selected Geohash", value=selected_geohash, key="geohash_input")

      if st.session_state.get("geohash_input"):
        selected_geohash = st.session_state["geohash_input"]
        geohash_data = city_data[city_data['geohash5'] == selected_geohash]
        if not geohash_data.empty:
          sample_data = geohash_data.sample(n=min(1000, len(geohash_data)), random_state=42)
          sample_data = sample_data[['mex_name', 'all_cuisines', 'rating', 'review_number', 'geohash5']]
          if option == "Predict Success":
            cuisine_type = st.text_input("Cuisine Type")
            if st.button("Predict"):
              output_placeholder = col2.empty()
              output_placeholder.warning("Processing...")
              prediction = predict_success(selected_geohash, cuisine_type, sample_data)
              output_placeholder.empty()
              st.session_state.prediction = prediction
          elif option == "Cuisine Analysis":
            if st.button("Analyze"):
              pending_placeholder = col2.empty()
              pending_placeholder.warning("Processing...")
              trends = identify_trends(sample_data)
              st.session_state.trends = trends
              pending_placeholder.empty()
            if st.session_state.trends is not None:
              st.session_state.trends = trends
        else:
          st.write("No data available for the selected geohash.")

    if option == "Recommend Locations":
      cuisine_type = st.text_input("Enter the cuisine type or dish name")
      if st.button("Recommend"):
        pending_placeholder = col2.empty()
        pending_placeholder.warning("Processing...")
        sample_data = data.groupby('geohash5').apply(lambda x: x.sample(n=min(100, len(x)), random_state=42)).reset_index(drop=True)
        sample_data = sample_data.sample(n=min(1000, len(sample_data)), random_state=42)
        sample_data = sample_data[['mex_name', 'all_cuisines', 'rating', 'review_number', 'geohash5']]
        recommendations = recommend_locations(cuisine_type, sample_data)
        st.session_state.recommendations = recommendations
        pending_placeholder.empty()

      if st.session_state.recommendations is not None:
        with col2:
          st.header("Recommended Areas")
          map_center = [data['latitude'].mean(), data['longitude'].mean()]
          m = fl.Map(location=map_center, zoom_start=10)
          fl.TileLayer('cartodbpositron').add_to(m)
          recommended_geohashes = st.session_state.recommendations
          geohash_polygons = [geohash_to_polygon(gh) for gh in recommended_geohashes]
          gdf = gpd.GeoDataFrame({'geohash5': recommended_geohashes, 'geometry': geohash_polygons})
          gdf.set_crs(epsg=4326, inplace=True)
          geojson = fl.GeoJson(
            gdf,
            style_function=lambda x: {
              "fillColor": "orange",
              "color": "orange",
              "weight": 2,
              "fillOpacity": 0.5
            },
            tooltip=fl.GeoJsonTooltip(fields=['geohash5'], aliases=['Geohash:'])
          ).add_to(m)
          st_folium(m, height=250, width=700)

  if option != "Recommend Locations":
    with col2:
      st.header("AI Analysis")
      response_container = st.container()
      with response_container:
        st.markdown(
          """
          <style>
          .response-box {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #26272f;
            color: white;
          }
          </style>
          """, unsafe_allow_html=True)
        if st.session_state.trends is not None:
          st.markdown(f'<div class="response-box">{st.session_state.trends}</div>', unsafe_allow_html=True)
        elif st.session_state.prediction is not None:
          st.markdown(f'<div class="response-box">{st.session_state.prediction}</div>', unsafe_allow_html=True)
        else:
          st.markdown('<div class="response-box"></div>', unsafe_allow_html=True)
else:
  st.warning("Please upload the restaurant data for insights and recommendations.")