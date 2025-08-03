import os
import pandas as pd
import geopandas as gpd
import numpy as np
from spark_session import create_spark_session
from sedona.register import SedonaRegistrator
from pyspark.sql import functions as F
from sedona.core.spatialOperator import JoinQuery
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from folium import folium
import matplotlib.pyplot as plt

# --- 1. Set Up and Load Data ---
# Create a Spark session with Sedona enabled
spark = create_spark_session()
SedonaRegistrator.registerAll(spark)

# This is the correct, robust way to get Sedona's functions
sedona_funcs = spark.sparkContext._jvm.org.apache.sedona.sql.functions

base_path = "/Users/vigneshthiagarajan/Downloads/Personal Projects/Geospatial Clustering/Datasets/"
print("Loading datasets...")

census_tracts_gdf = gpd.read_file(os.path.join(base_path, "tl_2022_06_tract/tl_2022_06_tract.shp"))
competitors_df = pd.read_csv(os.path.join(base_path, "Amazon_Warehouse_Locations_Geocoded.csv"))

competitors_gdf = gpd.GeoDataFrame(
    competitors_df,
    geometry=gpd.points_from_xy(competitors_df['longitude'], competitors_df['latitude']),
    crs="EPSG:4326"
)

print(f"Census Tracts loaded: {len(census_tracts_gdf)} polygons")
print(f"Competitor Hubs loaded: {len(competitors_gdf)} points")

# --- 2. Feature Engineering with Sedona ---
print("\n--- Step 2: Feature Engineering with Sedona ---")

# The functions are now accessed directly from the JVM
census_df_spark = sedona_funcs.from_pandas_df(spark, census_tracts_gdf).withColumnRenamed("geometry", "geom_census").cache()
competitors_df_spark = sedona_funcs.from_pandas_df(spark, competitors_gdf).withColumnRenamed("geometry", "geom_competitor").cache()

# Population and Income Features (from Census Data)
census_df_spark = census_df_spark.withColumn("total_population", F.col("POP100").cast("int"))
census_df_spark = census_df_spark.withColumn("median_income", F.col("INCOME").cast("double"))

# Proximity to Competitors Feature
print("Calculating proximity to competitors...")
competitor_buffers = competitors_df_spark.withColumn("competitor_buffer", sedona_funcs.ST_Buffer(F.col("geom_competitor"), F.lit(0.02)))
competitor_join = JoinQuery.SpatialJoin(census_df_spark, competitor_buffers, JoinQuery.Join.Intersects, True)
competitor_join = competitor_join.select("GEOID_left").distinct().withColumnRenamed("GEOID_left", "GEOID").withColumn("has_competitor_nearby", F.lit(1))

census_df_spark = census_df_spark.join(competitor_join, on="GEOID", how="left").na.fill(0, subset=["has_competitor_nearby"]).cache()

# --- 3. Build a Supervised Machine Learning Model ---
print("\n--- Step 3: Building ML Model ---")
# Create a target variable 'is_hub'
existing_hubs_tracts = JoinQuery.SpatialJoin(census_df_spark, competitors_df_spark, JoinQuery.Join.Contains, True)
existing_hubs_tracts = existing_hubs_tracts.select("GEOID_left").distinct().withColumnRenamed("GEOID_left", "GEOID").withColumn("is_hub", F.lit(1))
final_df = census_df_spark.join(existing_hubs_tracts, on="GEOID", how="left").na.fill(0, subset=["is_hub"])

# Drop columns not needed for the model and convert to Pandas for ML
features_df = final_df.select("total_population", "median_income", "has_competitor_nearby", "is_hub").toPandas().dropna()
X = features_df[['total_population', 'median_income', 'has_competitor_nearby']]
y = features_df['is_hub']

# Train a RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("RandomForestClassifier model trained successfully.")

# Predict suitability for all locations
all_predictions = model.predict_proba(X)[:, 1]
final_df = final_df.toPandas()
final_df['suitability_score'] = all_predictions
final_df['is_hub'] = y

# --- 4. Visualize Results and Output ---
print("\n--- Step 4: Visualizing Results ---")
california_map = folium.Map(location=[36.7783, -119.4179], zoom_start=6)

def style_function(feature):
    score = feature['properties']['suitability_score']
    if score >= 0.5:
        return {'fillColor': 'green', 'color': 'black', 'weight': 0.5, 'fillOpacity': score}
    else:
        return {'fillColor': 'red', 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.3}

folium.Choropleth(
    geo_data=gpd.GeoDataFrame(final_df, geometry='geom_census', crs='EPSG:4326'),
    name='Suitability Score',
    data=final_df,
    columns=['GEOID', 'suitability_score'],
    key_on='feature.properties.GEOID',
    fill_color='RdYlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Hub Suitability Score',
).add_to(california_map)

folium.LayerControl().add_to(california_map)
california_map.save(os.path.join(base_path, 'california_hub_suitability_map.html'))
print("Interactive map saved as 'california_hub_suitability_map.html'")
print("\n--- Project Complete! ---")

spark.stop()