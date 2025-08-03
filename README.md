# Geospatial Analytics for E-commerce Hubs 

This project uses geospatial analytics and machine learning to recommend optimal locations for new e-commerce delivery hubs in California. The analysis is performed on a large-scale dataset using **Apache Sedona** on a PySpark cluster.

The goal is to provide data-driven recommendations that minimize delivery times and maximize market reach for a logistics company.

---

###  Key Features

* **Location Suitability Model**: A supervised machine learning model (`RandomForestClassifier`) is trained to identify the most suitable locations for new hubs.
* **Feature Engineering**: Geospatial features like population density and competitor proximity are generated using Apache Sedona.
* **Scalable Analytics**: The entire workflow is built to handle large, real-world datasets by leveraging the distributed computing power of Apache Spark.
* **Interactive Visualization**: The final results are presented on an interactive HTML map for easy consumption and analysis.

---

###  Technologies Used

* **Apache Sedona**: For powerful geospatial processing at scale.
* **PySpark**: The core distributed computing framework.
* **GeoPandas**: For local data handling and visualization.
* **Scikit-learn**: For building the machine learning model.
* **Folium**: For generating the interactive output map.

---

###  How to Run the Project

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/Geospatial-Clustering-Amazon-Warehouses.git](https://github.com/your-username/Geospatial-Clustering-Amazon-Warehouses.git)
    ```

2.  **Set Up the Environment**:
    It is highly recommended to use a Conda environment. This project was developed using `Python 3.10`, `PySpark 3.4.1`, and `Apache-Sedona 1.4.1`.

    ```bash
    conda create -n sedona_env python=3.10
    conda activate sedona_env
    conda install -c conda-forge pyspark=3.4.1 apache-sedona=1.4.1 geopandas folium scikit-learn -y
    ```

3.  **Data Sources**:
    The raw data files are not included in this repository due to size. You can download the necessary files and place them in a folder named `Datasets` at the project's root.
    * **Census Tracts**: [U.S. Census Bureau TIGER/Line Shapefiles](https://catalog.data.gov/dataset/tiger-line-shapefile-2022-state-california-ca-census-tract)
    * **Competitor Data**: The `Amazon_Warehouse_Locations_Geocoded.csv` file was created by manually geocoding public addresses.

4.  **Run the Script**:
    ```bash
    python main.py
    ```
    This will generate an interactive HTML map named `california_hub_suitability_map.html` in your `Datasets` folder.
