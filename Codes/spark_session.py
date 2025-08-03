import os
from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer

def create_spark_session(app_name="GeospatialProject"):
    """
    Creates and returns a SparkSession with Sedona extensions.
    """
    os.environ['JAVA_HOME'] = "/opt/homebrew/opt/openjdk@17"
    os.environ['SPARK_HOME'] = "/Users/vigneshthiagarajan/Downloads/Personal Projects/Geospatial Clustering/geospatial_project_env/lib/python3.12/site-packages/pyspark"
    
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.serializer", KryoSerializer.getName)
        .config("spark.kryo.registrator", SedonaKryoRegistrator.getName)
        # --- NEW: Use Sedona JAR for Spark 3.5+ (Scala 2.12) ---
        .config("spark.jars.packages",
                "org.apache.sedona:sedona-python-adapter-3.0_2.12:1.4.1,"
                "org.datasyslab:geotools-wrapper:1.4.0-28.2")
        .getOrCreate()
    )

    SedonaRegistrator.registerAll(spark)
    print("Spark session with Sedona is ready.")
    return spark

if __name__ == '__main__':
    spark = create_spark_session()
    spark.stop()