import ee
import geopandas as gpd
import json
import logging
import os
import pandas as pd
import requests
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
from pprint import pprint
from shapely.geometry import Point
from typing import Tuple
#TODO: clean imports


# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)


class Config:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_KEY") # Earth System Studio API key
        self.blob_name = os.getenv("BLOB_NAME") # Google Cloud Storage blob name
        self.bucket_name = os.getenv("BUCKET_NAME") # Google Cloud Storage bucket name
        self.env = os.getenv("ENV", "prod").lower() # Environment: development (dev) or production (prod), default to 'prod'
        self.ess_project_id = os.getenv("ESS_PROJECT_ID") # Earth System Studio project ID
        self.gc_project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "geo-global-ecosystems-atlas") # Google Cloud project ID, default to 'geo-global-ecosystems-atlas'. Accessible if the code is running on Google Cloud Run.
        self.gee_asset_id = os.getenv("GEE_ASSET_ID") # Google Earth Engine asset ID
        self.validate()

    def validate(self):
        missing = [name for name, value in vars(self).items() if value in (None, "")]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}.")

        if self.env not in ("dev", "prod"):
            raise ValueError("The environment variable ENV must be set to 'dev' or 'prod'.")


class ESS_API:
    def __init__(self, config: Config) -> None:
        self.base_url: str = "https://earth-system-studio.allen.ai"
        self.api_key: str = config.api_key
        self.headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def test_connection(self) -> None:
        """Check if the API is reachable (note: this does not verify if the API key is valid)."""
        try:
            r = requests.get(f"{self.base_url}/", headers=self.headers, timeout=5)
            r.raise_for_status()
            logging.info("API connection successful.")
        except requests.RequestException as e:
            logging.error(f"API connection failed: {e}")
            sys.exit(1)

    def get_annotations(self, ess_project_id: str):
        """Fetch all annotations for a specific ESS project identified by its ID."""
        url = f"{self.base_url}/api/v1/projects/{ess_project_id}/annotations"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return r.json()
    
    def get_users(self, ess_project_id: str):
        """Fetch all users for a specific ESS project identified by its ID."""
        url = f"{self.base_url}/api/v1/projects/{ess_project_id}/users"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return r.json()


def reset_data_folder() -> None:
    """Delete the 'data' folder and all its contents, then recreate it as an empty directory."""
    if os.path.exists("exported_data"):
        shutil.rmtree("exported_data")
    os.makedirs("exported_data")


def get_annotations_filename(config: Config) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"annotations_{timestamp}_{config.ess_project_id}"


def restructure_annotations(geojson, config: Config) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:

    annotation_fields  = {
        "date": "date",
        "sample_id": "sample_id",
        "reference_grid_id": "grid_id",
        "latitude": "latitude",
        "longitude": "longitude",
        "method": "method",
        "interpretation_scale_min": "int_min",
        "interpretation_scale_max": "int_max",
        "iucn_realm": "iucn_realm",
        "iucn_biome": "iucn_biome",
        "iucn_ecosystem_functional_group": "iucn_efg",
        "iucn_efg_code_dominant_10m": "efg_10m",
        "iucn_efg_code_secondary_10m": "efg2_10m",
        "iucn_efg_code_dominant_100m": "efg_100m",
        "iucn_efg_code_secondary_100m": "efg2_100m",
        "source": "source",
        "data_producer": "producer",
        "interpreter_name": "int_name",
        "interpreter_confidence_level": "int_conf",
        "reviewer_name": "rev_name",
        "reviewer_confidence_level": "rev_conf",
        "sample_type": "samp_type",
        "valid_year_start": "year_start",
        "valid_year_end": "year_end",
        "homogeneity_estimate_dominant_10m": "hom_10m",
        "homogeneity_estimate_secondary_10m": "hom2_10m",
        "homogeneity_estimate_dominant_100m": "hom_100m",
        "homogeneity_estimate_secondary_100m": "hom2_100m",
        "interpreter_comment": "int_com",
        "reviewer_comment": "rev_com",
        "ess_task_id": "task_id",
        "ess_annotation_id": "annot_id"
    }

    # Field matchers for metadata_values (keys are column names in final dataframe)
    metadata_mapping = {
        "iucn_efg_code_secondary_10m": ["iucn_efg_code_secondary_10m"],
        "iucn_efg_code_dominant_100m": ["iucn_efg_code_dominant_100m", "iucn_efg_code_100m"],
        "iucn_efg_code_secondary_100m": ["iucn_efg_code_secondary_100m"],
        "interpreter_confidence_level": ["interpreter_confidence_level", "interpreter_confidence (1-5)"],
        "reviewer_confidence_level": ["reviewer_confidence_level", "reviewer_confidence (1-5)"],
        "homogeneity_estimate_dominant_10m": ["homogeneity_estimate_dominant_10m", "homogeneity_estimate_10m"],
        "homogeneity_estimate_secondary_10m": ["homogeneity_estimate_secondary_10m"],
        "homogeneity_estimate_dominant_100m": ["homogeneity_estimate_dominant_100m"],
        "homogeneity_estimate_secondary_100m": ["homogeneity_estimate_secondary_100m"],
        "interpreter_comment": ["interpreter_comment"],
        "reviewer_comment": ["reviewer_comment"],
    }

    # Direct mapping (from GeoJSON properties or structure)
    direct_mapping = {
        "latitude": lambda f: f["geometry"]["coordinates"][1],
        "longitude": lambda f: f["geometry"]["coordinates"][0],
        "iucn_efg_code_dominant_10m": lambda f: f["properties"].get("tag_name"),
        "interpreter_name": lambda f: f["properties"].get("annotator_id"),
        "valid_year_start": lambda f: f["properties"].get("start_time"),
        "valid_year_end": lambda f: f["properties"].get("end_time"),
        "ess_task_id": lambda f: f["properties"].get("task_id"),
        "ess_annotation_id": lambda f: f["properties"].get("id"),
        "reviewer_name": lambda f: f["properties"].get("reviewer_id"),
    }

    # Combine both mappings
    def extract_value(feature, column: str):
        if column in direct_mapping:
            return direct_mapping[column](feature)
        if column in metadata_mapping:
            meta = feature["properties"].get("metadata_values", [])
            for candidate in metadata_mapping[column]:
                for item in meta:
                    if item["name"] == candidate:
                        return item.get("value")
        return None

    # Build rows for DataFrame
    rows = []
    for feature in geojson["features"]:
        if feature["geometry"]["type"] != "Point":
            continue
        row = {col: extract_value(feature, col) for col in annotation_fields}
        rows.append(row)

    #TODO: post process the dict (or df)

    df = pd.DataFrame(rows)

    gdf_shp = df.rename(columns=annotation_fields)
    geometry = gpd.points_from_xy(gdf_shp["longitude"], gdf_shp["latitude"])
    gdf_shp = gpd.GeoDataFrame(gdf_shp.drop(columns=["latitude", "longitude"]), geometry=geometry)
    gdf_shp.set_crs('EPSG:4326', inplace=True)
    
    return df, gdf_shp


def upload_to_bucket(source_file_name, destination_bucket_name, destination_blob_name):
    """Uploads a file to the specified Cloud Storage bucket."""

    client = storage.Client()
    bucket = client.bucket(destination_bucket_name)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        logging.info(f'The blob {destination_bucket_name}/{destination_blob_name} already existed and has been overwritten.')
    
    blob.upload_from_filename(source_file_name)

    logging.info(f'Uploaded {source_file_name} to gs://{destination_bucket_name}/{destination_blob_name}.')


def main(config: Config):

    # 1. setup and test connection to the API
    api = ESS_API(config)
    api.test_connection()

    # 2. fetch and format data
    annotations_json = api.get_annotations(config.ess_project_id)
    gdf, gdf_shp = restructure_annotations(annotations_json, config)

    # 3. save data in local in csv and shapefile (zipped) formats. save also the raw geojson
    reset_data_folder()
    annotations_filename = get_annotations_filename(config)
    filepath = os.path.join("exported_data", f"raw_{annotations_filename}.geojson")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(annotations_json, f, indent=2)
    gdf.to_csv(os.path.join("exported_data", f"{annotations_filename}.csv"), index=False)
    gdf_shp.to_file(os.path.join("exported_data", f"{annotations_filename}.shp"))

    # Bundle all shapefile components into a zip file
    with zipfile.ZipFile(os.path.join("exported_data", f"{annotations_filename}.zip"), "w") as zipf:
        for ext in [".cpg", ".dbf", ".prj", ".shp", ".shx"]: # List of shapefile extensions to include
            filepath = os.path.join("exported_data", f"{annotations_filename}{ext}")
            if os.path.exists(filepath):
                zipf.write(filepath, arcname=f"{annotations_filename}{ext}")
                os.remove(filepath)
            else:
                logging.warning(f"{filepath} not found, skipping.")

    # 4. (optionnal) save data on google cloud and google earth engine assets
    # TODO: from here, clean this into functions and make it optionnal as an option in the command line
    
    
    # upload to GC
    upload_to_bucket(os.path.join("exported_data", f'{annotations_filename}.zip'), config.bucket_name, f'{config.blob_name}.zip')


    # upload to EE

    ee.Authenticate()
    ee.Initialize(project=config.gc_project_id)

    # Delete the old Earth Engine asset if it exists
    try:
        info = ee.data.getInfo(config.gee_asset_id)
        if info:
            ee.data.deleteAsset(config.gee_asset_id)
            logging.info(f'The asset {config.gee_asset_id} already existed and has been overwritten.')
    except Exception as e:
        logging.error(f'Error checking asset: {e}')

    # Build the Earth Engine CLI command to upload the zipped shapefile from the bucket to the Earth Engine assets folder
    cmd = ['earthengine', 'upload', 'table', f'--asset_id={config.gee_asset_id}', f'gs://{config.bucket_name}/{config.blob_name}.zip']

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f'Upload to Earth Engine assets folder started successfully: {result.stdout.strip()}')
    except subprocess.CalledProcessError as e:
        logging.error(f'Error uploading to Earth Engine assets folder: {e.stderr.strip()}')
 
    return 'Script executed successfully. Please check the Tasks section in the Google Earth Engine Code Editor for more details.', 200


if __name__ == "__main__":
    config = Config()
    main(config)

