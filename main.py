import argparse
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
import zipfile
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
from pprint import pprint
from typing import Tuple


"""
This script exports annotations from an Earth System Studio (ESS) project into a local folder named 'exported_data'.
If run with the '--online-export' flag, the annotations are also uploaded to Google Cloud Storage (GCS) and Google Earth Engine (GEE).

The script performs the following steps:
1. Verifies all prerequisites, including required environment variables and API connectivity.
2. Retrieves and formats annotations from the ESS project defined by the 'ESS_PROJECT_ID' environment variable.
3. Exports the annotations to the 'exported_data' folder in CSV, zipped Shapefile, and GeoJSON formats.

Optional (enabled with '--online-export'):
4. Uploads the zipped Shapefile to the GCS bucket specified by the 'BUCKET_NAME' and 'BLOB_NAME' environment variables.
5. Transfers the zipped Shapefile from GCS to the Earth Engine asset defined by the 'GEE_ASSET_ID' environment variable.
"""

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)


class Config:
    """
    Class responsible for loading and validating all required environment variables needed to execute the annotation export script. 
    It ensures that the configuration is complete and that optional variables are only validated when necessary (for online export).
    """

    def __init__(self, online_export=False):
        load_dotenv()
        self.online_export = online_export

        # Required environment variables
        self.api_key = os.getenv("API_KEY") # Earth System Studio API key
        self.ess_project_id = os.getenv("ESS_PROJECT_ID") # Earth System Studio project ID
        self.env = os.getenv("ENV", "prod").lower() # Environment: development (dev) or production (prod), default to 'prod'

        # Optional environment variables
        self.blob_name = os.getenv("BLOB_NAME") # Google Cloud Storage blob name
        self.bucket_name = os.getenv("BUCKET_NAME") # Google Cloud Storage bucket name
        self.gc_project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "geo-global-ecosystems-atlas") # Google Cloud project ID, default to 'geo-global-ecosystems-atlas'. Accessible if the code is running on Google Cloud Run.
        self.gee_asset_id = os.getenv("GEE_ASSET_ID") # Google Earth Engine asset ID

        self.validate()

    def validate(self):
        # Required environment variables
        required_vars = ["api_key", "ess_project_id", "env"]

        # Optional environment variables (required if online_export == True)
        if self.online_export:
            required_vars += ["blob_name", "bucket_name", "gc_project_id", "gee_asset_id"]

        missing = [name for name in required_vars if getattr(self, name) in (None, "")]

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

    def request_annotations(self, ess_project_id: str):
        """Fetch all annotations for a specific ESS project identified by its ID."""

        url = f"{self.base_url}/api/v1/projects/{ess_project_id}/annotations"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return r.json()
    
    def request_users(self, ess_project_id: str) -> list[dict]:
        """Fetch all users for a specific ESS project identified by its ID."""

        url_admins = f"{self.base_url}/api/v1/projects/{ess_project_id}/admins"
        r_admins  = requests.get(url_admins, headers=self.headers)
        r_admins.raise_for_status()
        admins = r_admins.json()

        url_users = f"{self.base_url}/api/v1/projects/{ess_project_id}/users"
        r_users = requests.get(url_users, headers=self.headers)
        r_users.raise_for_status()
        users = r_users.json()

        # Check for duplicates
        combined = admins + users
        unique_users = list({user["id"]: user for user in combined}.values())

        return unique_users
    
    def request_metadata_report(self, ess_project_id: str):
        """Provides a report of metadata annotations for a specific ESS project identified by its ID. Note: Only displays metadata fields that are assigned to at least one annotation."""

        url = f"{self.base_url}/api/v1/projects/{ess_project_id}/metadata-report"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return r.json()


def reset_data_folder() -> None:
    """Delete the 'exported_data' folder and all its contents, then recreate it as an empty directory."""

    if os.path.exists("exported_data"):
        shutil.rmtree("exported_data")
    os.makedirs("exported_data")


def get_annotations_filename(config: Config) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"annotations_{timestamp}_{config.ess_project_id}"


def restructure_annotations(config: Config, api: ESS_API) -> Tuple[dict, pd.DataFrame, gpd.GeoDataFrame]:

        # === 1. Request and validate data. ===

    raw_annotations_geojson = api.request_annotations(config.ess_project_id)
    if not raw_annotations_geojson or not raw_annotations_geojson.get('features'):
        raise ValueError(f"No annotations found for the project {config.ess_project_id}.")

    users = api.request_users(config.ess_project_id)
    if not users:
        raise ValueError(f"No users found for the project {config.ess_project_id}.")


        # === 2. Map the fields from the Global Ecosystems Atlas (GEA) specification to the Earth System Studio (ESS) annotation attributes and create a dictionary where each item represents an annotation with 32 fields. ===

    # List of the 32 fields associated with each annotation as defined in the document 'Global Ecosystems Atlas Training Dataset: Design and Specification'. Key = long name, value = short name (for shapefile).
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
        "data_provider": "provider",  # Called "source" in the data specification, to be changed (in the data spec)
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

    # In Earth System Studio, an annotation has both fixed, non-editable attributes and dynamic, user-defined attributes called metadata. Metadata are stored in the 'metadata_values' attribute of an annotation.
    # Since metadata fields in ESS are dynamic and user-defined for each project, different metadata names in different projects can refer to the same field in the GEA specification.
    # For example, the metadata attributes 'iucn_efg_code_dominant_100m' and 'iucn_efg_code_100m' in two different ESS projects may both correspond to the same GEA specification field: 'iucn_efg_code_dominant_100m'.

    # Mapping between the fields in the GEA specification and the metadata attributes of an annotation. Key = field from the specification, value = corresponding metadata attribute of an annotation in Earth System Studio.
    metadata_mapping = {
        "sample_id": ["sample_id"],
        "reference_grid_id": ["grid_id"],
        #"iucn_efg_code_dominant_10m": ["iucn_efg_code_dominant_10m", "iucn_efg_code_10m"],
        "iucn_efg_code_secondary_10m": ["iucn_efg_code_secondary_10m"],
        "iucn_efg_code_dominant_100m": ["iucn_efg_code_dominant_100m", "iucn_efg_code_100m", "Iucn_efg_code_100m"],
        "iucn_efg_code_secondary_100m": ["iucn_efg_code_secondary_100m"],
        "interpreter_confidence_level": ["interpreter_confidence_level", "interpreter_confidence (1-5)"],
        "reviewer_confidence_level": ["reviewer_confidence_level", "reviewer_confidence (1-5)"],
        "sample_type": ["sample_type"],
        "homogeneity_estimate_dominant_10m": ["homogeneity_estimate_dominant_10m", "homogeneity_estimate_10m"],
        "homogeneity_estimate_secondary_10m": ["homogeneity_estimate_secondary_10m"],
        "homogeneity_estimate_dominant_100m": ["homogeneity_estimate_dominant_100m", "homogeneity_estimate_100m"],
        "homogeneity_estimate_secondary_100m": ["homogeneity_estimate_secondary_100m"],
        "interpreter_comment": ["interpreter_comment"],
        "reviewer_comment": ["reviewer_comment"],
    }

    # Mapping between the fields in the GEA specification and the direct attributes of an annotation. Key = field from the specification, value = corresponding direct attribute of an annotation in Earth System Studio.
    direct_mapping = {
        "date": lambda f: f["properties"].get("updated_time"),
        "latitude": lambda f: round(f["geometry"]["coordinates"][1], 6),  # Rounding to 6 decimal places ensures stability up to ~10 cm precision, which is usually sufficient for geographic data.
        "longitude": lambda f: round(f["geometry"]["coordinates"][0], 6),
        "iucn_efg_code_dominant_10m": lambda f: f["properties"].get("tag_display_name"),
        "interpreter_name": lambda f: f["properties"].get("annotator_id"),
        "reviewer_name": lambda f: f["properties"].get("reviewer_id"),
        "valid_year_start": lambda f: f["properties"].get("start_time"),
        "valid_year_end": lambda f: f["properties"].get("end_time"),
        "ess_task_id": lambda f: f["properties"].get("task_id"),
        "ess_annotation_id": lambda f: f["properties"].get("id"),
    }

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

    # Build a dictionary of annotations where each element is an annotation with 32 fields.
    annotations = []
    for feature in raw_annotations_geojson["features"]:
        if feature["geometry"]["type"] != "Point":
            continue
        annotation = {col: extract_value(feature, col) for col in annotation_fields}
        annotations.append(annotation)


        # === 3. Post-processing of annotations: clean up existing values and fill in missing ones. ===

    df = pd.DataFrame(annotations)

    # 1. Clean column types and formats
    columns_to_int = ['interpreter_confidence_level', 'reviewer_confidence_level']
    df[columns_to_int] = df[columns_to_int].astype('UInt8')  # No need for extra checks here, confidence levels in ESS are supposed to be restricted to values between 1 and 5.
    
    columns_to_float = ['homogeneity_estimate_dominant_10m', 'homogeneity_estimate_secondary_10m', 'homogeneity_estimate_dominant_100m', 'homogeneity_estimate_secondary_100m']
    df[columns_to_float] = df[columns_to_float].astype("Float32")  # No need for extra checks here, homogeneity values in ESS are supposed to be restricted to between 0 and 100.

    columns_to_date = ["date", "valid_year_start", "valid_year_end"]
    df[columns_to_date] = df[columns_to_date].apply(lambda col: pd.to_datetime(col, format='mixed', utc=True).dt.strftime("%d-%m-%Y"))


    # 2. Check for possible data entry errors where homogeneity estimates exceed 100% when adding dominant and secondary EFG.
    sum_10m = df[['homogeneity_estimate_dominant_10m','homogeneity_estimate_secondary_10m']].sum(axis=1, skipna=True)
    sum_100m = df[['homogeneity_estimate_dominant_100m','homogeneity_estimate_secondary_100m']].sum(axis=1, skipna=True)
    invalid_homogeneities = (sum_10m > 100) | (sum_100m > 100)

    if invalid_homogeneities.any():
        error_msg = []
        for i in df[invalid_homogeneities].index:
            if sum_10m[i] > 100:
                error_msg.append(f"{sum_10m[i]}% for homogeneity at 10m in annotation '{df.at[i, 'ess_annotation_id']}' of task '{df.at[i, 'ess_task_id']}'")
            if sum_100m[i] > 100:
                error_msg.append(f"{sum_100m[i]}% for homogeneity at 100m in annotation '{df.at[i, 'ess_annotation_id']}' of task '{df.at[i, 'ess_task_id']}'")
        raise ValueError("Homogeneity values exceed 100% in the following annotations:\n" + "\n".join(error_msg))


    # 3. Fill columns with constant values
    constant_values = {
        "method": "Image interpretation",  # May vary (e.g., field data), but if exported from ESS, it's almost certainly "Image interpretation".
        "interpretation_scale_min": 10,
        "interpretation_scale_max": 100,
        "data_provider": "Global Ecosystem Atlas",
        "data_producer": "JCU Global Ecology Lab"  # May vary (e.g., country mapping team or expert's institution), but if exported from ESS, it's almost certainly "JCU Global Ecology Lab".
    }

    for field, value in constant_values.items():
        df[field] = value


    # 4. Set sample type to "Interactive" for projects where the 'sample_type' metadata is present (do not fill anything for projects where the metadata is not present [Indo-Pacific Attols, Antarctic, Arctic]).
    metadata_report = api.request_metadata_report(config.ess_project_id)
    if any(d['metadata_name'] == 'sample_type' for d in metadata_report):
        df['sample_type'] = df['sample_type'].fillna("Interactive")


    # 5. Replace the interpreter and reviewer IDs with their names when possible
    id_to_name = {user['id']: user['name'] for user in users}  # Build the mapping IDs → names
    df['interpreter_name'] = df['interpreter_name'].map(id_to_name).fillna(df['interpreter_name'])
    df['reviewer_name'] = df['reviewer_name'].map(id_to_name).fillna(df['reviewer_name'])

    # TODO: other columns to clean:

    # iucn_efg_code_[dominant|secondary]_10[0]m

    # iucn_realm, iucn_biome, iucn_efg


        # === 4. Return the values. ===

    # Geospatial annotations DataFrame
    gdf = df.rename(columns=annotation_fields)  # Rename GeoDataFrame columns with short names for safe export to shapefile
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    gdf = gdf[[*gdf.columns[:3], "geometry", *gdf.columns[3:-1]]]  # Reorder columns (geometry to 4th position)

    return raw_annotations_geojson, df, gdf


def upload_to_bucket(source_file_name, destination_bucket_name, destination_blob_name):
    """Uploads a file to the specified Cloud Storage bucket."""

    client = storage.Client()
    bucket = client.bucket(destination_bucket_name)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        logging.info(f'The blob {destination_bucket_name}/{destination_blob_name} already existed and has been overwritten.')
    
    blob.upload_from_filename(source_file_name)

    logging.info(f'Uploaded {source_file_name} to gs://{destination_bucket_name}/{destination_blob_name}.')


def main(config: Config, online_export: bool):

        # === 1. Set up and test the connection to the API. ===

    api = ESS_API(config)
    api.test_connection()


        # === 2. Retrieve and format the annotations. ===

    raw_annotations_geojson, df, gdf = restructure_annotations(config, api)


        # === 3. Save the annotations locally in the 'exported_data' folder as CSV, zipped Shapefile, and GeoJSON formats. Also save the raw GeoJSON as received from the API. ===

    reset_data_folder()
    annotations_filename = get_annotations_filename(config)

    # Raw GeoJSON
    filepath = os.path.join("exported_data", f"raw_{annotations_filename}.geojson")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(raw_annotations_geojson, f, indent=2)

    # CSV
    df.to_csv(os.path.join("exported_data", f"{annotations_filename}.csv"), index=False)
    
    # GeoJSON
    gdf.to_file(os.path.join("exported_data", f"{annotations_filename}.geojson"), driver="GeoJSON")

    # Zipped Shapefile
    gdf.to_file(os.path.join("exported_data", f"{annotations_filename}.shp"), driver="ESRI Shapefile")
    with zipfile.ZipFile(os.path.join("exported_data", f"{annotations_filename}.zip"), "w") as zipf:
        for ext in [".cpg", ".dbf", ".prj", ".shp", ".shx"]:  # List of shapefile extensions to include
            filepath = os.path.join("exported_data", f"{annotations_filename}{ext}")
            if os.path.exists(filepath):
                zipf.write(filepath, arcname=f"{annotations_filename}{ext}")
                os.remove(filepath)
            else:
                logging.warning(f"{filepath} not found, skipping.")


        # 4. === (Optional) Upload the annotations to Google Cloud and Earth Engine Assets using environment variable settings. ===

    if online_export:

        # upload_to_GC_bucket
        upload_to_bucket(os.path.join("exported_data", f'{annotations_filename}.zip'), config.bucket_name, f'{config.blob_name}.zip')

        # upload_to_EE_assets  # TODO: put EE export in a function
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
    
    logging.info('Script executed successfully. Please check the Tasks section in the Google Earth Engine Code Editor for more details.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for exporting annotations from an Earth System Studio project.")
    parser.add_argument("--online-export", action="store_true", help="Exports the annotations to Google Cloud and Google Earth Engine Assets.")
    args = parser.parse_args()

    config = Config(online_export=args.online_export)

    main(config, online_export=args.online_export)