import argparse
import ee
import geopandas as gpd
import json
import logging
import os
import pandas as pd
import re
import requests
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
from pprint import pprint
from typing import Optional, Tuple


"""
This script exports annotations from an Earth System Studio (ESS) project into a local folder named 'exported_data'.
If run with the '--online-export' flag, the annotations are also uploaded to Google Cloud Storage (GCS) and Google Earth Engine (GEE).

The script performs the following steps:
1. Verifies all prerequisites, including the configuration file, environment variables, and API connectivity.
2. Retrieves and formats annotations from the ESS projects defined under the 'ess_projects' key in the 'config.json' configuration file (see 'config.example.json').
3. Exports the annotations to the 'exported_data' folder in CSV, zipped Shapefile, and GeoJSON formats.

Optional (enabled with '--online-export'):
4. Uploads the zipped Shapefiles to the GCS bucket specified by the 'gc_bucket_name' and 'gc_blob_name' keys in the configuration file.
5. Transfers the zipped Shapefiles from GCS to the Earth Engine asset defined by the 'gee_asset_id' key in the configuration file.
"""

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)


class Config:
    """
    Class responsible for loading and validating all configuration variables required to run the annotation export script. 
    Loads non-sensitive settings from 'config.json', sensitive data (e.g., API key) from environment variables in a .env file, and additional parameters from command-line arguments.
    """

    def __init__(self, online_export=False):
        
            # === 1. Variable retrieved from the command line ===

        self.online_export = online_export


            # === 2. Variable retrieved from environment variables (.env file, secret variable)

        load_dotenv()
        self.api_key = os.getenv("API_KEY")  # Earth System Studio API key

        if not self.api_key:
            raise ValueError("Missing required environment variable 'API_KEY'.")


            # === 3. Variables retrieved from the JSON config file (JSON file, non-secret variables)

        if not os.path.exists("config.json"):
            raise FileNotFoundError("Configuration file 'config.json' not found.")
        
        with open("config.json", "r", encoding="utf-8") as f:
            config_data = json.load(f)

        self.gc_project_id = config_data.get("gc_project_id", "geo-global-ecosystems-atlas")  # Google Cloud project ID, default to 'geo-global-ecosystems-atlas'. Accessible as an environement variable (GOOGLE_CLOUD_PROJECT) if the code is running on Google Cloud Run.

        self.env = config_data.get("env", "prod").lower()  # Environment: development (dev) or production (prod), default to 'prod'
        
        if self.env not in ("dev", "prod"):
            raise ValueError("The 'env' key in the config file 'config.json' must be set to 'dev' or 'prod'.")
        
        self.ess_projects = config_data.get("ess_projects")

        if not isinstance(self.ess_projects, list) or not self.ess_projects:
            raise ValueError("The config file 'config.json' must contain at least one Earth System Studio project in the 'ess_projects' list.")

        for project in self.ess_projects:
            if not isinstance(project, dict):
                raise ValueError("Each project defined in the 'ess_projects' list of the config file 'config.json' must be a dictionary.")

            if not project.get("ess_project_id"):
                raise ValueError("Each project defined in the 'ess_projects' list of the config file 'config.json' must have a set 'ess_project_id' key.")

            if online_export:
                missing_keys = [key for key in ("gc_bucket_name", "gc_blob_name", "gee_asset_id") if not project.get(key)]
                if missing_keys:
                    raise ValueError(f"Missing required keys for online export in project '{project.get('ess_project_id')}': {', '.join(missing_keys)}")


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
    
    def request_project_name(self, ess_project_id: str):
        """Fetch the project name for a specific ESS project identified by its ID."""

        url = f"{self.base_url}/api/v1/projects/{ess_project_id}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return r.json()["name"]


def reset_data_folder() -> None:
    """Delete the 'exported_data' folder and all its contents, then recreate it as an empty directory."""

    if os.path.exists("exported_data"):
        shutil.rmtree("exported_data")

    os.makedirs("exported_data")


def create_project_data_folder(ess_project_name: str) -> str:
    """Create a folder inside the 'exported_data' directory using the given ESS project name and return its path."""

    folder_path = os.path.join("exported_data", ess_project_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def get_annotations_filename(ess_project_id: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"annotations_{timestamp}_{ess_project_id}"


def restructure_annotations(ess_project_id: str, api: ESS_API) -> Tuple[dict, pd.DataFrame, gpd.GeoDataFrame]:

        # === 1. Request and validate data. ===

    ess_project_name = api.request_project_name(ess_project_id)

    raw_annotations_geojson = api.request_annotations(ess_project_id)
    if not raw_annotations_geojson or not raw_annotations_geojson.get('features'):
        raise ValueError(f"No annotations found for the project {ess_project_name} ({ess_project_id}).")

    users = api.request_users(ess_project_id)
    if not users:
        raise ValueError(f"No users found for the project {ess_project_name} ({ess_project_id}).")


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

    # Build a dataframe of annotations where each row is an annotation with 32 fields (+ an additional DataFrame with extra information, not intended for export but useful for processing).
    annotations, annotations_extra = [], []

    for feature in raw_annotations_geojson["features"]:
        if feature["geometry"]["type"] != "Point":
            continue

        annotation = {col: extract_value(feature, col) for col in annotation_fields}
        annotations.append(annotation)

        annotation_extra = {'ess_annotation_id': feature["properties"].get("id"), 'ess_task_name': feature["properties"].get("task_name"), 'ess_task_status': feature["properties"].get("task_status")}
        annotations_extra.append(annotation_extra)

    df, df_extra = pd.DataFrame(annotations), pd.DataFrame(annotations_extra)


        # === 3. Post-processing of annotations: clean up existing values and fill in missing ones. ===

    # TODO: set the grid_id for the interactive points

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
            ess_annotation_id = df.at[i, 'ess_annotation_id']
            ess_task_name = df_extra.loc[df_extra['ess_annotation_id'] == ess_annotation_id, 'ess_task_name'].values[0]
            if sum_10m[i] > 100:
                error_msg.append(f"{sum_10m[i]}% for homogeneity at 10m in annotation '{ess_annotation_id}' of task '{ess_task_name}'")
            if sum_100m[i] > 100:
                error_msg.append(f"{sum_100m[i]}% for homogeneity at 100m in annotation '{ess_annotation_id}' of task '{ess_task_name}'")

        raise ValueError(f"In project {ess_project_name}, homogeneity values exceed 100% in the following annotations:\n" + "\n".join(error_msg))


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
    metadata_report = api.request_metadata_report(ess_project_id)
    if any(d['metadata_name'] == 'sample_type' for d in metadata_report):
        df['sample_type'] = df['sample_type'].fillna("Interactive")


    # 5. Replace the interpreter and reviewer IDs with their names when possible
    id_to_name = {user['id']: user['name'] for user in users}  # Build the mapping IDs → names
    df['interpreter_name'] = df['interpreter_name'].map(id_to_name).fillna(df['interpreter_name'])
    df['reviewer_name'] = df['reviewer_name'].map(id_to_name).fillna(df['reviewer_name'])


    # 6. Set the EFG codes for dominant and secondary EFGs at 10m and 100m, and set the character strings for the dominant EFG at 100m.
    def extract_IUCN_codes(input_str: str) -> Optional[Tuple[str, str, str]]:
        """Analyse a character string and identify whether an EFG code is present. If so, return a tuple of three values: the realm code, biome code, and EFG code. Otherwise, return None."""
        
        codes = None

        if input_str:
            match = re.match(r'^([A-Z]{1,3})_(\d{1,2})_(\d{1,2})', input_str.strip(), re.IGNORECASE)  # Match pattern: 1-3 letters + "_" + 1-2 digits + "_" + 1-2 digits
            if match:
                realm, biome, efg = match.groups()
                realm = realm.upper()
                codes = (realm, f"{realm}{biome}", f"{realm}{biome}.{efg}")
        
        return codes

    def load_IUCN_codes_dict(excel_path, sheet_name=0):
        """Open an .xlsx file and returns a dictionary mapping 'code' to 'name', if both columns are present."""

        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        if 'code' not in df.columns or 'name' not in df.columns:
            raise ValueError(f"Missing 'code' or 'name' columns in {excel_path}")
        
        return dict(zip(df['code'], df['name']))

    dict_realms = load_IUCN_codes_dict(os.path.join("resources", "IUCN-GET-realms.xlsx"))  # Would be nice if this information could be retrieved from a GET site API
    dict_biomes = load_IUCN_codes_dict(os.path.join("resources", "IUCN-GET-biomes.xlsx"))
    dict_efgs = load_IUCN_codes_dict(os.path.join("resources", "IUCN-GET-profiles-exported-2023-06-14.xlsx"), sheet_name="Short description")

    # Set the realm, biome, and EFG names for the dominant EFG at 100m.
    df[['iucn_realm', 'iucn_biome', 'iucn_ecosystem_functional_group']] = df['iucn_efg_code_dominant_100m'].apply(  # TODO: Check that it's for the code at 100m and not 10m
        lambda v: pd.Series((
            dict_realms.get(codes[0]) if (codes := extract_IUCN_codes(v)) else None,
            dict_biomes.get(codes[1]) if codes else None,
            dict_efgs.get(codes[2]) if codes else None
        ))
    )
    #TODO: remove code at start of the string

    # Set the EFG codes for dominant and secondary values at 10m and 100m.
    cols_to_update = ['iucn_efg_code_dominant_10m', 'iucn_efg_code_secondary_10m', 'iucn_efg_code_dominant_100m', 'iucn_efg_code_secondary_100m']
    for col in cols_to_update:
        df[col] = df[col].apply(lambda v: codes[2] if (codes := extract_IUCN_codes(v)) else v)


        # === 4. Return the values. ===

    # Geospatial annotations DataFrame
    gdf = df.rename(columns=annotation_fields)  # Rename GeoDataFrame columns with short names for safe export to shapefile
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    gdf = gdf[[*gdf.columns[:3], "geometry", *gdf.columns[3:-1]]]  # Reorder columns (geometry to 4th position)
    # TODO: 2 geo df, one with full labels for geojson and one with short labels for shp

    return raw_annotations_geojson, df, gdf


def upload_to_bucket(source_file_name: str, destination_bucket_name: str, destination_blob_name: str) -> None:
    """Upload a file to the specified Cloud Storage bucket."""

    client = storage.Client()
    bucket = client.bucket(destination_bucket_name)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        logging.info(f'The blob {destination_bucket_name}/{destination_blob_name} already existed and has been overwritten.')
    
    blob.upload_from_filename(source_file_name)

    logging.info(f'Uploaded {source_file_name} to gs://{destination_bucket_name}/{destination_blob_name}.')


def upload_to_asset(gc_project_id: str, source_gc_bucket_name: str, source_gc_blob_name: str, destination_gee_asset_id: str) -> None:
    """ Transfer a file from a Cloud Storage bucket to the specified Cloud Earth Engine asset. """

    # Initialise the Earth Engine project
    ee.Authenticate()
    ee.Initialize(project=gc_project_id)

    # Delete the old Earth Engine asset if it exists
    try:
        info = ee.data.getInfo(destination_gee_asset_id)
        if info:
            ee.data.deleteAsset(destination_gee_asset_id)
            logging.info(f'The asset {destination_gee_asset_id} already existed and has been overwritten.')
    except Exception as e:
        logging.error(f'Error checking asset: {e}')

    # Build the Earth Engine CLI command to upload the zipped shapefile from the bucket to the Earth Engine assets folder
    cmd = ['earthengine', 'upload', 'table', f'--asset_id={destination_gee_asset_id}', f'gs://{source_gc_bucket_name}/{source_gc_blob_name}.zip']

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f'Upload to Earth Engine assets folder started successfully: {result.stdout.strip()}')
    except subprocess.CalledProcessError as e:
        logging.error(f'Error uploading to Earth Engine assets folder: {e.stderr.strip()}')


def main(config: Config, online_export: bool):

        # === 1. Set up and test the API connection, and reset the data folder. ===

    api = ESS_API(config)
    api.test_connection()
    reset_data_folder()


    # Loop through all ESS projects specified in the configuration file 'config.json'.
    for i, ess_project in enumerate(config.ess_projects, start=1):


            # === 2. Retrieve and format the annotations. ===

        ess_project_id = ess_project.get('ess_project_id')
        ess_project_name = api.request_project_name(ess_project_id)

        raw_annotations_geojson, df, gdf = restructure_annotations(ess_project_id, api)


            # === 3. Save the annotations locally in the 'exported_data' folder as CSV, zipped Shapefile, and GeoJSON formats. Also save the raw GeoJSON as received from the API. ===

        annotations_filename = get_annotations_filename(ess_project_id)
        folder_path = create_project_data_folder(ess_project_name)

        # Raw GeoJSON
        with open(os.path.join(folder_path, f"raw_{annotations_filename}.geojson"), "w", encoding="utf-8") as f:
            json.dump(raw_annotations_geojson, f, indent=2)

        # CSV
        df.to_csv(os.path.join(folder_path, f"{annotations_filename}.csv"), index=False)
        
        # GeoJSON
        gdf.to_file(os.path.join(folder_path, f"{annotations_filename}.geojson"), driver="GeoJSON")

        # Zipped Shapefile
        gdf.to_file(os.path.join(folder_path, f"{annotations_filename}.shp"), driver="ESRI Shapefile")
        with zipfile.ZipFile(os.path.join(folder_path, f"{annotations_filename}.zip"), "w") as zipf:
            for ext in [".cpg", ".dbf", ".prj", ".shp", ".shx"]:  # List of shapefile extensions to include
                file_path = os.path.join(folder_path, f"{annotations_filename}{ext}")
                if os.path.exists(file_path):
                    zipf.write(file_path, arcname=f"{annotations_filename}{ext}")
                    os.remove(file_path)
                else:
                    logging.warning(f"{file_path} not found, skipping.")


            # 4. === (Optional) Upload the annotations to Google Cloud and Earth Engine Assets. ===

        if online_export:

            gc_bucket_name, gc_blob_name, gee_asset_id = ess_project.get('gc_bucket_name'), ess_project.get('gc_blob_name'), ess_project.get('gee_asset_id')
            upload_to_bucket(os.path.join(folder_path, f'{annotations_filename}.zip'), gc_bucket_name, f'{gc_blob_name}.zip')
            upload_to_asset(config.gc_project_id, gc_bucket_name, gc_blob_name, gee_asset_id)


    msg = f"Script executed successfully with {i} project{'s' if i != 1 else ''} exported."
    if online_export:
        msg += " Please check the 'Tasks' section in the Google Earth Engine Code Editor for more details."
    logging.info(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for exporting annotations from an Earth System Studio project.")
    parser.add_argument("--online-export", action="store_true", help="Exports the annotations to Google Cloud and Google Earth Engine Assets.")
    args = parser.parse_args()

    config = Config(online_export=args.online_export)

    main(config, online_export=args.online_export)