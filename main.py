import ee
import functions_framework
import geopandas as gpd
import logging
import os
import pandas as pd
import requests
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dotenv import load_dotenv
from google.cloud import storage


# This script exports annotations from an Earth System Studio (ESS) project to a Google Earth Engine (GEE) assets folder via Google Cloud Storage (GCS).
# The script performs the following steps:
# 1. Calls the ESS API and retrieves all annotations from the ESS project defined by the ESS_PROJECT_ID environment variable, then formats them into a GeoDataFrame.
# 2. Creates a shapefile from the GeoDataFrame and compresses it into a ZIP file.
# 3. Uploads the ZIP file to the Google Cloud Storage bucket specified by the BUCKET_NAME and BLOB_NAME environment variables.
# 4. Uploads the ZIP file from the bucket to the Earth Engine asset defined by the GEE_ASSET_ID environment variable.
# 
# Environment variables are defined either:
# - In a `.env` file for the development environment (local), or
# - In the "Edit & deploy new revision / Container(s) / Variables & Secrets" section for the production environment (Cloud Run).


# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Definition of global variables
load_dotenv()
gee_asset_id = os.getenv('GEE_ASSET_ID') # Google Earth Engine asset ID
blob_name = os.getenv('BLOB_NAME') # Google Cloud Storage blob name
bucket_name = os.getenv('BUCKET_NAME') # Google Cloud Storage bucket name
env = os.getenv('ENV', 'prod').lower() # Environment: development (dev) or production (prod), default to 'prod'
ess_project_id = os.getenv('ESS_PROJECT_ID') # Earth System Studio project ID
gc_project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'geo-global-ecosystems-atlas') # Google Cloud project ID, default to 'geo-global-ecosystems-atlas'
api_key = os.getenv('API_KEY') # Earth System Studio API key

# Check for missing required variables
required_vars = {
    'GEE_ASSET_ID': gee_asset_id,
    'BLOB_NAME': blob_name,
    'BUCKET_NAME': bucket_name,
    'ESS_PROJECT_ID': ess_project_id,
    'API_KEY': api_key
}

missing = [name for name, value in required_vars.items() if not value]

if missing:
    logging.error(f'Missing required environment variables: {", ".join(missing)}')
    sys.exit(1)


def get_gdf_annotations():

    logging.info('Starting request to the Earth System Studio API')

    r = requests.get(f'https://earth-system-studio.allen.ai/api/v1/projects/{ess_project_id}/annotations', headers={'Authorization': api_key})
    r.raise_for_status()

    gdfs = []
    geojson_data = r.json()

    if 'features' in geojson_data and geojson_data['features']:
        
        # Flatten the metadata fields
        for feature in geojson_data['features']:
            metadatas = feature['properties'].pop('metadata_values', [])
            for metadata in metadatas:
                feature['properties'][metadata['name']] = metadata['value']

        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
        gdfs.append(gdf)

    gdf = pd.concat(gdfs, ignore_index=True)

    gdf.set_crs('EPSG:4326', inplace=True)

    expected_cols = ['id', gdf.geometry.name, 'start_time', 'end_time', 'tag_name', 'tag_color', 'iucn_efg_code_10m', 'iucn_efg_code_100m', 'homogeneity_estimate_10m', 'homogeneity_estimate_100m', 'interpreter_confidence (1-5)']
    existing_cols = [col for col in expected_cols if col in gdf.columns]
    missing_cols = [col for col in expected_cols if col not in gdf.columns]

    if missing_cols:
        logging.warning(f'Missing columns: {missing_cols}')

    # Safely select only the existing columns
    gdf = gdf[existing_cols].copy()

    rename_dict = {
        'tag_name': 'efg_name',
        'tag_color': 'efg_colour',
        'iucn_efg_code_10m': 'code_10m',
        'iucn_efg_code_100m': 'code_100m',
        'homogeneity_estimate_10m': 'homog_10m',
        'homogeneity_estimate_100m': 'homog_100m',
        'interpreter_confidence (1-5)': 'confidence'
    }

    # Filter the dictionary to keep only keys that exist in df.columns
    filtered_rename_dict = {k: v for k, v in rename_dict.items() if k in gdf.columns}

    gdf.rename(columns=filtered_rename_dict, inplace=True)

    logging.info(f'End of API requests, {len(gdf)} annotations downloaded successfully')

    return gdf[gdf.geometry.type == 'Point'].copy()


def upload_to_bucket(source_file_name, destination_blob_name):
    """Uploads a file to the specified Cloud Storage bucket."""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        logging.info(f'The blob {bucket_name}/{destination_blob_name} already existed and has been overwritten.')
    
    blob.upload_from_filename(source_file_name)

    logging.info(f'Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}.')


@functions_framework.http
def main(request):

    # Initialisation of the Google Cloud project
    ee.Initialize(project=gc_project_id)

    # Retrieving and formatting annotations into a GeoDataFrame
    point_gdf = get_gdf_annotations()

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Temporary file name used for transfer to Google Cloud Storage
        tmp_file = 'ess_points'

        # Convert the GeoDataFrame to a shapefile
        point_gdf.to_file(os.path.join(temp_dir, f'{tmp_file}.shp')) 

        # Bundle all shapefile components into a zip file
        with zipfile.ZipFile(os.path.join(temp_dir, f'{tmp_file}.zip'), 'w') as zipf:
            for ext in ['.cpg', '.dbf', '.prj', '.shp', '.shx']: # List of shapefile extensions to include
                filepath = os.path.join(temp_dir, f'{tmp_file}{ext}')
                if os.path.exists(filepath):
                    zipf.write(filepath, arcname=f'{tmp_file}{ext}')
                else:
                    logging.warning(f'{filepath} not found, skipping.')

        # Upload the zipped shapefile to the Google Cloud Storage bucket
        upload_to_bucket(os.path.join(temp_dir, f'{tmp_file}.zip'), f'{blob_name}.zip')

        # Saves the GeoDataFrame as a CSV and the zipped shapefile in the local dev environment
        if env == 'dev':
            os.makedirs('data', exist_ok=True) # Create the folder if it doesn't exist
            shutil.copy(os.path.join(temp_dir, f'{tmp_file}.zip'), os.path.join('data', f'{tmp_file}.zip'))
            point_gdf.to_csv(os.path.join('data', f'{tmp_file}.csv'), index=False)

    # The temporary directory and all files inside it are automatically deleted here

    # Delete the old Earth Engine asset if it exists
    try:
        info = ee.data.getInfo(gee_asset_id)
        if info:
            ee.data.deleteAsset(gee_asset_id)
            logging.info(f'The asset {gee_asset_id} already existed and has been overwritten.')
    except Exception as e:
        logging.error(f'Error checking asset: {e}')

    # Build the Earth Engine CLI command to upload the zipped shapefile from the bucket to the Earth Engine assets folder
    cmd = ['earthengine', 'upload', 'table', f'--asset_id={gee_asset_id}', f'gs://{bucket_name}/{blob_name}.zip']

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f'Upload to Earth Engine assets folder started successfully: {result.stdout.strip()}')
    except subprocess.CalledProcessError as e:
        logging.error(f'Error uploading to Earth Engine assets folder: {e.stderr.strip()}')
 
    # The zipped shapefile in the bucket cannot be deleted automatically here because the transfer from the bucket to the Earth Engine assets folder takes some time (at least 30 seconds)

    return 'Script executed successfully. Please check the Tasks section in the Google Earth Engine Code Editor for more details.', 200


if __name__ == "__main__":
    main()
