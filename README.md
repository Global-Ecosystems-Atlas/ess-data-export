# Annotation export script for Earth System Studio projects

This script exports annotations from an Earth System Studio (ESS) project into a local folder named `exported_data`.  
If run with the `--online-export` flag, the annotations are also uploaded to Google Cloud Storage (GCS) and Google Earth Engine (GEE).

## Prerequisites

- Python 3.13+ (must be accessible from the command line)
- Git (must be accessible from the command line)
- An Earth System Studio API key
- (Optional) A Google Cloud project with access to GCS and GEE

## Installation & Setup (for Windows)

1. Open a terminal and navigate to the directory where the repository should be cloned.

2. Clone this repository and navigate into it.
    ```bash
    git clone https://github.com/Global-Ecosystems-Atlas/ess-data-export.git
    cd ess-data-export
    ```

3. Create and activate a virtual environment.
    ```bash
    py -m venv .venv
    .venv\Scripts\activate
    ```

3. Install the required Python packages.
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following environment variables.
    ```bash
    # === REQUIRED ENVIRONMENT VARIABLES ===

    # Earth System Studio API key.
    API_KEY=

    # Earth System Studio project ID from which to extract the annotations.
    ESS_PROJECT_ID=

    # Environment.
    ENV=dev


    # === OPTIONAL ENVIRONMENT VARIABLES (only required for --online-export) ===

    # Google Cloud Storage blob name. The last element is the file name — do not include an extension.
    BLOB_NAME=

    # Google Cloud Storage bucket name. Will be concatenated with the blob name.
    BUCKET_NAME=

    # Google Earth Engine asset ID where the annotations will be saved. The last element is the file name — do not include an extension.
    # The asset must exist before trying to save data to it.
    GEE_ASSET_ID=
    ```

## Usage

Run the script with the following command.
   ```bash
   py main.py [--online-export]
   ```

After execution, check the `exported_data` folder in the project directory for the exported files (CSV, GeoJSON, and zipped Shapefile), and if the `--online-export` flag was used, verify that the data has also been uploaded to your GEE assets at code.earthengine.google.com.

## Security Notes

- Keep API keys and tokens secure.
- Never commit API keys or tokens to version control systems.