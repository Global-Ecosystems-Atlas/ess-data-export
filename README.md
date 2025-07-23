# Annotation export script for Earth System Studio projects

This script exports annotations from multiple Earth System Studio (ESS) projects into a local folder named `exported_data`.  
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

4. Install the required Python packages.
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root directory with the Earth System Studio API key. Do not share this file or commit it to version control.
    ```bash
    # Earth System Studio API key.
    API_KEY=
    ```

6. Create a `config.json` file in the project root directory based on the example file `config.example.json`.
This file must follow the same structure as the example, with at least one required key: `ess_projects`, which contains a list of Earth System Studio projects.  
Each project should include the following keys:
- `ess_project_id` (_required_): Earth System Studio project ID to extract annotations from.
- `gc_bucket_name` (_required if online export_): Google Cloud Storage bucket name. Will be concatenated with the blob name.
- `gc_blob_name` (_required if online export_): Google Cloud Storage blob name. The last element is the filename — do not include an extension.
- `gee_asset_id` (_required if online export_): Google Earth Engine asset ID where the annotations will be saved. The last element is the filename — do not include an extension. The asset must exist before saving data to it.

## Usage

Run the script with the following command.
   ```bash
   py main.py [--online-export]
   ```

After execution, check the `exported_data` folder in the project directory for the exported files (CSV, GeoJSON, and zipped Shapefile), and if the `--online-export` flag was used, verify that the data has also been uploaded to your GEE assets at https://code.earthengine.google.com.

## Security Notes

- Keep API keys and tokens secure.
- Never commit API keys or tokens to version control systems.