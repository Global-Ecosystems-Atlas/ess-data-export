# Earth System Studio to Google Earth Engine Export

This script exports annotations from a Earth System Studio (ESS) project and uploads them to a Google Earth Engine (GEE) assets folder via Google Cloud Storage (GCS).

## Prerequisites

- Python **3.13+**
- Access to the **Earth System Studio API**
- Access to the **Google Cloud project** with Google Cloud Storage (GCS) and Google Earth Engine (GEE)

## Installation & Setup

1. Clone this repository and navigate into the project directory.

2. Create and activate a virtual environment (on Windows):

    ```bash
    py -m venv .venv
    .venv\Scripts\activate
    ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a ```.env``` file with the following environment variables:
    ```bash
    # Earth System Studio API key.
    API_KEY=

    # Google Cloud Storage blob name. The last element is the file name — do not include an extension.
    BLOB_NAME=

    # Google Cloud Storage bucket name. Will be concatenated with the blob name.
    BUCKET_NAME=

    # Environment.
    ENV=dev

    # Earth System Studio project ID from which to extract the annotations.
    ESS_PROJECT_ID=

    # Google Earth Engine asset ID where the annotations will be saved. The last element is the file name — do not include an extension.
    # The asset must exist before trying to save data to it.
    GEE_ASSET_ID=
    ```

5. Authenticate with Google Earth Engine:
    ```bash
    earthengine authenticate
    ```

6. Start the local HTTP server using Functions Framework:

    ```bash
    functions-framework --target=main
    ```

7. Visit the application in your browser: http://127.0.0.1:8080

8. Once the request completes, verify the asset is uploaded at: https://code.earthengine.google.com/