# !pip install google-api-python-client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from io import BytesIO




def authenticate():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = './service_account.json'
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def get_file_id_by_name(file_name, service, folder_id):
    PARENT_FOLDER_ID = '1hfHA-IhzhUUGiLtcqahIQx53YT-xG0zm'
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get('files', [])
    return files[0]['id'] if files else None

def upload_file(local_file_path, dest_file_name):
    PARENT_FOLDER_ID = '1hfHA-IhzhUUGiLtcqahIQx53YT-xG0zm'
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    # Check if the file already exists in Google Drive
    existing_file_id = get_file_id_by_name(dest_file_name, service, PARENT_FOLDER_ID)
    if existing_file_id:
        # If the file exists, update its content
        media = MediaFileUpload(local_file_path, resumable=True)
        file = service.files().update(
            fileId=existing_file_id,
            media_body=media
        ).execute()
        print('File updated. File ID: ', existing_file_id)
        return existing_file_id
    else:
        # If the file doesn't exist, create a new file
        file_metadata = {
            'name': dest_file_name,
            'parents': [PARENT_FOLDER_ID]
        }
        media = MediaFileUpload(local_file_path, resumable=True)
        file = service.files().create(
            body=file_metadata,
            media_body=media
        ).execute()
        file_id = file['id']
        print('File uploaded. File ID: ', file_id)
        return file_id
    

import argparse as arg
parser = arg.ArgumentParser(description="Params")
parser.add_argument("--path_file", type=str)
args = parser.parse_args()
args = vars(args)

path_file = args['path_file']

upload_file(path_file, path_file)
print('path_file: ', path_file)