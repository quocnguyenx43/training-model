# !pip install google-api-python-client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from io import BytesIO
import os




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

    progress = 0
    def callback(chunk, total_size):
        global progress
        progress += len(chunk)
        print(f"Uploaded {progress / total_size * 100:.2f}%")

    # Check if the file already exists in Google Drive
    existing_file_id = get_file_id_by_name(dest_file_name, service, PARENT_FOLDER_ID)
    if existing_file_id:
        # If the file exists, update its content
        media = MediaFileUpload(local_file_path, resumable=True)
        file = service.files().update(
            fileId=existing_file_id,
            media_body=media,
            progress_callback=callback,
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
            media_body=media,
            progress_callback=callback,
        ).execute()
        file_id = file['id']
        print('File uploaded. File ID: ', file_id)
        return file_id
    

import argparse as arg
parser = arg.ArgumentParser(description="Params")
parser.add_argument("--folder_name", type=str)
args = parser.parse_args()
args = vars(args)
folder_name = args['folder_name']
file_list = os.listdir(folder_name)

for file in file_list:
    path = folder_name + '/' + file
    print('path_file: ', path)
    upload_file(path, path)

# python upload_file_to_drive.py --path_file "results/logs/eval_task_2.log"
# python upload_file_to_drive.py --folder_name "models/task_1"