from google.cloud import storage
from google.oauth2 import service_account
import os
import json
import uuid
from typing import Optional
from fastapi import UploadFile, HTTPException


class CloudStorage:
    _instance: Optional['CloudStorage'] = None

    def __init__(self):
        try:
            creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
            if not creds_json:
                raise ValueError(
                    "GOOGLE_CREDENTIALS_JSON no está configurado en las variables de entorno")

            creds_info = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_info)

            self.client = storage.Client(
                credentials=credentials, project=creds_info.get("project_id"))

            bucket_name = os.getenv("GOOGLE_STORAGE_BUCKET")
            if not bucket_name:
                raise ValueError(
                    "GOOGLE_STORAGE_BUCKET no está configurado en las variables de entorno")

            self.bucket = self.client.bucket(bucket_name)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error al parsear GOOGLE_CREDENTIALS_JSON: {str(e)}")
        except Exception as e:
            raise ValueError(
                f"Error al inicializar Google Cloud Storage: {str(e)}")

    @classmethod
    def get_instance(cls) -> 'CloudStorage':
        if cls._instance is None:
            cls._instance = CloudStorage()
        return cls._instance

    def _get_public_url(self, blob_name: str) -> str:
        return f"https://storage.googleapis.com/{self.bucket.name}/{blob_name}"

    async def upload_file(self, file_path: str, destination_name: str, folder: str = "usuarios") -> str:
        try:
            blob_path = f"{folder}/{destination_name}"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(file_path)
            return self._get_public_url(blob_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al subir archivo a Google Cloud Storage: {str(e)}"
            )

    async def upload_from_file(self, file: UploadFile, folder: str = "usuarios") -> str:
        try:
            file_extension = file.filename.split(
                '.')[-1] if '.' in file.filename else 'jpg'
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            blob_path = f"{folder}/{unique_filename}"
            blob = self.bucket.blob(blob_path)
            contents = await file.read()
            blob.upload_from_string(
                contents,
                content_type=file.content_type or 'image/jpeg'
            )
            return self._get_public_url(blob_path)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al subir archivo a Google Cloud Storage: {str(e)}"
            )

    async def upload_from_bytes(self, file_bytes: bytes, filename: str, content_type: str = 'image/jpeg', folder: str = "usuarios") -> str:
        try:
            file_extension = filename.split(
                '.')[-1] if '.' in filename else 'jpg'
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            blob_path = f"{folder}/{unique_filename}"

            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(file_bytes, content_type=content_type)

            return self._get_public_url(blob_path)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error al subir archivo a Google Cloud Storage: {str(e)}"
            )

    async def delete_file(self, file_path: str) -> bool:
        try:
            blob = self.bucket.blob(file_path)
            blob.delete()
            return True
        except Exception as e:
            print(f"Error al eliminar archivo: {str(e)}")
            return False

    async def delete_file_by_url(self, file_url: str) -> bool:
        try:
            if self.bucket.name in file_url:
                parts = file_url.split(f"{self.bucket.name}/")
                if len(parts) >= 2:
                    file_path = parts[1]
                    return await self.delete_file(file_path)
            return False
        except Exception as e:
            print(f"Error al eliminar archivo por URL: {str(e)}")
            return False

    def get_public_url(self, blob_path: str) -> str:
        return self._get_public_url(blob_path)


def get_cloud_storage() -> CloudStorage:
    return CloudStorage.get_instance()
