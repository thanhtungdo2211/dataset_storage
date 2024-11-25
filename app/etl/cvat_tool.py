import requests
import json
from configure import Config

class CVATClient:
    def __init__(self):
        self.username = Config.cvat.CVAT_USERNAME
        self.password = Config.cvat.CVAT_PASSWORD
        self.domain = Config.cvat.CVAT_DOMAIN
        self.session = requests.Session()
        self.csrf_token = None
        self.session_id = None
        self.api_key = None

    def login(self):
        url = f"{self.domain}/api/auth/login"
        payload = json.dumps({
            "username": self.username,
            "password": self.password
        })
        headers = {
            'Content-Type': 'application/json',
        }

        response = self.session.post(url, headers=headers, data=payload)
        response.raise_for_status()

        data = response.json()
        self.api_key = data['key']
        self.csrf_token = self.session.cookies.get('csrftoken')
        self.session_id = self.session.cookies.get('sessionid')

    def get_all_data(self, id_job: int):
        if(not self.csrf_token or not self.session_id):
            self.login()
        url = f"{self.domain}/api/jobs/{id_job}/data/meta?org="
        if not self.csrf_token or not self.session_id:
            self.login()

        headers = {
            'Content-Type': 'application/json',
            'Cookie': f'csrftoken={self.csrf_token}; sessionid={self.session_id}'
        }

        response = self.session.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def delete_frames(self, id_job: int, idx: list):
        if(not self.csrf_token or not self.session_id):
            self.login()
        url = f"{self.domain}/api/jobs/{id_job}/data/meta"
        if not self.csrf_token or not self.session_id:
            self.login()

        headers = {
            'Content-Type': 'application/json',
            'Cookie': f'csrftoken={self.csrf_token}; sessionid={self.session_id}'
        }
        payload = json.dumps({
            "deleted_frames": idx
        })

        response = self.session.patch(url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
