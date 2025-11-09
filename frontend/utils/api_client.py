import requests
from typing import Optional

class APIClient:
    """Client pour communiquer avec le backend"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def upload_file(self, file):
        """Upload un fichier"""
        files = {"file": file}
        response = requests.post(f"{self.base_url}/api/upload", files=files)
        return response.json()
    
    def get_datasets(self):
        """Récupère la liste des datasets"""
        response = requests.get(f"{self.base_url}/api/datasets")
        return response.json()
    
    def mine_patterns(self, params: dict):
        """Lance l'extraction de motifs"""
        response = requests.post(f"{self.base_url}/api/mine", json=params)
        return response.json()
    
    def sample_patterns(self, params: dict):
        """Échantillonne les motifs"""
        response = requests.post(f"{self.base_url}/api/sample", json=params)
        return response.json()
