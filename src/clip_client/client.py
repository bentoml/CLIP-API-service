from __future__ import annotations

import bentoml
from clip_api_service import svc


class Client:
    def __init__(self, service_url: str):
        self._http_client = bentoml.client.HTTPClient(svc, service_url)

    def encode(self, inputs):
        pass

    def rank(self, inputs):
        pass
