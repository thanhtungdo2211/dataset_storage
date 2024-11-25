import cv2
import numpy as np

import base64
import json
import requests
import requests.utils
from enum import Enum
from typing import Any, Dict

from configure import Config

_USER_AGENT = f"{requests.utils.default_user_agent()}"
config = Config()


def make_requests_session() -> requests.Session:
    session = requests.Session()
    session.headers['User-Agent'] = _USER_AGENT
    return session


class LambdaType(Enum):
    DETECTOR = "detector"
    KEYPOINTOR = "keypointor"
    SIAMESER = "siameser"
    SEGMENTOR = "segmentor"
    CLASSIFIER = "classifier"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value


class LambdaGateway:
    NUCLIO_ROOT_URL = '/api/functions'

    def _http(
        self,
        method="get", scheme=None,
        host=None, port=None, function_namespace=None,
        url=None, headers=None, data=None
    ):
        NUCLIO_GATEWAY = '{}://{}:{}'.format(
            config.NUCLIO_SCHEME,
            config.NUCLIO_HOST,
            config.NUCLIO_PORT)
        NUCLIO_FUNCTION_NAMESPACE = config.NUCLIO_FUNCTION_NAMESPACE
        NUCLIO_TIMEOUT = config.NUCLIO_DEFAULT_TIMEOUT
        extra_headers = {
            'x-nuclio-project-name': 'default',
            'x-nuclio-function-namespace': NUCLIO_FUNCTION_NAMESPACE,
            'x-nuclio-invoke-via': 'domain-name',
            'X-Nuclio-Invoke-Timeout': f"{int(NUCLIO_TIMEOUT)}s",
        }
        if headers:
            extra_headers.update(headers)

        if url:
            url = "{}{}".format(NUCLIO_GATEWAY, url)
        else:
            url = NUCLIO_GATEWAY

        with make_requests_session() as session:
            reply = session.request(
                method,
                url,
                headers=extra_headers,
                timeout=int(NUCLIO_TIMEOUT),
                json=data
            )
            reply.raise_for_status()
            try:
                response = reply.json()
            except Exception:
                response = reply.text

        return response

    def get_all(self):
        response = self._http(url=self.NUCLIO_ROOT_URL)
        return response

    def get(self, func_id):
        response = self._http(url=self.NUCLIO_ROOT_URL + '/' + func_id)
        return response

    def invoke(self, func, payload):
        return self._invoke_via_dashboard(func, payload)

    def _invoke_via_dashboard(self, func, payload):
        return self._http(
            method="post",
            url='/api/function_invocations',
            data=payload, headers={
                'x-nuclio-function-name': func.id,
                'x-nuclio-path': '/'
            })


class LambdaFunction:
    def __init__(self, gateway: LambdaGateway, function_name: str):
        self.gateway = gateway
        function = self.load_function(function_name=function_name)

        # ID of the function (e.g. omz.public.yolo-v3)
        self.id = function['metadata']['name']
        # type of the function (e.g. detector, interactor)
        meta_anno = function['metadata'].get('annotations', {})
        kind = meta_anno.get('type')
        try:
            self.kind = LambdaType(kind)
        except ValueError:
            self.kind = LambdaType.UNKNOWN
        # dictionary of labels for the function (e.g. car, person)
        spec = json.loads(meta_anno.get('spec') or '[]')

        # Labels
        self.labels = self.parse_labels(spec)

        # mapping of labels and corresponding supported attributes
        self.func_attributes = {
            item['name']: item.get('attributes', []) for item in spec}
        for label, attributes in self.func_attributes.items():
            if len(
                [attr['name'] for attr in attributes]) != len(
                    set([attr['name'] for attr in attributes])):
                raise ValueError(
                    "`{}` lambda function has non-unique \
                        attributes for label {}".format(self.id, label))

        # description of the function
        self.description = function['spec']['description']

        # display name for the function
        self.name = meta_anno.get('name', self.id)
        self.version = int(meta_anno.get('version', '1'))
        self.help_message = meta_anno.get('help_message', '')

    def load_function(self, function_name: str):
        try:
            function = self.gateway.get(func_id=function_name)
            return function
        except Exception as e:
            raise Exception("Can not connect to function", e)
            exit()

    def to_dict(self):
        response = {
            'id': self.id,
            'kind': str(self.kind),
            'labels_v2': self.labels,
            'description': self.description,
            'name': self.name,
            'version': self.version,
            "help_message": self.help_message
        }

        return response

    def parse_attributes(self, attrs_spec):
        parsed_attributes = [{
            'name': attr['name'],
            'input_type': attr['input_type'],
            'values': attr['values'],
        } for attr in attrs_spec]

        if len(parsed_attributes) != len({
            attr['name'] for attr in attrs_spec
        }):
            raise ValueError(
                f"{self.id} lambda function has non-unique attributes")

        return parsed_attributes

    def parse_labels(self, spec):
        parsed_labels = []
        for label in spec:
            parsed_label = {
                'name': label['name'],
                'type': label.get('type', 'unknown'),
                'attributes': self.parse_attributes(
                    label.get('attributes', []))
            }
            if parsed_label['type'] == 'skeleton':
                parsed_label.update({
                    'sublabels': self.parse_labels(label['sublabels']),
                    'svg': label['svg']
                })
            parsed_labels.append(parsed_label)

        if len(parsed_labels) != len({label['name'] for label in spec}):
            raise ValueError(
                f"{self.id} lambda function has non-unique labels")

        return parsed_labels

    def invoke(
        self,
        data: Dict[str, Any] = None
    ):
        payload = {}
        data = {k: v for k, v in data.items() if v is not None}

        threshold = data.get("threshold")
        if threshold:
            payload.update({"threshold": threshold})

        # model_labels = self.labels

        image = data.get("image")
        if image is not None:
            payload.update({
                "image": self._get_image(image)
            })

        response = self.gateway.invoke(self, payload)

        return response

    def _get_image(self, image: np.array, encode_format=".jpg"):
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")
        _, im_arr = cv2.imencode(encode_format, image)
        return base64.b64encode(im_arr.tobytes()).decode('utf-8')
