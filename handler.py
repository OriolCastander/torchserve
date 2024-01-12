"""
The handler
"""

from ts.torch_handler.base_handler import BaseHandler
import numpy as np
import torch
import os
import json

import importlib.machinery
import types
import inspect

class Handler(BaseHandler):

    def __init__(self) -> None:
        super(Handler, self).__init__()

    def initialize(self, context)-> None:
        
        ##LOAD THE CONFIG FILE INTO MAPPING
        properties = context.system_properties
        model_dir = properties.get("model_dir")#Path to the model
        mapping_file_path = os.path.join(model_dir, "config.json")

        with open(mapping_file_path, "r") as f:
            config = json.load(f)
            self.mapping = config

        ##INSTANTIATE THE MODEL (first get the wrapper)
        loader = importlib.machinery.SourceFileLoader('wrapper_file', config["wrapperPath"])
        module = types.ModuleType(loader.name)
        loader.exec_module(module)

        members = inspect.getmembers(module, predicate=inspect.isclass)
        self.wrapper = members[0][1](config["basePath"])
        self.model = self.wrapper.loadModel(config["modelPath"])

    def preprocess(self, requests):
        """
        The data in the request must be structured in the following way:

            3 bytes (int-big) with the length of the metadata
            The corresponding amount of utf-8 bytes with the stringified json metadata
            The numpy array as bytes

        In the metadata there must be a key "shape" with a list of integers that specifies the data shape
        """
        #TODO: HANDLE MULTIPLE CONCURRENT REQUESTS? 
        for request in requests:
            requestData = request['body']

            metadataLength = int.from_bytes(requestData[:3])

            rawMetadata = requestData[3:metadataLength+3]
            metadata = json.loads(rawMetadata.decode('utf-8'))

            data = np.frombuffer(requestData[metadataLength+3:]).reshape(metadata["shape"])

            return torch.tensor(data, dtype=torch.float)

    def inference(self, x):
        return self.wrapper.predict(self.model, x)
    
    def postprocess(self, preds):
        
        result = preds.detach().numpy()

        metadata = {
            "shape": list(result.shape)
        }

        metadataBytes = str(json.dumps(metadata)).encode("utf-8")
        metadataLength = len(metadataBytes).to_bytes(3)

        return [metadataLength + metadataBytes + result.tobytes()]