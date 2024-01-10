"""
The handler
"""

from ts.torch_handler.base_handler import BaseHandler
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
        ##TODO: IMPROVE THIS, CREATE THE INPUT DIRECTLY USING NUMPY
        input = []
        for request in requests:
            for row in json.loads(request['body'])["data"]:
                input.append(row)

        tensor = torch.tensor(input, dtype=torch.float)
        
        ##TODO: CHECK THAT TENSOR HAS THE APPROPIATE SHAPE (SHAPE SHOULD BE INCLUDED IN MAPPING)

        return tensor

    def inference(self, x):
        return self.wrapper.predict(self.model, x)
    
    def postprocess(self, preds):

        return [preds.detach().numpy().tolist()]