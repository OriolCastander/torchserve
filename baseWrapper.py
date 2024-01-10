"""
An example of a wrapper that holds all the necessary info to submit and run a 
model on the service.

You may construct your own wrapper around your model or extend this base wrapper
It ought to contain these parameters and methods:

modelArgs
modelKwargs
modelClassPath
modelClass
loadModelClass()
predict()

See BaseWrapper class docs for more info on each one
"""



from typing import Union
import torch

import os
import importlib
import types
import inspect

import json


class BaseWrapper:
    """
    A mock example of a valid wrapper.

    Attributes:

        modelArgs (Union[list, str, None]):
            The args  necessary to instantiate the model via model = Model(*args, **kwargs)
            If it is a list, that list is fed directly. If it is a str, it will be treated as the 
            path to a json file where the modelArgs key will be said list. None is equivalent to an empty list

        modelKwargs (Union[dict, str, None]):
            The kwargs  necessary to instantiate the model via model = Model(*args, **kwargs)
            If it is a dict, that dict is fed directly. If it is a str, it will be treated as the 
            path to a json file where the modelKwargs key will be said dict. None is equivalent to an empty dict

    """
    def __init__(self, rootPath: str) -> None:
        self.modelArgs: Union[list, str, None] = [2,[2,2],2]
        self.modelKwargs: Union[list, str, None] = {}

        self.modelClassPath = os.path.join(rootPath, "input/testModel.py")
        self.modelClass: type[torch.nn.Module] = loadModelClass(self.modelClassPath)


    def loadModel(self, modelPath: str) -> torch.nn.Module:
        """
        Loads and returns the model saved in the model path (as a pth file)

        Args:
            modelPath (str): The absolute path of the model.pth file

        Returns:
            torch.nn.Module: the model
        """

        if type(self.modelArgs) == list:
            modelArgs = self.modelArgs
        elif type(self.modelArgs) == str:
            try:
                with open(self.modelArgs) as file:
                    modelArgs = json.load(file)["modelArgs"]
            except:
                raise Exception(f"Cannot open file with model args at {self.modelArgs}")
        elif self.modelArgs is None:
            modelArgs = []
        else:
            raise Exception("Unrecognized model args ", self.modelArgs)
        
        if type(self.modelKwargs) == dict:
            modelKwargs = self.modelKwargs
        elif type(self.modelKwargs) == str:
            try:
                with open(self.modelKwargs) as file:
                    modelKwargs = json.load(file)["modelKwargs"]
            except:
                raise Exception(f"Cannot open file with model args at {self.modelKwargs}")
        elif self.modelKwargs is None:
            modelKwargs = []
        else:
            raise Exception("Unrecognized model args ", self.modelArgs)
        
    
        model = self.modelClass(*modelArgs, **modelKwargs)
        model.load_state_dict(torch.load(modelPath))
        return model
    
    def predict(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Feeds the input data through the model to predict a result.
        Will be called from the handler.

        Args:
            model (torch.nn.Module): the model
            x (torch.Tensor): the input data as tensor

        Returns:
            torch.Tensor: the result 

        """

        #return model.forward(x) standard approach
        return model.network(x) ##IN MY CASE, model.network is a nn.Sequential and the "entry point"
        
def loadModelClass(modelClassPath: str) -> type[torch.nn.Module]:
    """
    Returns the class object of the model given the path of the model class file (the file where the model 
    is declared)

    Args:
        modelClassPath (str): The absolute path of the file that contains the model class

    Returns:
        type[torch.nn.Module]: the class object from which instances of the model can be instanciated
    """

    loader = importlib.machinery.SourceFileLoader('model_class_file', modelClassPath)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)

    try:
        loader.exec_module(module)
    except FileNotFoundError:
        raise Exception(f"File for model class could not be found in {modelClassPath}")
    
    members = inspect.getmembers(module, predicate=inspect.isclass)
    
    if len(members) == 0:
        raise Exception("Could not find a class in the model class file")
    
    elif len(members) > 1:
        raise Exception("Found multiple classes in the model class file, please declare only one")
    
    return members[0][1]

