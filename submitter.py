"""
This script handles the submission of a model to the service

It follows the following steps:

STEP 1: instantiates a wrapper (more info about the wrapper in baseWrapper)

STEP 2: creates the config.json file that will be pushed to the .mar file

STEP 3: Creates the key with keytool and the config.properties

STEP 4: Archives the model into the .mar file

"""

import asyncio

import argparse

import os
import importlib.machinery
import types
import inspect

import json

import subprocess

from baseWrapper import BaseWrapper

parser = argparse.ArgumentParser(description="Submits a model to a service")
parser.add_argument("--wrapper", default="baseWrapper.py", help="The path to your wrapper")
parser.add_argument("--name", required=True, help="The name of the model")
parser.add_argument("--model", default="input/testModel.pth", help="The path of the saved model")



async def main()->None:
    args = parser.parse_args()

    
    wrapperPath = os.path.join(os.path.dirname(__file__), args.wrapper) 
    wrapper = instantiateWrapper(wrapperPath)

    createConfigFile(args, wrapper)

    createKey()
    createConfigPropertiesFile()

    
    await archiveModel(args, wrapper)

    #DELETE THE CONFIG.JSON
    if os.path.exists(os.path.join(os.path.dirname(__file__), "output/config.json")):
        os.remove(os.path.join(os.path.dirname(__file__), "output/config.json"))


    

def instantiateWrapper(wrapperPath: str)->BaseWrapper:
    """
    Returns an instance of the custom wrapper class (whose file is provided in --wrapper)

    Args:
        wrapperPath (str): The absolute path of the wrapper file
    
    Returns:
        CustomWrapper: An instance of the custom wrapper class

    TODO:
        * Support for parameters in the wrapper initialization?
    """

    loader = importlib.machinery.SourceFileLoader('wrapper_file', wrapperPath)
    module = types.ModuleType(loader.name)
    
    try:
        loader.exec_module(module)
    except FileNotFoundError:
        raise Exception(f"File for wrapper could not be found in {wrapperPath}")
    
    members = inspect.getmembers(module, predicate=inspect.isclass)
    
    if len(members) == 0:
        raise Exception("Could not find a class in the wrapper file")
    
    elif len(members) > 1:
        raise Exception("Found multiple classes in the wrapper file, please declare only one")
    
    return members[0][1](os.path.dirname(__file__))



def createConfigFile(args, wrapper: BaseWrapper) -> None:
    """
    Creates the config.json file that will be loaded into the .mar

    TODO: more info into the allowed params in the data

    Args:
        args (Namespace): the args from the parser
        wrapper (BaseWrapper): the wrapper
    """

    dirname = os.path.dirname(__file__)
    outputPath = os.path.join(dirname, "output/")
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath[:-1]) #MAKE THE OUTPUT FOLDER, REMOVE THE LAST /
    
    data = {
        "modelArgs": wrapper.modelArgs,
        "modelKwargs": wrapper.modelKwargs,
        "wrapperPath": os.path.join(dirname, args.wrapper),
        "modelPath": os.path.join(dirname, args.model),
        "basePath": dirname,
    }

    with open(os.path.join(dirname, "output/config.json"), "w") as file:
        json.dump(data, file)


async def archiveModel(args, wrapper: BaseWrapper) -> None:
    """
    Runs torch-model-archive to store the .mar file

    Args:
        args (Namespace): the args from the parser
        wrapper (BaseWrapper): the wrapper
    """

    dirname = os.path.dirname(__file__)

    if not os.path.isdir(os.path.join(dirname, "output/model_store")):
        os.mkdir(os.path.join(dirname, "output/model_store"))

    process = await asyncio.create_subprocess_exec(*["torch-model-archiver", "--model-name", args.name,
        "--version", "1.0",
        "--model-file", os.path.join(dirname, wrapper.modelClassPath),
        "--serialized-file", os.path.join(dirname, args.model),
        "--export-path", "output/model_store",
        "--extra-files", os.path.join(dirname, "output/config.json"),
        "--handler", os.path.join(dirname, "handler.py"),
    ])

    stdout, stderr = await process.communicate()

def createKey(**kwargs) -> None:
    """
    Creates the key to authenticate connection  to the model.

    Kwargs:
        keystorePath (str, "__file__ output/keystore.p12"): the location where the key is saved
        storepass (str, "password"): 
    """

    defaultValues = {
        "keystorePath": os.path.join(os.path.dirname(__file__), "output/keystore.p12"),
        "storepass": "password",
    }

    values = {**defaultValues, **kwargs}

    subprocess.Popen(["keytool", "-genkey", "-keyalg", "RSA", "-alias", "ts", "-keystore", values["keystorePath"],
                    "-storepass", values["storepass"], "-storetype", "PKCS12", "-validity", "3600", "-keysize", "2041",
                    "-dname", "CN=www.MY_TS.com, OU=Cloud Service, O=model server, L=Palo Alto, ST=California, C=US"], stderr=subprocess.DEVNULL)
    #keytool -genkey -keyalg RSA -alias ts -keystore keystore.p12 -storepass changeit -storetype PKCS12 -validity 3600 -keysize 2048 -dname "CN=www.MY_TS.com, OU=Cloud Service, O=model server, L=Palo Alto, ST=California, C=US"
    

def createConfigPropertiesFile(**kwargs) -> None:
    """
    Creates the config.properties file with authentication and routing info for the model

    Kwargs:
        keystorePath (str, "__file__ output/keystore.p12": the location where the key is saved
        storepass (str, "password"): 
        inferencePort (str, "8443):
        path (str, "__file__ output/config.properties"): the path to the created file
    """

    defaultValues = {
        "keystorePath": os.path.join(os.path.dirname(__file__), "output/keystore.p12"),
        "storepass": "password",
        "inferencePort": "8443",
        "path": os.path.join(os.path.dirname(__file__), "output/config.properties")
    }

    values = {**defaultValues, **kwargs}

    configPropertiesText = f"""inference_address=https://127.0.0.1:{values["inferencePort"]}
management_address=https://127.0.0.1:8444
metrics_address=https://127.0.0.1:8445
keystore={values["keystorePath"]}
keystore_pass={values["storepass"]}
keystore_type=PKCS12

# cors_allowed_origin is required to enable CORS, use '*' or your domain name
cors_allowed_origin=*
# required if you want to use preflight request
cors_allowed_methods=GET, POST, PUT, OPTIONS
# required if the request has an Access-Control-Request-Headers header
cors_allowed_headers=X-Custom-Header"""

    with open(values["path"], "w") as file:
        file.write(configPropertiesText)

if __name__ == "__main__":
    asyncio.run(main())