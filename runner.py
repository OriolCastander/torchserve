"""
This script runs torchserve --start

TODO: ACCEPT MULTIPLE MODELS
"""


import asyncio
import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description="Submits a model to a service")
parser.add_argument("--path", required=True, help="The name of the model")
parser.add_argument("--configPath", required=True, help="The npath to where the config properties is stored")

async def main() -> None:

    args = parser.parse_args()

    processArgs(args)

    dirname = os.path.dirname(__file__)

    #process = await asyncio.create_subprocess_exec(*[
    process = subprocess.Popen([
        "torchserve", "--start", "--foreground", "--model-store", os.path.dirname(args.path), "--models",
        os.path.basename(args.path), "--ts-config", args.configPath, "--log-config", os.path.join(dirname, "customLogger.xml")
    ], stdout=subprocess.PIPE)

    while process.poll() is None:
        output = process.stdout.readline()
        print("output is: ", output)

    print("finished")


def processArgs(args) -> None:
    """
    Fails if args paths' are not valid, transforms them to absolute paths if they are relative
    """

    dirname = os.path.dirname(__file__)

    if args.path[0] != "/": args.path = os.path.join(dirname, args.path)
    if args.configPath[0] != "/": args.configPath = os.path.join(dirname, args.configPath)

    if not os.path.isfile(args.path):
        raise Exception(f"Path to .mar file: {args.path} is not valid")
    if os.path.splitext(args.path)[1] != ".mar":
        raise Exception(f"Path to .mar file {args.path} does not point to a .mar file")
    if not os.path.isfile(args.configPath):
        raise Exception(f"Path to config.properties file: {args.configPath} is not valid")
    if os.path.splitext(args.configPath)[1] != ".properties":
        raise Exception(f"Path to config.propertiesfile {args.configPath} does not point to a valid file")

    return

if __name__ == "__main__":
    asyncio.run(main())