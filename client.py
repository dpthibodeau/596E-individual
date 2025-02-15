import argparse

import numpy as np
from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

url = "http://127.0.0.1:5000/classify_age_gender"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        required=True,
        help="image file or folder with images",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Folder for output results"
    )
    parser.add_argument("--single-person", action="store_true")
    return parser


def main():
    parser = get_parser()
    params = vars(parser.parse_args())
    inputs = [{"text": params["input"]}]
    data_type = DataTypes.TEXT
    response = client.request(inputs, data_type, params)
    if params["output"] is not None:
        response = np.asarray(response)
        np.save(params["output"] + ".npy", response)
    else:
        print(response)


if __name__ == "__main__":
    main()
