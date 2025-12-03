import json

from gypscie_api import GypscieAPISession


def register_data_ingestion_function(gypscie_api: GypscieAPISession):
    files = {
        "files": open(file="preprocessing.py", mode="rb"),
    }
    payload = {
        "name": "load",
        "description": "Loads the data from the sources",
        "input_arity": "many",
        "output_arity": "many",
        "function_type": "ingestion",
        "params": json.dumps(
            [
                {
                    "name": "radar_data_path",
                    "data_type": "path",
                    "default_value": "inea_radar_guaratiba",
                },
                {
                    "name": "rain_gauge_data_path",
                    "data_type": "path",
                    "default_value": "alertario_rain_gauge",
                },
                {
                    "name": "grid_data_path",
                    "data_type": "path",
                    "default_value": "rionowcast_gridpoints",
                },
            ]
        ),
    }
    response = gypscie_api.post("/api/functions", files=files, data=payload)
    filepath = "data/response_load_function.json"
    save_response_to_file(response, filepath)


def register_preprocessing_function(gypscie_api: GypscieAPISession):
    files = {
        "files": open(file="preprocessing.py", mode="rb"),
    }
    payload = {
        "name": "preprocessing",
        "description": "Preprocesses the data to be used as input for COR model",
        "input_arity": "many",
        "output_arity": "one",
        "function_type": "integrator",
    }
    response = gypscie_api.post("/api/functions", files=files, data=payload)
    filepath = "data/response_preprocessing_function.json"
    save_response_to_file(response, filepath)


def register_predict_function(gypscie_api: GypscieAPISession):
    files = {
        "files": open(file="preprocessing.py", mode="rb"),
    }
    payload = {
        "name": "predict",
        "description": "Predicts the precipitation using the COR model",
        "input_arity": "many",
        "output_arity": "one",
        "function_type": "transformer",
        "params": json.dumps(
            [{"name": "model_path", "data_type": "path", "default_value": "model"}]
        ),
    }
    response = gypscie_api.post("/api/functions", files=files, data=payload)
    filepath = "data/response_predict_function.json"
    save_response_to_file(response, filepath)


def register_dataflow(gypscie_api: GypscieAPISession):
    files = {
        "files": open(file="conda.yaml", mode="rb"),
    }
    graph_data = [
        {
            "id": 1,
            "operation_name": "load",
            "name": "load",
            "input": ["radar_data_path", "rain_gauge_data_path", "grid_data_path"],
            "output": ["df_radar", "df_rain_gauge", "df_grid"],
            "filename": "preprocessing",
        },
        {
            "id": 2,
            "operation_name": "preprocessing",
            "name": "preprocessing",
            "input": ["df_radar", "df_rain_gauge", "df_grid"],
            "output": "X",
            "filename": "preprocessing",
        },
        {
            "id": 3,
            "operation_name": "predict",
            "name": "predict",
            "input": [
                "X",
                "model_path",
            ],
            "output": None,
            "filename": "preprocessing",
        },
    ]
    payload = {"name": "rionowcast_dataflow_v1", "graph": json.dumps(graph_data)}
    response = gypscie_api.post("/api/dataflows", files=files, data=payload)
    filepath = "data/response_dataflow_register.json"
    save_response_to_file(response, filepath)


def save_response_to_file(response, filepath):
    if response.status_code != 201:
        print(f"Error registering artifact: {response.text}")
        return
    response_json = response.json()
    with open(filepath, "w") as f:
        json.dump(response_json, f)
    print(f"Artifact registered and response saved to {filepath}")


def get_menu_options():
    return {
        "1": {
            "message": "Register Load Function",
            "function": register_data_ingestion_function,
        },
        "2": {
            "message": "Register Preprocessing Function",
            "function": register_preprocessing_function,
        },
        "3": {
            "message": "Register Predict Function",
            "function": register_predict_function,
        },
        "4": {"message": "Register Dataflow", "function": register_dataflow},
        "0": {"message": "Exit", "function": None},
    }


def display_menu(menu_options):
    print("\nSelect the function to register:")
    for option_id, option in menu_options.items():
        print(f"{option_id}. {option['message']}")


def run():
    gypscie_api = GypscieAPISession()
    menu_options = get_menu_options()
    while True:
        display_menu(menu_options)
        selected_option = input("Type the ID of the artifact you want to register: ")
        if selected_option not in menu_options.keys():
            print(f'Option "{selected_option}" is not valid. Try again.')
            continue
        if selected_option == "0":
            break
        selected_function = menu_options[selected_option]["function"]
        selected_function(gypscie_api)


if __name__ == "__main__":
    run()
