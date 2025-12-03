import json
import os

from gypscie_api import GypscieAPISession
from requests.exceptions import HTTPError


def execute_dataflow(gypscie_api: GypscieAPISession):
    data = {
        "dataflow_id": 36,
        "environment_id": 1,
        "parameters": [
            {
                "function_id": 42,
                "params": {
                    "radar_data_path": 178,
                    "rain_gauge_data_path": 179,
                    "grid_data_path": 177,
                },
            },
            {"function_id": 43},
            {
                "function_id": 45,
                "params": {
                    "model_path": 191
                },  # model was registered on Gypscie as a dataset
            },
        ],
        "project_id": 1,
    }
    response = gypscie_api.post("/api/dataflow_run", json=data)
    os.environ["DATAFLOW_TASK_ID"] = response.json().get("task_id")
    print(f"Dataflow execution started. Task ID: {os.environ['DATAFLOW_TASK_ID']}")
    filepath = "data/response_dataflow_run.json"
    save_response_to_file(response, filepath)


def monitor_dataflow(gypscie_api: GypscieAPISession):
    if "DATAFLOW_TASK_ID" not in os.environ:
        os.environ["DATAFLOW_TASK_ID"] = input("Type the task ID you want to monitor: ")
    try:
        response = gypscie_api.get(
            f"/api/status_dataflow_run/{os.environ['DATAFLOW_TASK_ID']}"
        )
    except HTTPError as err:
        if err.response.status_code == 404:
            print(f"Task {os.environ['DATAFLOW_TASK_ID']} not found")
            return
    dataflow_run = response.json()
    print(f"Dataflow status: {dataflow_run.get('status')}")
    print(f"Output datasets: {dataflow_run.get('output_datasets')}")
    if dataflow_run.get("status") == "SUCCESS":
        filepath = "data/response_dataflow_status.json"
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
        "1": {"message": "Execute Dataflow", "function": execute_dataflow},
        "2": {"message": "Monitor Dataflow", "function": monitor_dataflow},
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
        selected_option = input("Type the ID of the action you want to perform: ")
        if selected_option not in menu_options.keys():
            print(f'Option "{selected_option}" is not valid. Try again.')
            continue
        if selected_option == "0":
            break
        selected_function = menu_options[selected_option]["function"]
        selected_function(gypscie_api)


if __name__ == "__main__":
    run()
