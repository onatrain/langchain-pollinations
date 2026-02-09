import json
import base64


def convert_to_json(python_object):
    return json.dumps(python_object, ensure_ascii=False, indent=2)


def encode_to_base64(file_path):
    try:
        with open(file_path, "rb") as file:
            encoded_file = base64.b64encode(file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {file_path}")
    except Exception as e:
        raise
    else:
        return encoded_file


def file_to_data(file_path, format_type = "image/jpeg;base64"):
    encoded_file = encode_to_base64(file_path)
    return f"data:{format_type},{encoded_file}"


