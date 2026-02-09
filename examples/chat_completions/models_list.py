import dotenv

from langchain_pollinations import ModelInformation

from utils import convert_to_json

dotenv.load_dotenv()

mi = ModelInformation()
print("Modelos de texto compatibles con OpenAI:")
print(convert_to_json(mi.list_v1_models()))      # OpenAI-compatible (/v1/models)
print("\nTodos los modelos de texto:")
print(convert_to_json(mi.list_text_models()))    # (/text/models)
print("\nTodos los modelos de imagen:")
print(convert_to_json(mi.list_image_models()))   # (/image/models) incluye modelos de imagen y video
