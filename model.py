from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

load_dotenv()

class DrowningDetectionCNN:
    def __init__(self):
        self.client = InferenceHTTPClient(
            api_url=os.environ.get("API_URL"),
            api_key=os.environ.get("API_KEY")
        )

    def predict(self, path: str):
        try:
            result = self.client.infer(path, model_id=os.environ.get("MODEL_ID"))

            # Debug print
            print("🟡 Raw model response:", result)

            predictions = result.get("predictions", [])

            if not predictions or not isinstance(predictions, list):
                print("🔴 No predictions received or wrong format.")
                return {
                    "image": result.get("image", None),
                    "predictions": None
                }

            return {
                "image": result.get("image"),
                "predictions": predictions[0]
            }

        except Exception as e:
            print("🔴 SDK Error:", e)
            return {
                "image": None,
                "predictions": None
            }
