import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from model import DrowningDetectionCNN
import io
import sys
import types

# Prevent cv2 import errors if not needed
sys.modules['cv2'] = types.SimpleNamespace()

# Annotate the image with prediction results
def annotate_image(image: Image.Image, prediction: dict) -> Image.Image:
    draw = ImageDraw.Draw(image)

    x_center = prediction["x"]
    y_center = prediction["y"]
    width = prediction["width"]
    height = prediction["height"]
    confidence = prediction["confidence"]
    class_name = prediction["class"]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

    label = f"{class_name} ({confidence:.2f})"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text((x1, y1 - 20), label, fill="green", font=font)

    return image

# Streamlit main function
def main():
    st.title("Drowning Detection - Minor Project")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            st.image(image, caption="Original Image", use_container_width=True)

            # Save image for prediction
            with open("temp_uploaded.jpg", "wb") as f:
                f.write(file_bytes)

            # Initialize model and make prediction
            model = DrowningDetectionCNN()
            results = model.predict(path="temp_uploaded.jpg")

            st.write("🧪 Raw Results:", results)  # Debug line (optional)

            pred = results.get("predictions")

            # Safety check for predictions
            if pred is None or not isinstance(pred, dict):
                st.error("❌ No prediction returned. Please try a different image or check your model setup.")
                return

            # Check confidence and decide
            confidence = pred.get("confidence", 0)
            is_drowning = confidence < 0.89

            # Annotate the image
            annotated_image = annotate_image(image.copy(), pred)
            st.image(annotated_image, caption="Annotated Image", use_container_width=True)

            # Result messages
            if is_drowning:
                st.error("⚠️ Drowning Detected!")
            else:
                st.success("✅ No Drowning Detected.")

            st.info(f"Detected: {pred['class']} with {confidence:.2f} confidence")

        except Exception as e:
            st.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
