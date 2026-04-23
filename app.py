import cv2
from model import DrowningDetectionCNN

def main():
    model = DrowningDetectionCNN()
    results = model.predict(path="1.jpg")
    
    image = cv2.imread("1.jpg")

    pred = results["predictions"]
    x_center = pred["x"]
    y_center = pred["y"]
    width = pred["width"]
    height = pred["height"]
    confidence = pred["confidence"]
    class_name = pred["class"]

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite("annotated_1.jpg", image)

    cv2.imshow("Drowning Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
