
import easyocr
import cv2
import matplotlib.pyplot as plt

# Step 1: Load the pre-trained OCR model
reader = easyocr.Reader(['en'])

# Step 2: Read input image
image_path = input("Enter image path: ")
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
else:
    # Step 3: Perform text recognition
    results = reader.readtext(image)

    # Step 4: Display extracted text
    print("\n========== RECOGNIZED TEXT ==========\n")

    for result in results:
        text = result[1]
        confidence = result[2]

        print("Text:", text)
        print("Confidence:", round(confidence * 100, 2), "%")
        print("--------------------------------")

    # Step 5: Draw bounding boxes around detected text
    for result in results:
        points = result[0]
        points = [(int(x), int(y)) for x, y in points]

        for i in range(4):
            cv2.line(
                image,
                points[i],
                points[(i + 1) % 4],
                (0, 255, 0),
                2
            )

    # Step 6: Display output image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title("Text Recognition Output")
    plt.axis("off")
    plt.show()
