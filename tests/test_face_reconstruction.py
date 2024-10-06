import cv2
from matplotlib import pyplot as plt

from app.face_reconstruction import FaceReconstructor


def init_reconstructor() -> None:
    FaceReconstructor(
        root_dir="..",
        models_dir="../models",
        device="cuda",
        download_models=True
    )


def test_image(test_image_path: str) -> None:
    reconstructor = FaceReconstructor(
        root_dir="..",
        models_dir="../models",
        device="cuda"
    )

    test_image = cv2.imread(test_image_path)

    plt.figure(figsize=(6, 6))
    plt.imshow(test_image[:, :, ::-1])
    plt.axis("off")
    plt.title("Input Image")
    plt.show()

    images = reconstructor.generate_similar_images(test_image, num_images=4)

    plt.subplots(2, 2, figsize=(6, 6))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(f"Image {i + 1}")
    plt.show()


if __name__ == '__main__':
    init_reconstructor()
    test_image("../assets/examples/max.jpg")
    test_image("../assets/examples/artem.jpg")
