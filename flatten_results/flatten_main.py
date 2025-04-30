from src.clustering_images_fish_features import main


main("../data/black_flattened_image.csv", "/black_nc=2", n_components=2)
main("../data/black_flattened_image.csv", "/black_nc=3", n_components=3)
