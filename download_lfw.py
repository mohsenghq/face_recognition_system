from sklearn.datasets import fetch_lfw_people

# More comprehensive download
lfw_dataset = fetch_lfw_people(
    data_home='./face_database',  # Download location
    funneled=True,           # Use aligned faces
)