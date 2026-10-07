# Planetary Defense Analytics

Asteroid data exploration and prediction app for Potentially Hazardous Asteroid (PHA) classification, diameter regression, and orbital clustering.

## Run the Streamlit app

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads the saved model artifacts from `Models/` and catalogue data from `Data/`.

## Data files

The asteroid catalogue CSVs are tracked with Git LFS because they are larger than GitHub's regular file-size limit:

- `Data/pha_dataset.csv`
- `Data/pha_dataset_cleaned.csv`

Install Git LFS and fetch the data objects after cloning:

```powershell
git lfs install
git lfs pull
```

## Project contents

- `Notebook/` — data preparation, model training/evaluation, clustering, and explanations.
- `Models/` — serialized classifier, regressor, and K-Means artifacts.
- `app.py` — interactive Streamlit interface.

The model outputs are exploratory estimates, not a substitute for professional orbit determination or safety-critical analysis.
