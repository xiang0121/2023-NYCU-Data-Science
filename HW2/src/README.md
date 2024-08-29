## Dataset
- Unzip 311511056.zip to `./311511056`
    ```sh
    unzip 311511056.zip -d ./311511056
    ```
- Folder structure
    ```
    .
    ├── 31151156.py
    ├── README.md
    └── best_model_09431.pth
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python 311511056.py train
```

## Test & Write CSV
```sh
python 311511056.py test
```
The output csv file is `submission.csv`.