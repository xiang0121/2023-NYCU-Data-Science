## Dataset
- Unzip
    ```sh
    unzip 311511056.zip -d ./311511056
    ```
- Folder structure
    ```
    .
    ├── 311511056.sh
    ├── pretrain.py
    ├── fine_tune.py
    ├── README.md
    ├── requirement.txt
    └── pretrain_weight.pth
    ```

## Environment
- Python 3.6 or later version
    ```sh
    pip install -r requirements.txt
    ```

## Train
```sh
python pretrain.py
```

## Test & Write CSV
```sh
python fine_tune.py
```
The output csv file is `submission.csv`.

## Run the scrip
```sh
sh 311511056.sh
```
* 311511056.sh
    ```sh
    # Train the model
    python pretrain.py

    # Gererate the csv file
    python fine_tune.py
    ```