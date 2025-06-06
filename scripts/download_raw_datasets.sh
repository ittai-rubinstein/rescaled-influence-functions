#!/bin/bash

# Source environment
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh" || {
    echo "Failed to source environment variables." >&2
    exit 1
}

# Set raw data path
RAW_DATA_PATH="$DATA_DIRECTORY/datasets/raw_data"
mkdir -p "$RAW_DATA_PATH"
export RAW_DATA_PATH

# Helper: error if no dataset args given
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [sst2 imdb adult tape esc50 cdr diabetes dogfish spam]"
    exit 1
fi

# Loop over all args
for dataset in "$@"; do
    case "$dataset" in
        sst2)
            echo "Downloading SST-2..."
            curl -L -o "$RAW_DATA_PATH/SST-2.zip" https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
            unzip -o "$RAW_DATA_PATH/SST-2.zip" -d "$RAW_DATA_PATH/"
            ;;
        imdb)
            echo "Downloading IMDb from Kaggle..."
            echo "NOTE: Ensure Kaggle API token is set at ~/.kaggle/kaggle.json"
            kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p "$RAW_DATA_PATH/"
            unzip -o "$RAW_DATA_PATH/imdb-dataset-of-50k-movie-reviews.zip" -d "$RAW_DATA_PATH/"
            mv "$RAW_DATA_PATH/IMDB Dataset.csv" "$RAW_DATA_PATH/imdb.csv"
            ;;
        adult)
            echo "Downloading Adult (Census Income) dataset..."
            curl -L -o "$RAW_DATA_PATH/adult.data" https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
            curl -L -o "$RAW_DATA_PATH/adult.test" https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
            curl -L -o "$RAW_DATA_PATH/adult.names" https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
            ;;
        tape)
            echo "Downloading TAPE model files..."
            wget -P "$RAW_DATA_PATH/" http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/pfam.model
            wget -P "$RAW_DATA_PATH/" http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/pfam.vocab

            echo "Downloading TAPE datasets..."
            mkdir -p "$RAW_DATA_PATH/tape"
            (
              cd "$RAW_DATA_PATH/tape" || exit 1
              wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz
              wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/proteinnet.tar.gz
              wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz
              wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz
              wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz

              echo "Extracting TAPE datasets..."
              for file in *.tar.gz; do
                  tar -xzf "$file"
                  rm "$file"
              done
            )
            ;;
        esc50)
            echo "Downloading ESC-50 dataset..."
            mkdir -p "$RAW_DATA_PATH/ESC-50"
            (
              cd "$RAW_DATA_PATH/ESC-50" || exit 1
              wget https://github.com/karoldvl/ESC-50/archive/master.zip -O ESC-50-master.zip
              unzip -o ESC-50-master.zip
              mv ESC-50-master/audio . || true
              mv ESC-50-master/meta . || true
              rm -rf ESC-50-master ESC-50-master.zip
            )
            ;;
        cdr)
            echo "Downloading CDR dataset..."
            mkdir -p "$RAW_DATA_PATH/cdr"
            curl -L -o "$RAW_DATA_PATH/cdr/cdr.db" https://worksheets.codalab.org/rest/bundles/0x38e021ee339a4f6aa21d9c4a16c1b7ee/contents/blob/
            ;;
        diabetes)
            echo "Downloading Diabetes dataset..."
            mkdir -p "$RAW_DATA_PATH/diabetes"
            curl -L -o "$RAW_DATA_PATH/diabetes/dataset_diabetes.zip" https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
            unzip -o "$RAW_DATA_PATH/diabetes/dataset_diabetes.zip" -d "$RAW_DATA_PATH/diabetes/"
            ;;
        dogfish)
            echo "Downloading Dogfish dataset..."
            mkdir -p "$RAW_DATA_PATH/dogfish"
            wget -O "$RAW_DATA_PATH/dogfish/dogfish_train.npz" http://mitra.stanford.edu/kundaje/pangwei/dogfish_900_300_inception_features_train.npz
            wget -O "$RAW_DATA_PATH/dogfish/dogfish_test.npz" http://mitra.stanford.edu/kundaje/pangwei/dogfish_900_300_inception_features_test.npz
            ;;
        spam)
            echo "Downloading Enron Spam dataset..."
            mkdir -p "$RAW_DATA_PATH/enron"
            # Not the safest way to download things but their SSL certificate doesn't work...
            wget --no-check-certificate -O "$RAW_DATA_PATH/enron/enron1.tar.gz" http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz
            (
              cd "$RAW_DATA_PATH/enron" || exit 1
              tar -xzf enron1.tar.gz
            )
            ;;
        *)
            echo "Unknown dataset: $dataset" >&2
            ;;
    esac
done

echo "All requested datasets have been downloaded to $RAW_DATA_PATH."

