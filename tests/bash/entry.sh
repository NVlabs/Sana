#/bin/bash
set -e


echo "Testing inference"
bash tests/bash/inference/test_inference.sh

echo "Testing training"
bash tests/bash/training/test_training_all.sh
