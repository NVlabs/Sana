#/bin/bash
set -e


echo "Testing inference"
bash tests/bash/test_inference.sh

echo "Testing training"
bash tests/bash/test_training_1epoch.sh
