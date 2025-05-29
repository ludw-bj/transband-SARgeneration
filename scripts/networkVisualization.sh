set -ex
python VggVisualization.py --model VGG16 --use_gpu --layers_ids 15,22
python GramVisualization.py --model VGG16 --use_gpu --layers_ids 3

