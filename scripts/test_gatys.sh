set -ex

python test.py --dataroot ./datasets/x2ka --name x2ka_gatys --model gatys --content_layer 1,3 --style_source total