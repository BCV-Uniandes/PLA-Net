WEIGHTS_PATHS='/media/SSD1/pruiz/PLA-Net/ScientificReports/LM+Advs'
DEVICE=2
python ensamble.py --device $DEVICE --batch_size 30 --save $WEIGHTS_PATHS --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary --advs

