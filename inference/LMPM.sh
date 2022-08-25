WEIGHTS_PATHS='/media/SSD1/pruiz/PLA-Net/PLA-Net/LMPM'
#WEIGHTS_PATHS='/media/SSD1/pruiz/PLA-Net/ScientificReports/LMPM'
DEVICE=3
python ensamble.py --device $DEVICE --batch_size 60 --save $WEIGHTS_PATHS --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary --use_prot

