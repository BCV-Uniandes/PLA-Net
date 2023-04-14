device=2
TARGET=ada
EXPERIMENT='/media/SSD5/pruiz/2022-2023/PLA-Net/log/ISBI/PLA-Net/'
BS=10

CUDA_VISIBLE_DEVICES=$DEVICE python inference.py --freeze_molecule --use_gpu --conv_encode_edge --learn_t --batch_size $BS --binary --use_prot --target $TARGET --inference_path $EXPERIMENT
