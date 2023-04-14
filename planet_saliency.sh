EXPERIMENT='BINARY___'
LOAD_EXPERIMENT='/data/pruiz/PLA-Net/LM/BINARY_'
FOLD='/Fold4/'
MODEL='BS_2560-NF_full_valid_best.pth'
TARGET='ada' # 'hivpr' 'pur2' 'tysy')
CROSSVAL=4

CUDA_VISIBLE_DEVICES=0 python saliency_maps.py --use_gpu --conv_encode_edge --num_layers 20 --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.2 --batch_size 1 --cross_val $CROSSVAL --save $EXPERIMENT$TARGET --model_load_init_path $LOAD_EXPERIMENT$t$FOLD  --binary --target $TARGET --epochs 300  --lr 5e-4 --saliency --use_prot
