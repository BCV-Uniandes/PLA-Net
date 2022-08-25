device=0
TARGET=akt2

### For Ligand Module
BS=2560
EXPERIMENT='LM/BINARY_'

CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 1 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET 
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 2 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET 
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 3 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET 
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 4 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET 
# Test
CUDA_VISIBLE_DEVICES=$DEVICE python ensamble.py --batch_size 30 --save $EXPERIMENT$TARGET --target $TARGET --use_gpu --conv_encode_edge --balanced_loader --binary

### For Protein Module using Ligand Module weights
BS=8
EXPERIMENT='LMPM/BINARY_'
LM_PATH='./log/LM'

CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 1 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 2 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 3 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 4 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET  --use_prot --LMPM  --freeze_molecule --model_load_init_path $LM_PATH
# Test
CUDA_VISIBLE_DEVICES=$DEVICE python ensamble.py --batch_size 30 --save $EXPERIMENT$TARGET --target $TARGET --use_gpu --conv_encode_edge --balanced_loader --binary  --use_prot --LMPM

### For Adversarial Training 
EXPERIMENT='AM/BINARY_'

CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 1 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --advs
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 2 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --advs
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 3 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --advs
CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 4 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET --advs
# Test
CUDA_VISIBLE_DEVICES=$DEVICE python ensamble.py --batch_size 30 --save $EXPERIMENT$TARGET --target $TARGET --use_gpu --conv_encode_edge --balanced_loader --binary 

### For training PLA-Net
EXPERIMENT='PLANet/BINARY_'
ADVS_PATH='./log/AM/'
PROTS_PATH='./log/LMPM/'

CUDA_VISIBLE_DEVICES=$DEVICE python main.py --cross_val 1 --model_load_init_path $ADVS_PATH --model_load_prot_init_path $PROTS_PATH --batch_size $BS --save $EXPERIMENT$TARGET --target $TARGET --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary  --use_prot  --PLANET --lr 5e-4
CUDA_VISIBLE_DEVICES=$DEVICE python main.py --cross_val 2 --model_load_init_path $ADVS_PATH --model_load_prot_init_path $PROTS_PATH --batch_size $BS --save $EXPERIMENT$TARGET --target $TARGET --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary  --use_prot --PLANET --lr 5e-4
CUDA_VISIBLE_DEVICES=$DEVICE python main.py --cross_val 3 --model_load_init_path $ADVS_PATH --model_load_prot_init_path $PROTS_PATH --batch_size $BS --save $EXPERIMENT$TARGET --target $TARGET --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary  --use_prot  --PLANET --lr 5e-4
CUDA_VISIBLE_DEVICES=$DEVICE python main.py --cross_val 4 --model_load_init_path $ADVS_PATH --model_load_prot_init_path $PROTS_PATH --batch_size $BS --save $EXPERIMENT$TARGET --target $TARGET --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary  --use_prot  --PLANET --lr 5e-4
CUDA_VISIBLE_DEVICES=$DEVICE python test_ensamble_planet.py --freeze_molecule --use_gpu --conv_encode_edge --learn_t --batch_size $BS --binary --use_prot --target $TARGET --save $EXPERIMENT

CUDA_VISIBLE_DEVICES=$device python main.py --use_gpu --conv_encode_edge --learn_t --cross_val 4 --save $EXPERIMENT$TARGET --batch_size $BS --balanced_loader --batch_size $BS --binary --target $TARGET 
#     # CUDA_VISIBLE_DEVICES=$DEVICE python ensamble.py --batch_size 30 --save $EXPERIMENT$TARGET --target $TARGET --freeze_molecule --use_gpu --conv_encode_edge --learn_t --balanced_loader  --binary  --use_prot 
# done