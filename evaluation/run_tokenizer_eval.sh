# resolution=128
resolution=256
SAMPLES_FOR_TEST=50000
BATCH_SIZE=10

# resolution=512
# SAMPLES_FOR_TEST=1000
# BATCH_SIZE=50

spatial_scale=1

configs_of_training_lists=()

for cfg_dir in "${configs_of_training_lists[@]}"
do
    configs_of_training=$cfg_dir"config.yaml"
    checkpoint_dir=$cfg_dir"models/"
    
    echo "Start testing on $configs_of_training"
    checkpoint_arr=()
    # for (( i=10; i<=10000; i+=10 ))
    # for (( i=10000; i<=10000000; i+=10000 ))
    # for (( i=80000; i<=80000; i+=1 ))
    for (( i=50000; i<=50000; i+=1 ))
    do
        checkpoint="eval_${i}.pt"
        # checkpoint="checkpoint_${i}.pt"
        if [ ! -f "${checkpoint_dir}${checkpoint}" ]; then
            break
        fi
        checkpoint_arr+=("$checkpoint")
    done

    for checkpoint in "${checkpoint_arr[@]}"
    do
        echo "-----------------------"
        echo "Testing on $checkpoint"
        NUMBER_OF_GPUS=4
        torchrun --nnodes=1 --nproc_per_node=$NUMBER_OF_GPUS ./evaluation/eval_tokenizer.py --config_file=$configs_of_training \
        --checkpoint_path=$checkpoint_dir$checkpoint --num_samples=$(($SAMPLES_FOR_TEST/$NUMBER_OF_GPUS)) --batch_size=$BATCH_SIZE \
        --resolution=$resolution --spatial_scale=$spatial_scale \
        opts evaluation.metrics=[mse,fid,is,lpips,psnr,ssim]
        # opts evaluation.metrics=[mse,fid,is,sfid,fdd,lpips,psnr,ssim]
    done

    echo "=====> Done"
    echo "@@@@@@@@@@@@@@@@@@@"
done

