#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="3:59:00"
MYNTASKS="1"
MYCPU="5"
MYGRES="gpu:v100s:2"

JOB_INFO="train stepsize"
MODEL_NAME="CompVis/stable-diffusion-v1-4"
TRAIN_DIR="./haoyu/dataset"
MYCOMMEND="python train_text_to_image.py --pretrained_model_name_or_path=$MODEL_NAME --train_data_dir=$TRAIN_DIR --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=6 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=2000 --learning_rate=5e-06 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="results/sd-haoyu-model" --checkpointing_steps 100"

# MYCOMMEND="accelerate launch --main_process_port 9528 --gpu_ids '0,1' --num_processes 1 --num_machines 1 --mixed_precision="fp16" test.py"

MYCOMMEND2="python3 test.py -gpu_id 0 -model 1 -attack 1 --pgd_norm 7 -batch_size 50 -path Final/VANILLA_62162198_1/iter_50 --alpha 1000 --num_iter 100 --num_stop 2000 --test_subset --seed 1"

MYCOMMEND3="python3 test.py -gpu_id 0 -model 1 -attack 1 --pgd_norm 7 -batch_size 50 -path Final/VANILLA_62162198_1/iter_50 --alpha 5000 --num_iter 100 --num_stop 2000 --test_subset --seed 1"

MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"

cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
# cat ./slurm_files/sconfigs1_scavenger.sb > submit.sb
# cat ./slurm_files/sconfigs1.sb > submit.sb
echo "#SBATCH --time=${MYTIME}             # limit of wall clock time - how long the job will run (same as -t)" >> submit.sb
echo "#SBATCH --cpus-per-task=${MYCPU}           # number of CPUs (or cores) per task (same as -c)" >> submit.sb
echo "#SBATCH --ntasks=${MYNTASKS}                  # number of tasks - how many tasks (nodes) that you require (same as -n)" >> submit.sb
echo "#SBATCH --gres=${MYGRES}" >> submit.sb
# echo "#SBATCH --nodelist=nvl-001" >> submit.sb
echo "#SBATCH -o ${MY_JOB_ROOT_PATH}/logfile/%j.log" >> submit.sb
echo "#SBATCH -e ${MY_JOB_ROOT_PATH}/logfile/%j.err" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND} --job_id \${SLURM_JOB_ID}_1\"" >> submit.sb
echo "MYCOMMEND2=\"${MYCOMMEND2} --job_id \${SLURM_JOB_ID}_2\"" >> submit.sb
echo "MYCOMMEND3=\"${MYCOMMEND3} --job_id \${SLURM_JOB_ID}_3\"" >> submit.sb
cat ./slurm_files/sconfigs3.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --time=${MYTIME}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --cpus-per-task=${MYCPU}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --gres=${MYGRES}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
