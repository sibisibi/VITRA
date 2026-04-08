cd /home/nas4_user/sibeenkim/work/VITRA

source /home/nas4_user/sibeenkim/anaconda3/etc/profile.d/conda.sh
conda activate vitra

DATA_DIR="/home/nas4_dataset/human/tasterob"
MANO_MODEL_DIR="/home/nas4_dataset/human/Code/models"

export CUDA_VISIBLE_DEVICES=$gpu
export EGL_DEVICE_ID=$gpu

python -m data.tools.process_tasterob \
    --data_dir "${DATA_DIR}" \
    --mano_model_dir "${MANO_MODEL_DIR}" \
    --index_path data/tasterob/index.csv \
    --moge_model_path weights/moge/model.pt \
    --postprocess \
    --save_all \
    --index 46396 # DoubleHand/Bathroom/21412