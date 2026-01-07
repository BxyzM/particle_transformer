#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=1
#[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=3
samples_per_epoch=$((20000000 / $NGPUS))
samples_per_epoch_val=2000000  # 20% of training set
# Use fetch-step 1.0 to load full files, not just 1%
dataopts="--num-workers 2 --fetch-step 1.0"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kin"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# Binary classification type: TTbar_vs_QCD, WZ_vs_QCD, HToCC_vs_QCD
BINARY_TYPE=$3
[[ -z ${BINARY_TYPE} ]] && BINARY_TYPE="TTbar_vs_QCD"

# Set data files based on binary type
case ${BINARY_TYPE} in
    TTbar_vs_QCD)
        # Use single files with ~1M jets each to avoid sampling issues
        TRAIN_FILES="TTBar:${DATADIR}/train/TTBar_*.root QCD:${DATADIR}/train/ZJetsToNuNu_*.root"
        VAL_FILES="${DATADIR}/val_5M/TTBar_*.root ${DATADIR}/val_5M/ZJetsToNuNu_*.root"
        TEST_FILES="TTBar:${DATADIR}/test_20M/TTBar_*.root QCD:${DATADIR}/test_20M/ZJetsToNuNu_*.root"
        ;;
    WZ_vs_QCD)
        # Use single files with ~1M jets each to avoid sampling issues
        TRAIN_FILES="WToQQ:${DATADIR}/train/WToQQ_000.root ZToQQ:${DATADIR}/train/ZToQQ_000.root QCD:${DATADIR}/train/ZJetsToNuNu_000.root"
        VAL_FILES="${DATADIR}/val_5M/WToQQ_120.root ${DATADIR}/val_5M/ZToQQ_120.root ${DATADIR}/val_5M/ZJetsToNuNu_120.root"
        TEST_FILES="WToQQ:${DATADIR}/test_20M/WToQQ_100.root ZToQQ:${DATADIR}/test_20M/ZToQQ_100.root QCD:${DATADIR}/test_20M/ZJetsToNuNu_100.root"
        ;;
    HToCC_vs_QCD)
        # Use single files with ~1M jets each to avoid sampling issues
        TRAIN_FILES="HToCC:${DATADIR}/train/HToCC_000.root QCD:${DATADIR}/train/ZJetsToNuNu_000.root"
        VAL_FILES="${DATADIR}/val_5M/HToCC_120.root ${DATADIR}/val_5M/ZJetsToNuNu_120.root"
        TEST_FILES="HToCC:${DATADIR}/test_20M/HToCC_100.root QCD:${DATADIR}/test_20M/ZJetsToNuNu_100.root"
        ;;
    *)
        echo "Invalid binary type ${BINARY_TYPE}! Use: TTbar_vs_QCD, WZ_vs_QCD, or HToCC_vs_QCD"
        exit 1
        ;;
esac

$CMD \
    --data-train ${TRAIN_FILES} \
    --data-val ${VAL_FILES} \
    --data-test ${TEST_FILES} \
    --data-config data/JetClassBinary/JetClass_${BINARY_TYPE}_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClassBinary/${BINARY_TYPE}/${FEATURE_TYPE}/${model}/samples${samples_per_epoch}_epochs${epochs}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/JetClassBinary_${BINARY_TYPE}_${FEATURE_TYPE}_${model}_samples${samples_per_epoch}_epochs${epochs}_{auto}${suffix}.log  --predict-output pred_${BINARY_TYPE}.root \
    --tensorboard JetClassBinary_${BINARY_TYPE}_${FEATURE_TYPE}_${model}_samples${samples_per_epoch}_epochs${epochs}${suffix} \
    "${@:4}"

