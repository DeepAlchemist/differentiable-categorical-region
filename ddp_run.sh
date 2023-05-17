#
#CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
#  --config-file ./configs/EXP/OAP.yml OUTPUT_DIR "logs/mk/OAP/oap-glbFgPrt2-woBN"
#	CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
#	--eval-only --config-file ./configs/EXP/OAP.yml OUTPUT_DIR "logs/mk/OAP/test" \
#	MODEL.WEIGHTS '/home/caffe/code/fastReID/logs/mk/OAP/oap-glb-lcl1-v2/model_final.pth'
#CUDA_VISIBLE_DEVICES=0 python ./tests/cam_test.py \
#	--vis-only --config-file ./configs/EXP/CAMA.yml OUTPUT_DIR "logs/mk/test" \
#	MODEL.WEIGHTS '/home/caffe/code/fastReID/logs/mk/cama2t0.1stp3randcls/model_final.pth'

### Training
dataset='duke'
# bsl+sr
#CUDA_VISIBLE_DEVICES=1 python ./tools/train_net.py --config-file ./configs/EXP/GSS.yml MODEL.FREEZE_LAYERS '["stem", "middle", "transform", "part_transform"]' MODEL.BRANCH "['globe', 'part']" MODEL.NUM_PART 3 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False INPUT.RSE False OUTPUT_DIR "./logs/${dataset}/numk/gss-part3-softSRtau1"

# bsl+fg
#CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py --config-file ./configs/EXP/GSS.yml MODEL.FREEZE_LAYERS '["stem", "middle", "transform"]' MODEL.BRANCH "['globe', 'foreground']" MODEL.NUM_PART 0 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False INPUT.RSE False OUTPUT_DIR "./logs/${dataset}/gss-glb-fg"
# bsl+sr+fg(+pr)(+rse)
#CUDA_VISIBLE_DEVICES=2 python ./tools/train_net.py --config-file ./configs/EXP/GSS.yml MODEL.FREEZE_LAYERS '["stem", "middle", "transform", "part_transform", "fg_transform"]' MODEL.BRANCH "['globe', 'part', 'foreground']" MODEL.NUM_PART 3 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False INPUT.RSE True OUTPUT_DIR "./logs/${dataset}/gss-part3-fg-prKLtau10.woBg-rse" 

#TEST.METRIC partial

# Testing
#CUDA_VISIBLE_DEVICES=7 python ./tools/train_net.py --eval-only --config-file ./configs/EXP/GSS.yml MODEL.BRANCH "['globe', 'part']" MODEL.NUM_PART 7 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False OUTPUT_DIR "logs/test" 

#MODEL.WEIGHTS './logs/duke/gss-part3-nodetach/model_final.pth'

# Visualizing
CUDA_VISIBLE_DEVICES=7 python ./tests/probmap_test.py --vis-only --config-file ./configs/EXP/GSS.yml MODEL.BRANCH "['globe', 'foreground']" OUTPUT_DIR "logs/vis_only" MODEL.WEIGHTS './logs/duke/gss-glb-fg/model_final.pth' 
#CUDA_VISIBLE_DEVICES=7 python ./tests/probmap_test.py --vis-only --config-file ./configs/EXP/GSS.yml MODEL.BRANCH "['globe', 'part']" MODEL.NUM_PART 4 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False OUTPUT_DIR "logs/vis_only" MODEL.WEIGHTS './logs/duke/numk/gss-part4/model_final.pth' 
#CUDA_VISIBLE_DEVICES=6 python ./tests/probmap_test.py --vis-only --config-file ./configs/EXP/GSS.yml MODEL.BRANCH "['globe', 'part', 'foreground']" MODEL.NUM_PART 4 MODEL.BACKBONE.WITH_NL True MODEL.BACKBONE.WITH_NLKEY True OUTPUT_DIR "logs/vis_only" MODEL.WEIGHTS './logs/occluded_duke/gss-part4-fg-prKLtau10.woBg-rse-pab/model_final.pth' 


################
### Training ###
################
# * detaching feature map for segmentation head
# * opening fg_transform
#CUDA_VISIBLE_DEVICES=1 python ./tools/train_net.py \
#  --config-file ./configs/EXP/GSS.yml \
#  MODEL.NUM_BRANCH 2 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False \
#  INPUT.RSE False MODEL.LOSSES.OCCLUDED.ENABLED False \
#  OUTPUT_DIR "./logs/MSMT/Gss-glbFg" \
#  OUTPUT_DIR "./logs/Market/woFgGT/GssNlKey-glbFgPrt3-gtFG-triLs-occEn-c4_3-era" \ NlDivKey

##################
### Evaluation ###
##################
# './logs/Market/GssNlDivKey-glbFgPrt3-gtFG-triLs-c4_3-era/model_final.pth'
# './logs/OccDuke/0112/oap3occNlDivKey-glbFgPrt3-gtFG-triLs-c4_3-era/model_final.pth'
# './logs/OccDuke/woFgGT/GssNlKey-glbFgPrt3-triLs-c4_3-era/model_final.pth'
#CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py \
#	--eval-only --config-file ./configs/EXP/GSS.yml \
#	MODEL.NUM_BRANCH 2 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False \
#	OUTPUT_DIR "logs/test" \
#	MODEL.WEIGHTS './logs/Duke/Gss-glbFg-gtFG-triLs-c4_3/model_final.pth'

#####################
### Visualization ###
#####################
# * Probability Map, Occluded-Duke, Train: 430 138 116 198 735 3619 182
# * Probability Map, Occluded-Duke, Query: 27 39
# * Ranking Result: 4177_c7_f0059658,
#CUDA_VISIBLE_DEVICES=1 python ./tests/probmap_test.py --vis-only --config-file ./configs/EXP/GSS.yml MODEL.BRANCH "['globe', 'part', 'foreground']" MODEL.NUM_PART 3 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False OUTPUT_DIR "logs/vis_only" MODEL.WEIGHTS './logs/duke/debug/gss-part3-fg-prKLtau10.woBg-rse/model_final.pth' 
#MODEL.WEIGHTS './logs/Market/woFgGT/Gss-glbFg-triLs-c4_3/model_final.pth'
#MODEL.WEIGHTS './logs/Duke/Gss-glb-gtFG-triLs-c4_3/model_final.pth'

#CUDA_VISIBLE_DEVICES=0 python ./demo/visualize_result.py \
#	--config-file ./configs/EXP/GSS.yml --dataset-name OccludedDukeMTMC \
#	--num-vis 30 --vis-label \
#	--opts MODEL.NUM_BRANCH 5 MODEL.BACKBONE.WITH_NL True MODEL.BACKBONE.WITH_NLKEY True \
#	OUTPUT_DIR "logs/test" \
#	MODEL.WEIGHTS './logs/OccDuke/0112/oap3occNlDivKey-glbFgPrt3-gtFG-triLs-c4_3-era/model_final.pth'


#########################
### Stronger Baseline ###
#########################
#CUDA_VISIBLE_DEVICES=1 python ./tools/train_net.py \
#  --config-file ./configs/EXP/SBS.yml OUTPUT_DIR "logs/mk/SBS/sbs-bsl-Denorm"

########################
### Combine Datasets ###
########################
#CUDA_VISIBLE_DEVICES=2,3 python ./tools/train_net.py --num-gpus 2 \
#  --config-file ./configs/EXP/AGW.yml OUTPUT_DIR "logs/mk/XiangMu/agw-R101IBN-OimArcS30M0.5-final"
#CUDA_VISIBLE_DEVICES=7 python ./tools/train_net.py \
#  --eval-only --config-file ./configs/EXP/AGW.yml OUTPUT_DIR "logs/mk/XiangMu/test" \
#  MODEL.WEIGHTS '/home/caffe/code/fastReID/logs/mk/XiangMu/agw-R101IBN-OimArcS30M0.5-pretrained-384/model_final.pth'

