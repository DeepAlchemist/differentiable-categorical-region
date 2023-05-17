#python caffe_export.py --config-file /export/home/lxy/cvpalgo-fast-reid/logs/dukemtmc/R34/config.yaml \
#--name "baseline_R34" \
#--output logs/caffe_R34 \
#--opts MODEL.WEIGHTS /export/home/lxy/cvpalgo-fast-reid/logs/dukemtmc/R34/model_final.pth

#python onnx_export.py --config-file /home/caffe/code/fastReID/configs/EXP/MGN.yml \
#--name "mgn" \
#--output outputs/onnx_model \
#--opts MODEL.WEIGHTS /home/caffe/code/fastReID/logs/mk/XiangMu/mgn-R50IBN-OimArcS30/model_final.pth

#python onnx_inference.py --model-path outputs/onnx_model/agw_v2.onnx \
# --input test_data/*.jpg --height 384 --output onnx_output

#python extract_feature.py --model-path "/home/caffe/code/fastReID/tools/deploy/outputs/onnx_model/mgn.onnx"

