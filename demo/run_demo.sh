python demo/visualize_result.py --config-file logs/dukemtmc/mgn_R50-ibn/config.yaml \
--parallel --vis-label --dataset-name 'DukeMTMC' --output logs/mgn_duke_vis \
--opts MODEL.WEIGHTS logs/dukemtmc/mgn_R50-ibn/model_final.pth

#python demo.py --config-file /home/caffe/code/fastReID/configs/EXP/AGW.yml \
#--input /home/caffe/code/fastReID/tools/deploy/test_data/*.jpg --output /home/caffe/code/fastReID/tools/deploy/torch_output/ \
#--opts MODEL.WEIGHTS /home/caffe/code/fastReID/logs/mk/agw-oim-d2048s30m0.5/model_0175199.pth
