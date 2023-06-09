### Code of Learning Differentiable Categorical Region with Gumbel-Softmax for Person Re-identification

- Baseline

``` 
CUDA_VISIBLE_DEVICES=0 python ./tools/train_net.py --config-file ./configs/EXP/GSS.yml MODEL.FREEZE_LAYERS '["stem", "middle", "transform"]' MODEL.BRANCH "['globe', ]" MODEL.NUM_PART 0 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False INPUT.RSE False OUTPUT_DIR "./logs/${dataset}/gss-glb"
```

- DCR

```
CUDA_VISIBLE_DEVICES=1 python ./tools/train_net.py --config-file ./configs/EXP/GSS.yml MODEL.FREEZE_LAYERS '["stem", "middle", "transform", "part_transform"]' MODEL.BRANCH "['globe', 'part']" MODEL.NUM_PART 3 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False INPUT.RSE False OUTPUT_DIR "./logs/${dataset}/numk/gss-part3-dcr"
```

- DCR+

```
CUDA_VISIBLE_DEVICES=2 python ./tools/train_net.py --config-file ./configs/EXP/GSS.yml MODEL.FREEZE_LAYERS '["stem", "middle", "transform", "part_transform", "fg_transform"]' MODEL.BRANCH "['globe', 'part', 'foreground']" MODEL.NUM_PART 3 MODEL.BACKBONE.WITH_NL False MODEL.BACKBONE.WITH_NLKEY False INPUT.RSE True OUTPUT_DIR "./logs/${dataset}/gss-part3-DCR+"
```



