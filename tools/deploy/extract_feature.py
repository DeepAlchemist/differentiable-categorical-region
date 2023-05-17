import argparse
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime
import tqdm

# python ./tools/deploy/extract_feature.py --model-path /home/caffe/code/fastReID/tools/deploy/outputs/onnx_model/agw.onnx --gallery-dir /home/caffe/code/Pedestron/cache/gallery_bbox/ --probe-path /home/caffe/code/Pedestron/cache/probe_bbox/512_306_578_506_1056.jpg
def get_parser():
    parser = argparse.ArgumentParser(description="onnx model inference")

    parser.add_argument(
        "--model-path",
        default="onnx_model/baseline.onnx",
        help="onnx model path",
        # nargs="+",
    )
    parser.add_argument(
        "--gallery-dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--probe-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="height of image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="width of image"
    )
    return parser

def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img

def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)

def get_score(fea1, fea2):
    score = np.sum(np.array(fea1) * np.array(fea2))
    return score

if __name__ == "__main__":
    ### Single model
    args = get_parser().parse_args()

    ort_sess = onnxruntime.InferenceSession(args.model_path)

    input_name = ort_sess.get_inputs()[0].name

    # img_dir = '/home/caffe/code/abd-net/images/'
    # img_paths = ['id1_1.jpg', 'id1_2.jpg', 'id2_1.jpg', 'id2_1.jpg']
    # img_paths = [img_dir + it for it in img_paths]
    # features = []
    # for path in tqdm.tqdm(img_paths):
    #     image = preprocess(path, args.height, args.width)
    #     feat = ort_sess.run(None, {input_name: image})[0]
    #     feat = normalize(feat, axis=1)
    #     features.append(feat)
    #
    # f1_1, f1_2, f2_1, f2_2 = features
    # print(get_score(f1_1, f1_1), get_score(f1_1, f1_2), get_score(f1_1, f2_1), get_score(f1_1, f2_2))
    # print(get_score(f1_2, f1_1), get_score(f1_2, f1_2), get_score(f1_2, f2_1), get_score(f1_2, f2_2))
    # print(get_score(f2_1, f1_1), get_score(f2_1, f1_2), get_score(f2_1, f2_1), get_score(f2_1, f2_2))
    # print(get_score(f2_2, f1_1), get_score(f2_2, f1_2), get_score(f2_2, f2_1), get_score(f2_2, f2_2))

    # probe
    p_path = args.probe_path
    image = preprocess(p_path, args.height, args.width)
    p_feat = ort_sess.run(None, {input_name: image})[0]
    p_feat = normalize(p_feat, axis=1)

    # gallery
    g_dir = args.gallery_dir
    g_paths = os.listdir(g_dir)
    g_paths = [os.path.join(g_dir, item) for item in g_paths if item.endswith('.jpg')]

    sim_dict = dict()
    for path in tqdm.tqdm(g_paths):
        image = preprocess(path, args.height, args.width)
        feat = ort_sess.run(None, {input_name: image})[0]
        feat = normalize(feat, axis=1)
        sim = get_score(p_feat, feat)
        sim_dict.update({os.path.basename(path).split('.')[0]: str(sim)})

    names = ['name', 'sim']
    formats = ['S30', 'f8']
    dtype = dict(names=names, formats=formats)
    arr = np.array(list(sim_dict.items()), dtype=dtype)
    arr = np.sort(arr, order='sim')[::-1]
    np.save('./rank_result.npy', arr)

    # ---------------------------------------------------------------
    ### Multiple model
    # args = get_parser().parse_args()
    # ort_sesses = [onnxruntime.InferenceSession(p) for p in args.model_path]
    # input_names = [ort_sess.get_inputs()[0].name for ort_sess in ort_sesses]
    #
    # img_dir = '/home/caffe/code/abd-net/images/'
    # img_paths = ['id1_1.jpg', 'id1_2.jpg', 'id2_1.jpg', 'id2_1.jpg']
    # img_paths = [img_dir + it for it in img_paths]
    # features = []
    # for path in tqdm.tqdm(img_paths):
    #     image = preprocess(path, args.height, args.width)
    #     feats = [ort_sess.run(None, {input_name: image})[0] for ort_sess, input_name in zip(ort_sesses, input_names)]
    #     feats = [normalize(feat, axis=1) for feat in feats]
    #     features.append(np.concatenate(feats, axis=1))
    #
    # f1_1, f1_2, f2_1, f2_2 = features
    # print(get_score(f1_1, f1_1), get_score(f1_1, f1_2), get_score(f1_1, f2_1), get_score(f1_1, f2_2))
    # print(get_score(f1_2, f1_1), get_score(f1_2, f1_2), get_score(f1_2, f2_1), get_score(f1_2, f2_2))
    # print(get_score(f2_1, f1_1), get_score(f2_1, f1_2), get_score(f2_1, f2_1), get_score(f2_1, f2_2))
    # print(get_score(f2_2, f1_1), get_score(f2_2, f1_2), get_score(f2_2, f2_1), get_score(f2_2, f2_2))
