import argparse
import importlib
import logging
import os

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine

# 从 pipeline 模块导入人脸处理组件
# 假设 pipeline 目录及其内容 (facealign.py, facedetect.py, facefeatures.py) 存在
from pipeline.facealign import FaceAlign
from pipeline.facedetect import FaceDetect
from pipeline.facefeatures import FaceFeatures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局变量用于存储模型实例，避免重复加载
_face_detector = None
_face_align = None
_face_features = None

def initialize_models(retinaface_weights, arcface_weights, retinaface_config_path, use_cuda=True):
    """
    初始化人脸检测、对齐和特征提取模型。
    Args:
        retinaface_weights (str): RetinaFace 模型的权重文件路径。
        arcface_weights (str): ArcFace 模型的权重文件路径。
        retinaface_config_path (str): RetinaFace 配置文件的路径（例如：RetinaFace.data.config）。
        use_cuda (bool): 是否使用 CUDA (GPU) 进行推理。
    """
    global _face_detector, _face_align, _face_features

    if _face_detector is None:
        try:
            # 动态导入 RetinaFace 配置
            cfg_module = importlib.import_module(retinaface_config_path)
            cfg = cfg_module.cfg_re50 # 假设配置文件中包含 cfg_re50
            
            _face_detector = FaceDetect(
                retinaface_weights,
                cuda=use_cuda,
                cfg=cfg,
            )
            logger.info(f"RetinaFace 检测器加载成功: {retinaface_weights}")
        except Exception as e:
            logger.error(f"加载 RetinaFace 检测器失败: {e}", exc_info=True)
            raise

    if _face_align is None:
        try:
            _face_align = FaceAlign(target_size=112)
            logger.info("人脸对齐器加载成功。")
        except Exception as e:
            logger.error(f"加载人脸对齐器失败: {e}", exc_info=True)
            raise

    if _face_features is None:
        try:
            _face_features = FaceFeatures(
                arcface_weights,
                cuda=use_cuda,
            )
            logger.info(f"ArcFace 特征提取器加载成功: {arcface_weights}")
        except Exception as e:
            logger.error(f"加载 ArcFace 特征提取器失败: {e}", exc_info=True)
            raise

def load_image(image_path):
    """
    加载图像文件。如果图像宽度大于 640 像素，则将其缩小到 640 像素并等比例调整高度。
    Args:
        image_path (str): 图像文件的路径。
    Returns:
        numpy.array: 加载并转换为 OpenCV BGR 格式的图像。
    Raises:
        FileNotFoundError: 如果图像文件不存在。
        ValueError: 如果无法加载或转换图像。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"错误: 找不到图像文件: {image_path}。请检查路径是否正确。")

    try:
        # 使用 PIL 读取图像以获得更好的兼容性，然后转换为 OpenCV 格式
        pil_image = Image.open(image_path)
        # 将 PIL 图像转换为 OpenCV 格式 (RGB 转 BGR)
        image = np.array(pil_image.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"错误: 无法加载或转换图像 {image_path}: {e}") from e

    if image is None:
        raise ValueError(f"错误: 无法从路径 {image_path} 加载图像。请检查路径是否正确。")

    # 检查图像宽度，如果大于 640 像素则进行缩小
    max_width = 640
    height, width, _ = image.shape
    if width > max_width:
        new_width = max_width
        new_height = int(height * (new_width / width))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"图像 {image_path} 宽度已从 {width} 缩小到 {new_width}。")

    return image

def extract_face_features(img_path):
    """
    从给定图片路径提取人脸特征。
    Args:
        img_path (str): 图片文件路径。
    Returns:
        numpy.array: 人脸特征向量。
    Raises:
        FileNotFoundError: 当图片文件不存在时。
        ValueError: 当检测不到人脸或特征提取失败时。
    """
    # 检查输入文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片文件不存在: {img_path}")

    try:
        # 加载图片
        image = load_image(img_path)

        # 人脸检测
        faces, landmarks, keypoints = _face_detector(image)
        if len(faces) == 0:
            raise ValueError("未检测到人脸")

        # 人脸对齐
        aligned_faces = _face_align(image, faces, landmarks)

        # 特征提取
        features = _face_features(aligned_faces)
        if len(features) == 0 or features[0] is None:
            raise ValueError("特征提取失败")

    except Exception as e:
        # 重新抛出更具体的异常
        raise ValueError(f"无法提取人脸特征: {str(e)}") from e

    # 返回第一个检测到的人脸特征
    return np.array(features[0])

def compare_face_features(feature1, feature2, similarity_threshold=0.60):
    """
    比较两个人脸特征向量，判断是否为同一个人。
    Args:
        feature1 (numpy.array): 第一个人脸的特征向量。
        feature2 (numpy.array): 第二个人脸的特征向量。
        similarity_threshold (float): 余弦相似度阈值。高于此阈值则认为为同一个人。
                                      ArcFace 通常推荐 0.60 - 0.64 左右。
    Returns:
        tuple: 包含以下元素的元组 (cosine_distance, similarity, is_same_person)
               - cosine_distance (float): 余弦距离得分。
               - similarity (float): 余弦相似度得分。
               - is_same_person (bool): 根据推荐阈值判断是否为同一个人。
    """
    # 计算余弦距离
    # 注意：scipy.spatial.distance.cosine 返回的是余弦“距离”，而不是相似度。
    # 余弦相似度 = 1 - 余弦距离
    cosine_distance = cosine(feature1, feature2)
    similarity = 1 - cosine_distance
    logger.info(f"计算得到余弦距离: {cosine_distance:.4f}, 余弦相似度: {similarity:.4f}")

    # 判断是否为同一个人
    is_same_person = similarity >= similarity_threshold
    logger.info(f"根据阈值 {similarity_threshold:.2f} 判断，是否为同一个人: {is_same_person}")
    return cosine_distance, similarity, is_same_person

def main():
    """
    主函数：解析命令行参数，初始化模型，执行人脸比对。
    """
    parser = argparse.ArgumentParser(
        description="比较两张人脸图像的相似度",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("image_path1", help="第一张人脸图像的路径")
    parser.add_argument("image_path2", help="第二张人脸图像的路径")
    parser.add_argument(
        "--retinaface_weights",
        type=str,
        default="./weights/retinaface/Resnet50_Gender_Final.pth",
        help="RetinaFace 模型的权重文件路径",
    )
    parser.add_argument(
        "--arcface_weights",
        type=str,
        default="./weights/arcface/arcface_resnet50_epoch_30.pth",
        help="ArcFace 模型的权重文件路径",
    )
    parser.add_argument(
        "--retinaface_config",
        type=str,
        default="RetinaFace.data.config",
        help="RetinaFace 配置文件的导入路径 (例如: RetinaFace.data.config)",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="不使用 CUDA (GPU) 进行推理，强制使用 CPU",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.60,
        help="判断是否为同一个人的余弦相似度阈值",
    )

    args = parser.parse_args()

    logger.info(f"开始比较图像: {args.image_path1} 和 {args.image_path2}")

    try:
        # 初始化模型
        initialize_models(
            args.retinaface_weights,
            args.arcface_weights,
            args.retinaface_config,
            use_cuda=not args.no_cuda,
        )

        # 提取两个人脸的特征向量
        logger.info(f"正在从 {args.image_path1} 提取人脸特征...")
        face1_feature = extract_face_features(args.image_path1)
        logger.info(f"正在从 {args.image_path2} 提取人脸特征...")
        face2_feature = extract_face_features(args.image_path2)
        logger.info("人脸特征提取完成。")

        # 比较人脸特征
        cosine_distance, similarity, is_same_person_bool = compare_face_features(
            face1_feature, face2_feature, args.similarity_threshold
        )

        # 将布尔值转换为中文的“是”或“否”
        is_same_person_str = "是" if is_same_person_bool else "否"

        logger.info(
            f"最终结果 - 余弦相似度: {similarity:.4f}, 是否同一个人: {is_same_person_str}"
        )

    except FileNotFoundError as e:
        logger.error(f"文件错误: {e}")
    except ValueError as e:
        logger.error(f"人脸处理错误: {e}")
    except Exception as e:
        logger.error(f"发生未知错误: {e}", exc_info=True)

    logger.info("图像比较任务完成。")

if __name__ == "__main__":
    main()
