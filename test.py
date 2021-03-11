import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2

#用于装载和使用初始模型的函数和类。
import inception
# inception.data_dir = 'inception/'
model = inception.Inception()
resized_image = model.resized_image
y_pred = model.y_pred
y_logits = model.y_logits

#将初始模型的图设置为默认图，
#所以这个with_block中的所有变化都被画出来了。
with model.graph.as_default():

    #为目标类编号添加一个占位符变量。
    pl_cls_target = tf.compat.v1.placeholder(dtype=tf.int32)
    #添加一个新的损失函数。
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])

    #获取关于损失函数的梯度。
    #调整输入图像的大小。
    gradient = tf.gradients(loss, resized_image)

session = tf.compat.v1.Session(graph=model.graph)


def find_adversary_noise(image_path, cls_target, noise_limit=3.0,
                         required_score=0.99, max_iterations=100):
    """
    找出必须添加到给定图像的噪声。
    它被分类为目标类。

    image_path: 输入的图像路径(必须是 *.jpg 文件).
    cls_target: 目标类数(1-1000之间的整数)。
    noise_limit: 噪声中像素值的限制。
    required_score: 当目标类分数达到这个值时停止。
    max_iterations: 执行的优化迭代的最大数量。
    """

    # 用图像创建一个封条。
    feed_dict = model._create_feed_dict(image_path=image_path)

    # 使用TensorFlow计算预测的类得分。
    pred, image = session.run([y_pred, resized_image],
                              feed_dict=feed_dict)

    # 转换为一维数组。
    pred = np.squeeze(pred)

    # 预测类数。
    cls_source = np.argmax(pred)

    # probability or confidence
    score_source_org = pred.max()

    # 源和目标类的名称。
    name_source = model.name_lookup.cls_to_name(cls_source,
                                                only_first_name=True)
    name_target = model.name_lookup.cls_to_name(cls_target,
                                                only_first_name=True)

    # 将噪声初始化为零。
    noise = 0

    # 执行一些优化迭代来寻找。
    # 造成输入图像错误分类的噪声。
    for i in range(max_iterations):
        print("Iteration:", i)

        # 噪声图像仅仅是输入图像和噪声的总和。
        noisy_image = image + noise

        noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=255.0)

        feed_dict = {model.tensor_name_resized_image: noisy_image,
                     pl_cls_target: cls_target}

        # 计算预测的类分数和梯度。
        pred, grad = session.run([y_pred, gradient],
                                 feed_dict=feed_dict)

        # 将预测的类分数转换为一个单一的数组。
        pred = np.squeeze(pred)

        # 源和目标类的得分(概率)。
        score_source = pred[cls_source]
        score_target = pred[cls_target]

        # 压缩渐变数组的维度。
        grad = np.array(grad).squeeze()

        grad_absmax = np.abs(grad).max()

        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        step_size = 7 / grad_absmax
        
        # 为source等类打印分数。
        msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))

        msg = "Target score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))

        # 打印梯度的统计数据。
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))

        # 换行
        print()

        # 如果目标类的分数不够高。
        if score_target < required_score:

            #通过减少梯度来更新图像噪声。
            noise -= step_size * grad

            # 确保噪音在预期范围内。
            # 这避免了扭曲图像太多。
            noise = np.clip(a=noise,
                            a_min=-noise_limit,
                            a_max=noise_limit)
        else:
            # 取消优化，因为分数足够高。
            break

    return image.squeeze(), noisy_image.squeeze(), noise, \
           name_source, name_target, \
           score_source, score_source_org, score_target

def normalize_image(x):
    # 获取输入中所有像素的最小值和最大值。
    x_min = x.min()
    x_max = x.max()

    # 标准化，所有值都在0.0到1.0之间。
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def plot_images(image, noise, noisy_image,
                name_source, name_target,
                score_source, score_source_org, score_target):
    """
    绘制图像、噪声图像和噪声。还显示了类名和分数。
    注意，噪音被放大，以使用全系列的颜色，否则，如果噪音很低，那将很难看到。

    image: 原始输入图像
    noise: 噪声
    noisy_image: 加入噪声的图像
    name_source: Name of the source-class.
    name_target: Name of the target-class.
    score_source: Score for the source-class.
    score_source_org: Original score for the source-class.
    score_target: Score for the target-class.
    """

    # 创建图
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # 调整垂直间距。
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # 使用插值平滑像素
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    ax = axes.flat[0]
    ax.imshow(image / 255.0, interpolation=interpolation)
    #msg = "Original Image:\n{0} ({1:.2%})"
    #msg = "Filpped Original Image:\n{0} ({1:.2%})"
    #msg = "Adversarial Image:\n{0} ({1:.2%})"
    msg = "Filpped Adversarial Image:\n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)

    ax = axes.flat[1]
    ax.imshow(noisy_image / 255.0, interpolation=interpolation)
    
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(noisy_image / 255.0)
    plt.savefig('images/GrannySmithto141.png', bbox_inches='tight',pad_inches=0.0)
    
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)

    # Plot the noise.
    # 颜色被放大了，否则很难看到。
    ax = axes.flat[2]
    ax.imshow(normalize_image(noise), interpolation=interpolation)
    xlabel = "Amplified Noise"
    ax.set_xlabel(xlabel)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    #显示
    plt.show()


def adversary_example(image_path, cls_target,
                      noise_limit, required_score):
    """
    查找并绘制给定图像的对抗式噪声。

    image_path: 输入的图像路径(必须是 *.jpg 文件).
    cls_target: 目标类数(1-1000之间的整数)。
    noise_limit: 噪声中像素值的限制。
    required_score: 当目标类分数达到这个值时停止。
    """

    # 找到对抗的噪音。
    image, noisy_image, noise, \
    name_source, name_target, \
    score_source, score_source_org, score_target = \
        find_adversary_noise(image_path=image_path,
                             cls_target=cls_target,
                             noise_limit=noise_limit,
                             required_score=required_score)

    # 绘制图像和噪声。
    plot_images(image=image, noise=noise, noisy_image=noisy_image,
                name_source=name_source, name_target=name_target,
                score_source=score_source,
                score_source_org=score_source_org,
                score_target=score_target)

    # 打印一些噪音的统计数字。
    msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
    print(msg.format(noise.min(), noise.max(),
                     noise.mean(), noise.std()))

image_path = "images/GrannySmithto140_f.png"

adversary_example(image_path=image_path,
                  cls_target=141,
                  noise_limit=3.0,
                  required_score=0.99)




