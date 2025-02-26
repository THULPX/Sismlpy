import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
import numpy as np

# ----------------------------- Transfer Learning -----------------------------

# 介绍：
# 迁移学习（Transfer Learning）是一种将已经在一个任务上训练得到的知识迁移到另一个相关任务的机器学习方法。尤其是在数据不足的情况下，迁移学习能够显著提高模型的性能。通过使用在大数据集（如ImageNet）上预训练的模型，可以加速训练过程并提高模型在特定任务上的效果。

# 输入输出：
# 输入：
# - base_model: 预训练模型。
# - new_model: 新的任务模型，通常是在预训练模型基础上进行微调。
# 输出：
# - fine-tuned model: 微调后的模型。

# 算法步骤：
# 1. 选择一个合适的预训练模型（如VGG16、ResNet等），并冻结其卷积层。
# 2. 在新的任务上训练模型，仅更新新加的全连接层（Dense Layer）。
# 3. 如果需要，可以解冻部分卷积层并进一步训练，以微调模型。

# 主要参数：
# - base_model: 预训练模型。
# - fine_tune_layers: 选择哪些层进行微调。

# ----------------------------- 示例：使用VGG16进行迁移学习 -----------------------------

# 加载VGG16预训练模型（不包括顶层分类器）
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 冻结VGG16的卷积层
base_model.trainable = False

# 构建新的模型
model = models.Sequential([
    base_model,  # 加入VGG16预训练模型
    layers.Flatten(),  # 展平层
    layers.Dense(256, activation='relu'),  # 新的全连接层
    layers.Dense(10, activation='softmax')  # 输出层，假设是10类分类问题
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 假设我们有训练数据集train_images和train_labels
# 这里创建一些虚拟数据，实际情况中应替换为真实数据
train_images = np.random.rand(100, 224, 224, 3)  # 100张224x224的RGB图像
train_labels = np.random.randint(0, 10, size=(100, 1))  # 100个标签，假设有10个类别

# 将标签转换为one-hot编码
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# ----------------------------- 微调预训练模型 -----------------------------

# 解冻部分卷积层进行微调
base_model.trainable = True
fine_tune_layers = 15  # 假设解冻最后15层

# 冻结前面的卷积层
for layer in base_model.layers[:-fine_tune_layers]:
    layer.trainable = False

# 重新编译模型，以适应微调
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 继续训练（微调）
model.fit(train_images, train_labels, epochs=5, batch_size=32)
