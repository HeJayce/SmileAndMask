## 笑脸识别与口罩识别

### 1.安装tensorflow和keras

本次的实验是在已经安装好Anaconda的情况下进行的，使用anaconda可以减小很多库的安装，并且可以使用Jupyter Notebook

##### 在Anaconda Prompt下输入命令

安装tensorflow：

``` powershell
pip install tensorflow==1.4 -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

安装keras

```powershell
pip install keras==2.0.8 -i  https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 

### **2.HOG（方向梯度直方图）**

**HOG**是应用在[计算机视觉](https://zh.wikipedia.org/wiki/计算机视觉)和[图像处理](https://zh.wikipedia.org/wiki/图像处理)领域，用于[目标检测](https://zh.wikipedia.org/w/index.php?title=目标检测&action=edit&redlink=1)的特征描述器。这项技术是用来计算局部图像梯度的方向信息的统计值。这种方法跟[边缘方向直方图](https://zh.wikipedia.org/w/index.php?title=边缘方向直方图&action=edit&redlink=1)、[尺度不变特征变换](https://zh.wikipedia.org/wiki/尺度不变特征变换)以及[形状上下文方法](https://zh.wikipedia.org/w/index.php?title=形状上下文方法&action=edit&redlink=1)有很多相似之处，但与它们的不同点是：HOG描述器是在一个网格密集的大小统一的细胞单元上计算，而且为了提高性能，还采用了重叠的局部对比度归一化技术。

HOG描述器最重要的思想是：在一副图像中，局部目标的表象和形状能够被梯度或边缘的方向密度分布很好地描述。具体的实现方法是：首先将图像分成小的连通区域，我们把它叫细胞单元。然后采集细胞单元中各像素点的梯度的或边缘的方向直方图。最后把这些直方图组合起来就可以构成特征描述器。为了提高性能，我们还可以把这些局部直方图在图像的更大的范围内进行对比度归一化，所采用的方法是：先计算各直方图在这个区间中的密度，然后根据这个密度对区间中的各个细胞单元做归一化。通过这个归一化后，能对光照变化和阴影获得更好的效果。

![image-20200707214109007](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707214109007.png)



### 3.卷积神经网络
卷积神经网络(CNN）是一类包含卷积计算且具有深度结构的前馈神经网络（Feedforward Neural Networks），是深度学习（deep learning）的代表算法之一 。卷积神经网络具有表征学习（representation learning）能力，能够按其阶层结构对输入信息进行平移不变分类（shift-invariant classification），因此也被称为“平移不变人工神经网络（Shift-Invariant Artificial Neural Networks, SIANN）”

卷积神经网络长期以来是图像识别领域的核心算法之一，并在学习数据充足时有稳定的表现 。对于一般的大规模图像分类问题，卷积神经网络可用于构建阶层分类器（hierarchical classifier） ，也可以在精细分类识别（fine-grained recognition）中用于提取图像的判别特征以供其它分类器进行学习 。对于后者，特征提取可以人为地将图像的不同部分分别输入卷积神经网络 ，也可以由卷积神经网络通过非监督学习自行提取。



### 4.使用GENKI4K数据集进行数据的训练

1）.在Jupyter Notebook文件的所在目录下（一般是USER下）创建一个文件夹，其子目录为test，train，validation



![image-20200707214507632](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707214507632.png)

在上3个文件夹下分别创建smile 和 unsmile 文件夹

2）将数据集的图片按照文件夹的名称分类，数量如下：

![image-20200707214754845](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707214754845.png)

3）依靠之前的猫狗数据集训练方法，构建卷积网络：

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

4）查看每一个图层

```python
model.summary()
```

![image-20200707215101879](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707215101879.png)

5）对图片进行归一化处理

```python
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

```python
from keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen=ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 目标文件目录
        train_dir,
        #所有图片的size必须是150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir,
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary'
```

![image-20200707215252264](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707215252264.png)

6)输出图像形状：

```python
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch)
    break
```

![image-20200707215354857](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707215354857.png)

7）训练模型并保存

```python
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
```

![image-20200707215453875](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707215453875.png)

```python
model.save('genki4k/smile1.h5')
```

8）画出模型训练的loss和accuracy

```python
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
```

![image-20200707215657595](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707215657595.png)

### 5.数据增强

```python
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
```

```python
from keras.preprocessing import image
fnames = [os.path.join(train_smile_dir, fname) for fname in os.listdir(train_smile_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
```

![image-20200707215850352](C:\Users\jayce\AppData\Roaming\Typora\typora-user-images\image-20200707215850352.png)

继续构建卷积网络比进行归一化处理

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

```python
#归一化处理
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=60,  
      validation_data=validation_generator,
      validation_steps=50)
```





