# 安装过程出现的问题

## 报错

1. tensorflow的GPU环境搭建粗略步骤<https://www.tensorflow.org/install/gpu>
    1. 先安装nvidia驱动
    2. 再安装cuda
    3. 再安装cudnn
    4. 使用nvidia中的demo测试，然后再使用tensorflow测试。
2. 当前使用的系统版本
|编号|名称|版本|说明|
|---|---|---|---|
|1|windows server|2016|nvidia要求的window版本最新的只能是这个。|
|2|GPU驱动|||
|3|CUDA|11.5|下载地址<>。**11.6版本的CUDA没有对应版本的cuDNN**。|
|4|cuDNN|cudnn-windows-x86_64-8.3.2.44_cuda11.5-archive|下载地址<https://developer.nvidia.com/rdp/cudnn-download>。注意对应x86_64的两个windows文件都需要下载，一个是Local Installer for Windows (Exe)和Local Installer for Windows (Zip)，前面是安装cuDNN，后面的zip是用于补全部分dll文件的。|
|5||||

3. 使用框架时报错

    ```Python
    import tensorflow as tf

    tf.test.is_gpu_available()

    import tensorflow as tf
    print("TF version:", tf.__version__)

    # 检测Tensorflow是否支持GPU
    print("GPU is ", "available" if tf.test.is_gpu_available() else "NOT available.")
    ```

    代码的时候报以下错误

    ```Script
    PS D:\codeSpace\test> & D:/Python310/python.exe d:/codeSpace/test/test.py
    WARNING:tensorflow:From d:\codeSpace\test\test.py:3: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use `tf.config.list_physical_devices('GPU')` instead.
    2022-02-09 15:21:51.500071: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-02-09 15:21:52.540180: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found
    2022-02-09 15:21:52.540269: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries
    mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup
    the required libraries for your platform.
    Skipping registering GPU devices...
    TF version: 2.8.0
    2022-02-09 15:21:52.550397: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries
    mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup
    the required libraries for your platform.
    Skipping registering GPU devices...
    GPU is  NOT available.
    ```

    解决方法：<https://blog.csdn.net/wilde123/article/details/116903346>
    解决步骤：
        1. 在<https://developer.nvidia.com/rdp/cudnn-download>连接中下载对应的cudnn-windows-x86_64-8.3.2.44_cuda11.5-archive.zip文件。
        2. 将其解压，然后将其中bin目录中的cudnn64_8.dll文件拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin目录下即可。
    原因就是在C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin目录下缺失cudnn64_8.dll文件。

4. 推荐使用的查看GPU状态代码<http://t.zoukankan.com/Renyi-Fan-p-13461855.html>

```Python
gpus = tf.config.list_physical_devices(device_type='GPU')
cpus = tf.config.list_physical_devices(device_type='CPU')
print(gpus, cpus)
```
