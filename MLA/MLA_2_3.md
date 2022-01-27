# yolo v3 windows 安装步骤

## 1. 准备

[参考步骤](https://zhuanlan.zhihu.com/p/45845454)

1. 下载yolo
2. 下载C/C++编译环境，这里使用的是mingw x64
   1. <https://sourceforge.net/projects/mingw-w64/files/mingw-w64/mingw-w64-release/>
   2. [步骤实验参考](https://blog.csdn.net/pdcxs007/article/details/8582559)
3. 已经训练好的权重下载
   1. <https://pjreddie.com/media/files/yolov3.weights>
4. 训练数据集
   1. <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar>
   2. <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar>
   3. <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar>
5. [opencv的官方页面](https://opencv.org/releases/)
   1. [opencv下载](https://udomain.dl.sourceforge.net/project/opencvlibrary/4.5.4/opencv-4.5.4-vc14_vc15.exe)

## 2. 报错

1. 报错:找不到cudnn.h。在cudn的目录下搜索这个cudnn.h文件(服务器上找到的位置：C:\tools\cuda\cuda\include)，然后将它拷贝到opencv的include目录下(拷贝到的位置D:\codeField\darknet-master\include)。
   1. asdfaf
   2. 111
2. 提示缺少cudnn.lib。将C:\tools\cuda\cuda\lib\x64\cudnn.lib拷贝到D:\codeField\darknet-master\3rdparty\pthreads\lib目录下，问题得以解决。D:\codeField\darknet-master\3rdparty\pthreads\lib这个目录下是opencv中唯一放置lib文件的地方。
3. 在编译完成之后，使用darknet.exe的时候会提示“无法启动此程序，因为计算机中丢失opencv_world454.dll....”。[解决方法](https://blog.csdn.net/op_chaos/article/details/114023937)。将D:\Program Files\opencv\build\x64\vc15\bin目录下的所有.lib文件拷贝到C:\Windows\System32目录下，问题得以解决。
4. 运行报错：CUDA status Error: file: D:\codeField\darknet-master\src\dark_cuda.c : cuda_set_device() : line: 38 : build time: Dec  4 2021 - 17:08:20
   1. 尝试使用C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\extras\demo_suite中的.\bandwidthTest.exe
   2. 运行C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\demo_suite目录下的deviceQuery.exe总是提示初始化失败。而且cuda 11.1也重装了。
   3. 估计是cuda的版本和nvidia驱动的版本不一致造成的。直接结果是使用nvidia-smi命令中显示的cuda版本和使用nvcc -V显示的cuda版本不一致。可能需要重新安装nvidia的驱动，另外驱动大师显示该服务器的的显卡是GRID的。很奇怪。
   4. **要求将bandwidthTest.exe和deviceQuery.exe命令运行成功**，这样成功了才认为GPU可以使用了。然后再使用tensorflow中的函数来调用GPU，能够识别那么就认为tensorflow可以使用GPU了。[参考](https://blog.csdn.net/xiangxiang613/article/details/112603083)
      1. 对于这个问题按照CUDA的说明，有可能是操作系统和CUDA不匹配造成的。
