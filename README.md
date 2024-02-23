# yolov9.infer

> NOTE: The code here is just for reference, I have no time to norm it.

It' same that there still no right way to export yolov8 to onnx format, so I try to export it.

1. clone yolov9 from [github](https://github.com/WongKinYiu/yolov9)
2. copy the `export.py` to the `yolov9/weights` folder
3. download the `gelan-c.pt` 
4. run `python export.py`
5. cp `yolov9/weights/gelan-c.onnx` to this folder 
6. run `python main.py`

If everything is ok, you will see the result like this:

![](./assets/samples.png)



## Reference
- https://github.com/WongKinYiu/yolov9