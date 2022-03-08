### build target

假设输入的target size: 64，即batch size为64，而64张图中所含目标数目不同，假设第一张图片有四个物体，那么`target[0]`的size为（4，5），也就是四个物体，每一行含有xywh cls五个信息（范围是0到1，是相对于图片size的比例，乘以任意特征图尺寸边长就能得到在该特征图尺寸上的图像xywh）

1、处理输入数据

算出真实框相对于13x13这个维度map的位置，对xy向下取整得到所属grid cell的整数坐标

~~~python
x = target[0][:,0:1] * 13
y = target[0][:,1:2] * 13
w = target[0][:,2:3] * 13
h = target[0][:,3:4] * 13

gx = torch.floor(x)
gy = torch.floor(x)
~~~

2、计算IOU

假设所有真实框和anchor（9个）的左上角都在原点，算他们的IOU，拿到与这4个目标IOU最大的anchor，假设他们对应关系如下

~~~
物体1			物体2			物体3			物体4
anchor0		 anchor3	   anchor2		anchor1
~~~

3、获得正样本和Mask

遍历这四个物体，判断他们对应的anchor是不是属于这一个特征图对应的三个anchor

例如：对于13x13的特征图，对应的anchor是0 1 2，而物体2对应anchor3，说明物体2应该由其他维度的特征图负责预测

如果对应的anchor属于此特征图：

4、获得调整量

~~~python
tx = x - gx;
ty = y - gy;
~~~

$$
tw = log(\frac{w}{anchor-width}) \\
th = log(\frac{h}{anchor-height})
$$

同时得到正、负样本Mask：

~~~python
#----------------------------------------#
#   noobj_mask代表无目标的特征点
#----------------------------------------#
noobj_mask[b, best_n, gx, gy] = 0
#----------------------------------------#
#   mask代表有目标的特征点
#----------------------------------------#
mask[b, best_n, gx, gy] = 1
~~~

### get_ignore

经过get target我们初步得到了副样本mask：noobjmask，在此基础上，再次筛选负样本：通过网络的输出生成507个预测框，将这507个预测框中与target生成的真实框IOU大于某个阈值的框从负样本中剔除，因为他们如果作为负样本的话loss会特别大

1、根据prediction生成预测框

注意网络输出xywh的范围是0-1，其中xy表示相对于其gridcell右上角的偏移量，wh需要经过exp再去修正anchor

~~~python
pred_boxes[..., 0] = x.data + grid_x
pred_boxes[..., 1] = y.data + grid_y
pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
~~~

2、根据输入target生成实际框

~~~python
gx = target[i][:, 0:1] * in_w
gy = target[i][:, 1:2] * in_h
gw = target[i][:, 2:3] * in_w
gh = target[i][:, 3:4] * in_h
~~~

3、比较IOU:

size:507x4 代表了507个预测框和真实框的IOU，选取其中IOU大于threshold的设置为忽略（即从负样本mask中剔除）

~~~python
noobj_mask[i][anch_ious_max>self.ignore_threshold] = 0
~~~

