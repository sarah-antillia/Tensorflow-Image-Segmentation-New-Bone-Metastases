<h2>Tensorflow-Image-Segmentation-New-Bone-Metastases (Updated: 2025/05/12)</h2>

This is the first experiment of Image Segmentation for New-Bone-Metastases 
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417423008783">
BM-Seg: A new bone metastases segmentation dataset and ensemble of CNN-based segmentation approach</a>
<br>
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/drive/folders/1T4ldxMgGlLT3ji2yEqaPNq2Ppf4sjiuz">
BM-Seg.zip</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (26).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (26).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (26).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (42).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (42).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (42).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (258).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (258).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (258).jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this New-Bone-Metastases Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
We used the following publicly available BM-Seg dataset:<br>
<a href="https://drive.google.com/drive/folders/1T4ldxMgGlLT3ji2yEqaPNq2Ppf4sjiuz">
BM-Seg.zip</a>
<br>
<b>BM-Seg: A new bone metastases segmentation dataset and ensemble of CNN-based segmentation approach
</b><br>
Marwa Afnouch, Olfa Gaddour, Yosr Hentati, Fares Bougourzi, Mohamed Abid, Ihsen Alouani, Abdelmalik Taleb Ahmed
<br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417423008783">
https://www.sciencedirect.com/science/article/abs/pii/S0957417423008783</a>
<br>
<br>
<h3>
<a id="2">
2 New-Bone-Metastases ImageMask Dataset
</a>
</h3>
 If you would like to train this New-Bone-Metastases Segmentation model by yourself,
 please download the dataset from the google drive 
 <a href="https://drive.google.com/drive/folders/1T4ldxMgGlLT3ji2yEqaPNq2Ppf4sjiuz">
BM-Seg.zip</a>, expand the downloaded and put it under <b>./generator</b> folder and 
run <a href="./generator/split_master.py">split_master</a>
<br>
<pre>
./dataset
└─New-Bone-Metastases
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>New-Bone-Metastases Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/New-Bone-Metastases_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large, but enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained New-Bone-MetastasesTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at start (epoch 1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at end (epoch 98,99,100)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was terminated at epoch 100.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/train_console_output_at_epoch_100.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for New-Bone-Metastases.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/evaluate_console_output_at_epoch_100.png" width="720" height="auto">
<br><br>Image-Segmentation-New-Bone-Metastases

<a href="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this New-Bone-Metastases/test was not so low, and dice_coef not so high as shown below.
<br>
<pre>
loss,0.1223
dice_coef,0.7834
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for New-Bone-Metastases.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (26).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (26).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (26).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (42).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (42).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (42).jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (220).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (220).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (220).jpg" width="320" height="auto"></td>
</tr


>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (258).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (258).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (258).jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (291).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (291).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (291).jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/images/1 (297).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test/masks/1 (297).jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/New-Bone-Metastases/mini_test_output/1 (297).jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. BM-Seg: A new bone metastases segmentation dataset and ensemble of CNN-based segmentation approach
</b><br>
Marwa Afnouch, Olfa Gaddour, Yosr Hentati, Fares Bougourzi, Mohamed Abid, Ihsen Alouani, Abdelmalik Taleb Ahmed
<br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417423008783">
https://www.sciencedirect.com/science/article/abs/pii/S0957417423008783
</a>
<br>
<br>
<b>2. EH-AttUnetplus : Ensemble of trained Hybrid-AttUnet++ models (EH-AttUnet++)
</b><br>
Marwa-Afnouch<br>
<a href="https://github.com/Marwa-Afnouch/EH-AttUnetplus?tab=readme-ov-file">
https://github.com/Marwa-Afnouch/EH-AttUnetplus?tab=readme-ov-file
</a>
<br>
<br>
