# Diary of a Frustrated Alchemist

## Quick Update

* 03/19 02:33 AM, faild at ```run_webqa.py``` line 577. Missing confusing img.pkl files.
* 03/27 05:45 PM, the author instructed to use WebQA_x101. Having no idea how it is managed.
* 03/27 09:41 PM, successfully reproduced the WebQA load image demo. Thankfully, this demo works smoothly.

## What is missing

* missing python libraries

1. missing tqdm 

You need to install tqdm from conda-forge
```
conda install -c conda-forge tqdm
```

2. missing visdom
```
conda install -c conda-forge visdom
```

## Preparation Work

The author did a super bad job on managing the code for publication. We would like to use the following settings to clearly describe how to try to run the code. The guideline of our attempts are as follows:
1. Replace any confusing, distracting absolute path to a relatively path.
2. Separate out the path to dataset, path to checkpoint, and path to model weight from external sources
3. Add comments!!!!!!!!!!

Our project hierarchy.

```
├── 11777_project
├── apex
├── dataset
└── detectron_weights
```

As can be seen, 
* we cloned the ```apex``` to the path that is in parallel to the ```11777_project```. 
* The ```dataset``` is supposed to store the checkpoint and light weighted dataset (i.e., the index file). Since my working machine could not save the entire *WebQA* dataset (the 71GB TSV file), I mounted an external disk to save that.
* ```detectron_weights``` is a path that store the layer-7 weight of the work in *detectron*, which can be downloaded from ```https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz```. Of course, you need to untar it because the code will load the weight and biase pkl files.

We also should the file hierarchy in the ```11777_project``` directory, just to let you know if you are also in the same painful way as us to simply run the ```WebQA_Baseline```.

```
11777_project/
└── baseline
    ├── WebQA_Baseline
    └── WebQA_x101fpn
```

Since we are expecting a super large workload on renovating the author's source code, we simply copied the majority of the source code from the author's provided github repositories and prepared to refactorize it. 

4. The ```imgs.lineidx``` must be saved in the same directory as the extracted ```imgs.tsv```, at least the author assumed you do so.

## Painful walkthrough of the path issue fixing 
1. (WebQA_Baseline) The author uses "VLP" rather than "WebQA_Baseline" as the working directory. Therefore, in the README.md, "cd VLP" actually means "cd WebQA_Baseline".
2. (WebQA_Baseline) There are super many places that the author uses an absolute path to specify a folder/file. You can search all of them out using keyword ```yingshac```. You need to check the following files
```vlp/run_webqa.py```, ```pytorch_pretrained_bert/modeling.py```

You need to download the fc_7's weight from ```https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/detectron_weights.tar.gz``` and unzip it to a folder you know. For me, I unzipped it to the ```WebQA_Baseline``` folder and make the following code correction

```
# before correction
self.vis_embed[0].weight.data.copy_(torch.from_numpy(pickle.load(
        open('/home/yingshac/CYS/WebQnA/cpts/detectron_weights/fc7_w.pkl', 'rb'))))
self.vis_embed[0].bias.data.copy_(torch.from_numpy(pickle.load(
        open('/home/yingshac/CYS/WebQnA/cpts/detectron_weights/fc7_b.pkl', 'rb'))))

# after correction
self.vis_embed[0].weight.data.copy_(torch.from_numpy(pickle.load(
        open('../detectron_weights/fc7_w.pkl', 'rb'))))
self.vis_embed[0].bias.data.copy_(torch.from_numpy(pickle.load(
        open('../detectron_weights/fc7_b.pkl', 'rb'))))
```


Here, we would like to clarify several paths:

* ```gold_feature_folder```, ```distractor_feature_folder```, ```x_distractor_feature_folder``` are paths that **WILL** be used to store the gold features from the training set.
* ```txt_dataset_json_path``` actually means the ```WebQA_train_val.json```. 
* ```model_recover_path``` needs to specify to ```None``` for the first run.


3. (WebQA_x101fpn) We would like to run the ```featureExtraction.py```. As expected, super many confusing absolute paths. 
* The ```config="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"``` directed to a strange path and it seems that the ```config``` variable is not using. I don't know whether we can ignore the file at present.
* File ```faster_rcnn_X_101_32x8d_FPN_3x.yaml``` and  ```e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp-427.pkl``` can be found in ```11777_project/baseline/WebQA_x101fpn/detectron-vlp```. So, we can simply change the path so that the code reads

```
    ## the original code
    # cfg.merge_from_file("/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    # cfg.MODEL.WEIGHTS = "/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp-427.pkl"

    cfg.merge_from_file("detectron-vlp/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron-vlp/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp-427.pkl"
```

4. (WebQA_Baseline) You may see the following error 
```
vis_pe, scores, img, cls_label = self.img_data_tsv[image_id//10000000][image_id % 10000000]

KeyError: 3
```
The reason is that the author thinks (maybe for some preliminary versions) images for gold/distractor/x-distractors are from different tsv files. However, WebQA only provides one TSV file (maybe all three sources are concatenated into the same file source). Since we assign the same tsv file path to all the three file sources, we force the img_data_tsv to load image from the gold source, which brings the following modification
```
vis_pe, scores, img, cls_label = self.img_data_tsv[0][image_id % 10000000]
```

## Where is the dataset?

I followed the provided ```download_imgs.sh``` to download the dataset, but I think the website has a timeout strategy to prevent an overlong connection. At my machine, the downloading process stopped after downloading around 30 chunks. So I choose to save shared Google Drive folder and let Google Drive to synchronize the dataset to my local folder. If you have a similar problem, you can try to use Google Drive as I did.

Since the dataset is too large, I store it on a windows sector, which can be mound in path

```
/media/UoneWorkspace/MMML_dataset/dataset/WebQA/WebQA_imgs_7z_chunks/
```

I execute the following code to unzip the compressed dataset

```
7z x imgs.7z.001
```

You are supposed to see a very big file named *imgs.tsv* in the specified folder above

The file is 
```
/media/UoneWorkspace/MMML_dataset/dataset/WebQA/WebQA_imgs_7z_chunks/imgs.tsv
```

## How to activate conda

Conda environment needs to be initialized each time you start a terminal

```
source conda_init.sh
```

Then, you need to activate the vlp environment 

```
conda activate vlp
```
## First Step Retrieval training

### run_webqa.py
```
python run_webqa.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 128 --save_loss_curve --output_dir data/output/filter_debug --ckpts_dir ./data/ckpts/filter_debug --use_x_distractors --do_train --num_train_epochs 1
```

### run_webqa_vinval.py
```
python run_webqa_vinvl.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 128 --save_loss_curve --output_dir data/output/filter_debug --ckpts_dir ./data/ckpts/filter_debug --use_x_distractors --do_train --num_train_epochs 1
```

