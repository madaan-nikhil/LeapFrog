# Diary of a Frustrated Alchemist

## Quick Update

* 03/19 02:33 AM, faild at ```run_webqa.py``` line 577. Missing confusing img.pkl files.
* 03/27 05:45 PM, the author instructed to use WebQA_x101. Having no idea how it is managed.

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

## Where is the dataset?

Since the dataset is too large, I store it on a windows sector, which can be fmound in path

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
```
python run_webqa.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 128 --save_loss_curve --output_dir data/output/filter_debug --ckpts_dir ./data/ckpts/filter_debug --use_x_distractors --do_train --num_train_epochs 1
```

