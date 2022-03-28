"""
CREATED BY 11777 victims to figure out whether the dataloader is working.

We are pushed to crazy about why the code can provide error everywhere. 
This script is designed to test whether the dataset that author's code is designed for matches the provided dataset.

"""

import sys
sys.path.append("../../WebQA_Baseline")


from pytorch_pretrained_bert.tokenization import BertTokenizer
import vlp.webqa_VinVL_loader as webqa_VinVL_loader

"""
Load Bert Tokenizer
"""
tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, 
        do_lower_case=args.do_lower_case,
        cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank))
if args.max_position_embeddings:
    tokenizer.max_len = args.max_position_embeddings

ImgDataTsv_dict = {}
if args.gold_img_tsv is not None: ImgDataTsv_dict[0] = args.gold_img_tsv
if args.neg_img_tsv is not None: ImgDataTsv_dict[1] = args.neg_img_tsv
if args.x_neg_img_tsv is not None: ImgDataTsv_dict[2] = args.x_neg_img_tsv
processor = webqa_VinVL_loader.Preprocess4webqa_VinVL(\
    args.max_pred, 
    args.mask_prob,
    list(tokenizer.vocab.keys()), 
    tokenizer.convert_tokens_to_ids, 
    seed=args.seed, 
    max_len=args.max_seq_length,
    len_vis_input=args.len_vis_input, 
    max_len_a=args.max_len_a, 
    max_len_b=args.max_len_b,
    max_len_img_cxt=args.max_len_img_cxt, 
    new_segment_ids=args.new_segment_ids,
    truncate_config={'trunc_seg': args.trunc_seg, 'always_truncate_tail': args.always_truncate_tail}, 
    use_img_meta=args.use_img_meta, 
    use_img_content=args.use_img_content, 
    use_txt_fact=args.use_txt_fact, 
    ImgDataTsv_dict = ImgDataTsv_dict)

train_dataset = webqa_VinVL_loader.webqaDataset_filter_with_both(\
    dataset_json_path=args.txt_dataset_json_path, # path to the QA json file
    split=args.split, # whether to split the dataset to train/test/val
    Qcate=args.Qcate, # list of types of question categories, e.g. Yes/No, Shape, Number
    batch_size=args.train_batch_size, # batch size
    tokenizer=tokenizer, # bert tokenizer
    use_num_samples=args.use_num_samples, # TODO: not sure
    processor=processor, # TODO: webqa_VinVL_loader.Preprocess4webqa_VinVL, feature extractor?
    answer_provided_by='txt',
    max_snippets=args.txt_filter_max_choices,
    max_imgs=args.img_filter_max_choices,
    device=device)