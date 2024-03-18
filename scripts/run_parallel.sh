torchrun --nproc-per-node 3 /cpfs01/user/shanyirong/Superfiltering/code_ifd/data_analysis_parallel.py \
    --data_path /cpfs01/user/shanyirong/Superfiltering/data/share_data/subjective5w_pool_with_score/retrieval_refined_bench/origin/v1_retrieval_creation_v0_filtered.jsonl\
    --save_path /cpfs01/user/shanyirong/Superfiltering/data/share_data/subjective5w_pool_with_ifdscore/retrieval_refined_bench/origin/v1_retrieval_creation_v0_filtered.jsonl \
    --model_name_or_path /cpfs01/shared/public/public_hdd/yehaochen/deploy/7B/aliyun_Ampere_7B_FT_0_19_commit_19c178f_subset_baseline_hf/ \
    --max_length 32768