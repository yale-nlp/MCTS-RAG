CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python run_src/do_generate.py \
    --dataset_name FMT \
    --test_json_filename test_all_with_evidence \
    --api gpt-4o \
    --model_ckpt meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
    --note default \
    --half_precision \
    --num_rollouts 16 \
    --tensor_parallel_size 1 \
    --temperature 0.1 \
    --start_idx 3\
    --verbose  