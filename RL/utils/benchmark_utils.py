import torch
import os
from torch.distributed.tensor import DTensor

# Merge the per-rank checkpoint to a pytorch_model.bin (FSDP/ZERO-3 trained):
# /path/to/ckpt might have the following dir structure:
#     --- config.json
#     --- model_world_size_8_rank_0.pt 
#     --- model_world_size_8_rank_1.pt 
#     ... ...
#     --- model_world_size_8_rank_7.pt 
#     --- tokenizer.json
#     --- tokenizer_config.json
ckpt_dir = "/path/to/ckpt"
world_size = 8

# Step 1: Load all state_dicts
all_state_dicts = []
for rank in range(world_size):
    shard_path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    print(f"Loading {shard_path}")
    state_dict = torch.load(shard_path, map_location="cpu")
    all_state_dicts.append(state_dict)

# Step 2: Build key -> list[local_tensor] mapping
full_state_dict = {}
for key in all_state_dicts[0].keys():
    shards = [sd[key] for sd in all_state_dicts]
    if isinstance(shards[0], DTensor):
        local_tensors = [shard.to_local() for shard in shards]
        # Assume Shard(0): concat along dim 0
        full_tensor = torch.cat(local_tensors, dim=0)
        full_state_dict[key] = full_tensor
        print(f"[DTensor] {key}: {full_tensor.shape}")
    else:
        # Replicated parameter
        full_state_dict[key] = shards[0]
        print(f"[Replicated] {key}: {shards[0].shape}")

# Step 3: Save as Huggingface compatible model format
torch.save(full_state_dict, "pytorch_model.bin")
print("Saved full model to: pytorch_model.bin")


from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm


# Verify the merged model is good
# Assuming the model is compatible with Qwen2-7B architecture and let's verify it on gsm8k
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
model.load_state_dict(torch.load("pytorch_model.bin"))
model= model.to("cuda")
model.eval() # set to evaluation mode
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
tokenizer.pad_token = tokenizer.eos_token # set padding token

# Step 2: Load GSM8K test dataset
dataset = load_dataset("gsm8k", "main")
test_dataset = dataset["test"]

# Step 3: Function to extract the final answer
def extract_final_answer(text):
    match = re.search(r"So the answer is (.+)", text)
    if match:
        return match.group(1).strip()
    else:
        return None

# Step 4: Prepare prompts for each question
prompts = [f"Question: {example['question']}\nAnswer:" for example in test_dataset]

# Step 5: Set batch size for efficient processing
batch_size = 16

# Step 6: Initialize lists from answers
model_answers = []
ground_truth_answers = [extract_final_answer(example["answer"]) for example in test_dataset]

# Step 7: Process prompts in batches and generate responses
for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompts = prompts[i:i + batch_size]
    # Tokenize the batch
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
    # Generate responses without gradient computation
    with torch.no_grad():
        outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False
        )
    # Decode generated outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Extract answers from generated texts
    batch_model_answers = [extract_final_answer(text) for text in generated_texts]
    model_answers.extend(batch_model_answers)

# Step 8: calculate accuracy
correct = sum(1 for m, g in zip(model_answers, ground_truth_answers) if m == g)
accuracy = correct / len(test_dataset)
print(f"Accuracy: {accuracy: .4f}")
