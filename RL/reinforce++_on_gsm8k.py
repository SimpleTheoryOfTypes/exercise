import numpy as np
import torch
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

# Configuration
# MODEL_NAME = "EleutherAI/gpt-neo-125M"  # Use a small, readily available model
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Use a small, readily available model
DATASET_NAME = "gsm8k"
TASK_NAME = "main"
SPLIT = "train"  # Or "test"
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
GAMMA = 1.0  # Discount factor
NUM_EPOCHS = 3
CLIP_EPSILON = 0.2
SEED = 42
MAX_LENGTH = 512
TRAIN_PERCENT = 0.1  # 10% of the training data

# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Determine the device to use
device = torch.device("cpu")  # Use CPU

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if it doesn't exist

# Load dataset
dataset = load_dataset(DATASET_NAME, TASK_NAME, split=SPLIT)


# Preprocess data and tokenize questions
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]  # Get the question
    inputs = tokenizer(questions, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()  # Important for the model
    return inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("torch")

# Take a subset of the training data
train_size = int(len(tokenized_dataset) * TRAIN_PERCENT)
tokenized_dataset = tokenized_dataset.select(range(train_size))
print(f"Training on {len(tokenized_dataset)} examples")

dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Define the reward function (Function-level reward for GSM8k)
def get_reward(model_output, target_string):
    """
    Calculates the reward for a generated answer on the GSM8K dataset.
    This function aims to provide a more granular reward by checking for the
    presence of the correct numerical answer within the model's output.
    """
    # Extract the predicted answer from the model output.  This is the key part
    # where we try to isolate the numeric answer.  This is a simplified approach
    # and might need more refinement for real-world scenarios.
    try:
        # 1. Attempt to find the last number in the generated text
        numbers = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", model_output)]  # Find all numbers
        if numbers:
            predicted_answer = numbers[-1]
        else:
            predicted_answer = None
        # predicted_answer = float(re.search(r"[-+]?\d*\.\d+|\d+", model_output).group(0))
    except:
        predicted_answer = None  # Assign None if no number is found

    # Extract the correct answer from the target string.
    try:
        correct_answer = float(re.search(r"[-+]?\d*\.\d+|\d+", target_string).group(0))
    except:
        return 0.0  # Return 0 reward if correct answer not found

    # Compare predicted and correct answers
    if predicted_answer is not None and np.isclose(predicted_answer, correct_answer, rtol=1e-02):
        return 1.0  # High reward for correct answer
    else:
        return 0.0  # Zero reward for incorrect or no answer


# Function to plot the training loss curve
def plot_loss_curve(losses, epochs):
    """
    Plots the training loss curve.

    Args:
        losses (list): A list of average training losses for each epoch.
        epochs (int): Number of epochs
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, marker="o", linestyle="-")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.show()


def print_sample_output(model_input, model_output, target_text, reward):
    """Prints a sample model output, target text, and reward."""
    # Define color codes for different sections of the output
    color_input = "\033[34m"  # Blue
    color_output = "\033[32m"  # Green
    color_target = "\033[33m"  # Yellow
    color_reset = "\033[0m"  # Reset to default color

    print("-" * 80)
    print(f"{color_input}Model Input:\n{model_input}{color_reset}\n")
    print(f"{color_output}Model Output:\n{model_output}{color_reset}\n")
    print(f"{color_target}Target Text:\n{target_text}{color_reset}\n")
    print(f"Reward: {reward:.4f}")
    print("-" * 80)

def print_gsm8k_sample(sample):
    """Prints all fields of a sample from the GSM8k dataset."""
    print("-" * 80)
    print("GSM8k Sample Data:")
    for key, value in sample.items():
        print(f"{key}: {value}")
    print("-" * 80)


# Main training loop
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
all_epoch_losses = []  # To store average loss per epoch

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    all_rewards = []  # Accumulate rewards for baseline calculation
    epoch_losses = []

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        input_ids = batch["input_ids"].to(device)  # Use CPU
        attention_mask = batch["attention_mask"].to(device)  # Use CPU
        labels = batch["labels"].to(device)  # unused. # Use CPU

        # 1. Generate model output ( নীতি)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,  # Use max_new_tokens instead of max_length
            num_return_sequences=1,  # Generate only one sequence per input
            do_sample=True,  # Sample
            top_k=0,
            temperature=0.7,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # 2. Calculate Rewards (r(x,y))
        #target_texts = []
        #for i in range(len(input_ids)):
        #    #  i is the batch index (0, 1, ..., BATCH_SIZE-1)
        #    original_dataset_index = batch["input_ids"][i][0].item()
        #    try:
        #        target_texts.append(dataset[original_dataset_index]["answer"])
        #    except IndexError:
        #        import pdb; pdb.set_trace()

        batch_rewards = []
        target_texts = batch["answer"]
        for i in range(len(generated_texts)):
            reward = get_reward(generated_texts[i], target_texts[i])  # Use the reward function
            batch_rewards.append(reward)
        batch_rewards = np.array(batch_rewards, dtype=np.float32)
        all_rewards.extend(batch_rewards)  # Accumulate for global baseline

        # 3. Calculate Loss and update model
        optimizer.zero_grad()

        # Get log probabilities.  Important to use the generated outputs
        log_probs = []
        for i in range(len(outputs)):
            # Get the log probabilities of the generated tokens.
            output_logits = model(input_ids=input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0)).logits
            # Need to slice the output_logits to the length of the generated output
            generated_token_ids = outputs[i]
            logits = output_logits[0, : generated_token_ids.shape[0] - 1]  # shape: (seq_len-1, vocab_size)

            # Ensure generated_token_ids is not longer than logits
            generated_token_ids = generated_token_ids[: logits.shape[0] + 1]

            probs = torch.softmax(logits, dim=-1)  # (seq_len-1, vocab_size)
            # Use the generated_token_ids to get the log probabilities of the generated tokens.
            action_log_probs = torch.log(
                probs.gather(dim=-1, index=generated_token_ids[1:].unsqueeze(-1)).squeeze(-1)
            )  # shape: (seq_len-1)
            log_probs.append(action_log_probs)

            # Print a sample output
            if i == 0:  # Print only for the first item in the batch for simplicity
                print_sample_output(input_texts[i], generated_texts[i], target_texts[i], batch_rewards[i])
                print_gsm8k_sample(dataset[i])

        # Calculate the policy loss with REINFORCE++
        global_baseline = np.mean(all_rewards)  # Calculate global baseline
        policy_loss = []
        for i in range(len(log_probs)):
            episode_reward = torch.tensor(batch_rewards[i]).to(device)  # Use CPU
            episode_log_probs = log_probs[i]
            advantages = episode_reward - global_baseline
            loss = -episode_log_probs * advantages
            policy_loss.append(loss.mean())
        policy_loss = torch.stack(policy_loss).mean()

        loss = policy_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        epoch_losses.append(loss.item())

    avg_epoch_loss = total_loss / len(dataloader)
    all_epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss}")

# Plotting the loss curve after training is complete
plot_loss_curve(all_epoch_losses, NUM_EPOCHS)
print("Training complete!")

