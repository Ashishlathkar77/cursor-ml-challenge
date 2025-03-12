import huggingface_hub
import transformers as tr
import torch

# Manually fetch your Hugging Face token
hf_token = "<Your_HuggingFace_Token>"

# Authenticate using the token
huggingface_hub.login(token=hf_token)

# Load model paths
amateur_path = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
expert_path = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'

# Load tokenizers and models
amateur_tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
expert_tokenizer = tr.AutoTokenizer.from_pretrained(expert_path)
amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path)

# Prepare the user message
user_message = """Give a very very brief docstring (1-2 sentences) for the following function:
function updateEloScores(
    scores,
    results,
    kFactor = 4,
) {
    for (const result of results) {
        const { first, second, outcome } = result;
        const firstScore = scores[first] ?? 1000;
        const secondScore = scores[second] ?? 1000;

        const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
        const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
        let sa = 0.5;
        if (outcome === 1) {
            sa = 1;
        } else if (outcome === -1) {
            sa = 0;
        }
        scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
        scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
    }
    return scores;
}\n"""

# Tokenize the user input for both amateur and expert
amateur_input_ids = amateur_tokenizer.encode(user_message, return_tensors='pt', padding=True, truncation=True)
expert_input_ids = expert_tokenizer.encode(user_message, return_tensors='pt', padding=True, truncation=True)

def contrastive_generation(amateur_model, expert_model, amateur_tokenizer, expert_tokenizer, prompt, max_tokens=50):
    # Generate hypotheses using the amateur (smaller) model
    amateur_output = amateur_model.generate(
        input_ids=amateur_input_ids,
        max_length=len(amateur_input_ids[0]) + max_tokens,
        num_return_sequences=5,  # Generate 5 hypotheses
        do_sample=True,  # Sampling for diversity
        top_p=0.9,  # Nucleus sampling for diversity
        temperature=0.7  # Slightly reduce temperature to avoid extreme randomness
    )

    # Decode the generated hypotheses
    hypotheses = [amateur_tokenizer.decode(output, skip_special_tokens=True) for output in amateur_output]

    # Score hypotheses using the expert model
    scores = []
    for hypothesis in hypotheses:
        # Tokenize hypothesis for expert model
        expert_input_ids = expert_tokenizer.encode(hypothesis, return_tensors='pt', padding=True, truncation=True)

        # Get log-likelihood scores from the expert model
        with torch.no_grad():
            expert_output = expert_model(input_ids=expert_input_ids, labels=expert_input_ids)
            log_likelihood = expert_output.loss.item()  # Negative log-likelihood
        scores.append(log_likelihood)

    # Select the best hypothesis (lowest log-likelihood, more likely to be correct)
    best_hypothesis = hypotheses[scores.index(min(scores))]

    return best_hypothesis

# Get the best hypothesis using contrastive generation
best_hypothesis = contrastive_generation(amateur_model, expert_model, amateur_tokenizer, expert_tokenizer, user_message)
print("Best Hypothesis:", best_hypothesis)