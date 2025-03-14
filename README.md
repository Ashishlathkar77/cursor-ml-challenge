# Cursor ML takehome interview

## Problem 
Your goal is to implement [Contrastive Decoding](https://arxiv.org/abs/2210.15097) with HuggingFace transformers and PyTorch.

Your code should use `Qwen/Qwen2.5-3B-Instruct` as the large model and `Qwen/Qwen2.5-Coder-0.5B-Instruct` as the small model and be implemented in `main.py`.

Your code should be correct first, but also efficient. Implement the token-level algorithm, rather than the beam search algorithm.

In addition to implementing main.py, please answer the following questions:

1. What should you do if the two models have different tokenizers?
2. Do you think contrastive decoding is used in practice?

## My Answers to Given Questions

### 1. What should you do if the two models have different tokenizers?

In my experience, when working with two models that have different tokenizers, the main challenge is ensuring that the tokens generated by both models align correctly for downstream processing. If two models use different tokenizers, they might tokenize the input text differently, which could lead to inconsistent results when generating or decoding outputs.

To handle this, here are the steps I would take:

- **Check Tokenizer Compatibility:** First, I would ensure that the tokenizers of both models are compatible with each other. If the models come from the same family (e.g., both are from the Hugging Face ecosystem), they might share similar tokenization methods, which would make the integration smoother. However, this is not always guaranteed.
  
- **Align Tokenizers:** If the tokenizers are indeed different (for example, one uses byte pair encoding and the other uses wordpiece), I would convert the tokenized input from one model to match the format expected by the other. This could involve padding, truncation, or altering special tokens like `eos`, `pad`, or `unk`.

- **Standardize Tokenization:** Another approach is to standardize tokenization by using a common tokenizer for both models. This can be achieved by selecting a universal tokenizer from the Hugging Face library (such as `GPT2Tokenizer` or `BertTokenizer`) and using it across both models. By doing this, we would ensure that both models receive the same tokenized representation of the input, reducing inconsistencies between them.

- **Handle Special Tokens:** Some models might have unique special tokens (e.g., `bos`, `eos`, `pad`). I would ensure that these special tokens are handled properly, particularly when dealing with text generation. For example, the `eos_token` for one model might be different from the `eos_token` for the other, leading to potential issues if not aligned properly.

- **Pre-tokenization Adjustments:** If the tokenizers are inherently incompatible (say, one uses subword tokenization and the other tokenizes words directly), I might preprocess the input text by applying a consistent tokenization strategy before feeding it to both models. This could involve segmenting the input text into consistent subword units, thus mitigating the discrepancy.

In summary, ensuring compatibility between two different tokenizers is key for achieving effective contrastive decoding. Standardizing tokenization or using a shared tokenizer are practical solutions to ensure smooth interactions between the models.

### 2. Do you think contrastive decoding is used in practice?

Yes, I do believe contrastive decoding has practical use cases, especially when working with large and small language models in production settings.

From my experience, contrastive decoding works particularly well in scenarios where we want to leverage the strengths of both large and small models. Here's why I think contrastive decoding could be useful:

- **Efficiency and Cost Optimization:** Large models tend to be computationally expensive, and while they can generate high-quality outputs, they may not always be necessary for simpler tasks. By using a smaller model in conjunction with a larger model, contrastive decoding allows us to take advantage of the smaller model’s efficiency for simple tasks, while still using the larger model for more complex, nuanced generation tasks. This hybrid approach can reduce computational costs without sacrificing quality.

- **Scaling Down for Speed:** The contrastive decoding approach allows for faster and more efficient text generation by having the smaller model handle simpler, straightforward components of the task, while the larger model handles more intricate, complex aspects. For example, the small model might be used for generating initial outputs or suggestions, while the large model refines or finalizes the output. This can significantly speed up the inference time compared to using only the large model.

- **Fine-Tuning Accuracy:** Contrastive decoding also facilitates better fine-tuning. By using the smaller model to refine the token-level predictions of the larger model, we can leverage the strengths of both models. This can help achieve better accuracy by having the smaller model contribute to more precise or specific predictions based on token-level details, while the larger model provides context and understanding.

- **Training and Prompt Optimization:** The contrastive decoding technique also enables the integration of prompts and responses from both models, where one model can serve as a baseline (the smaller model) and the other (the larger model) can improve upon that baseline. This allows for more controlled generation and potentially more diverse outputs.

However, there are some considerations to keep in mind:
- **Training and Calibration:** The technique requires proper training and calibration between the two models to ensure they work together effectively. In practice, this means that both models should be fine-tuned to understand each other's outputs and provide complementary results.
- **Additional Complexity:** Contrastive decoding can add complexity to the decoding pipeline, as it involves coordinating outputs from two different models. Depending on the application, this added complexity might not always be justified if simpler, single-model solutions can achieve the same result.

In conclusion, I believe contrastive decoding is definitely used in practice, especially when there's a need for efficiency, scalability, or cost optimization, and when combining the strengths of different models can lead to better results. The technique has real-world potential for balancing performance and resource usage, particularly in large-scale NLP applications where efficiency matters.