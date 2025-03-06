import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch.cuda

loaded_model = None
model_name = None

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def move_cursor(row, col):
    print(f"\033[{row};{col}H", end='')

def prune_past_key_values(past_key_values, keep_first=100, keep_last=100):
    """
    Dummy function that prunes all tokens except the first `keep_first`
    and the last `keep_last` along the sequence dimension.
    """
    from transformers.cache_utils import DynamicCache

    new_past = DynamicCache()
    for layer_idx, layer_past in enumerate(past_key_values):
        # Each layer_past is (key, value)
        # For GPT-like models: [batch_size, num_heads, seq_len, head_dim]
        key, value = layer_past
        seq_len = key.size(2)

        if seq_len > (keep_first + keep_last):
            # Keep the first `keep_first` and last `keep_last` tokens
            key = torch.cat([key[:, :, :keep_first, :], key[:, :, -keep_last:, :]], dim=2)
            value = torch.cat([value[:, :, :keep_first, :], value[:, :, -keep_last:, :]], dim=2)

        new_past.update(key, value, layer_idx)

    return new_past

def analyze_model_architecture(model, model_name_input):
    config = model.config

    num_layers = getattr(config, 'num_decoder_layers', None) or getattr(config, 'num_hidden_layers', None) or 0
    layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-05)
    attention_dropout = getattr(config, 'attention_dropout', 0.0)
    hidden_dropout = getattr(config, 'hidden_dropout', 0.0)
    rotary_emb_class_name = getattr(getattr(config, 'rotary_embedding', None), '__class__.__name__', 'RotaryEmbedding')
    hidden_act = getattr(config, 'hidden_act', 'silu')
    max_position_embeddings = getattr(config, 'max_position_embeddings', 'N/A')
    norm_layer_class_name = getattr(config.norm_layers[0], '__class__.__name__', 'LayerNorm') if hasattr(config, 'norm_layers') and config.norm_layers else 'LayerNorm'
    mlp_layer_class_name = getattr(model.config.mlp_layers[0], '__class__.__name__', 'MLPBlock') if hasattr(model.config, 'mlp_layers') and model.config.mlp_layers else 'MLPBlock'
    attention_layer_class_name = getattr(model.config.attention_layers[0], '__class__.__name__', 'AttentionMechanism') if hasattr(model.config, 'attention_layers') and model.config.attention_layers else 'AttentionMechanism'
    decoder_layer_class_name = getattr(model.config.decoder_layers[0], '__class__.__name__', 'DecoderLayer') if hasattr(model.config, 'decoder_layers') and model.config.decoder_layers else 'DecoderLayer'

    explanation = f"""
Model Architecture Interpretation:

This is the architecture of the {model_name_input} model. Let's break down the key components:

**{model.__class__.__name__}:** This is the main model class.

  **(model): {model.base_model_prefix.capitalize()}Model:** This is the core model.

    **(embed_tokens): Embedding({config.vocab_size}, {config.hidden_size}, padding_idx={config.pad_token_id}):**
      - Input embedding layer.
      - **{config.hidden_size}:** Embedding dimension / Hidden size.
      - **padding_idx={config.pad_token_id}:** Padding token index.

    **(layers): ModuleList( (0-{num_layers - 1}): {num_layers} x {decoder_layer_class_name}(...) ):**
      - Stack of {num_layers} decoder layers.

        **({decoder_layer_class_name}):** Decoder layer (details may vary):
          - **(self_attn): {attention_layer_class_name}(...):** Self-attention mechanism.
            - **(o_proj): Linear(in_features={config.hidden_size}, out_features={config.hidden_size}, bias=False):** Output projection.
            - **(qkv_proj): Linear(in_features={config.hidden_size}, out_features={3 * config.hidden_size}, bias=False):** Q, K, V projections.
            - **Number of Attention Heads:** {config.num_attention_heads}

          - **(mlp): {mlp_layer_class_name}(...):** Multi-Layer Perceptron.
            - **(gate_up_proj): Linear(in_features={config.hidden_size}, out_features={config.intermediate_size}, bias=False):** Gate and Up projection.
            - **(down_proj): Linear(in_features={config.intermediate_size}, out_features={config.hidden_size}, bias=False):** Down projection.
            - **(activation_fn): {hidden_act}:** Activation function.

          - **(input_layernorm): {norm_layer_class_name}(({config.hidden_size},), eps={layer_norm_eps}):** Input Layer Normalization.
          - **(post_attention_layernorm): {norm_layer_class_name}(({config.hidden_size},), eps={layer_norm_eps}):** Post-Attention Layer Normalization.
          - **(resid_attn_dropout): Dropout(p={attention_dropout}, inplace=False):** Attention Dropout.
          - **(resid_mlp_dropout): Dropout(p={hidden_dropout}, inplace=False):** MLP Dropout.

    **(norm): {norm_layer_class_name}(({config.hidden_size},), eps={layer_norm_eps}):** Final Layer Normalization.
    **(rotary_emb): {rotary_emb_class_name}():** Rotary Positional Embeddings.

  **(lm_head): Linear(in_features={config.hidden_size}, out_features={config.vocab_size}, bias=False):** Language Model Head.

**Key Takeaways:**

- **Model Size:** Hidden dimension: {config.hidden_size}, Number of layers: {num_layers}.
- **Attention Heads:** Number of attention heads: {config.num_attention_heads}, Number of key-value heads: {config.num_key_value_heads}.
- **Context Window:** Context window size: {max_position_embeddings}.
- **Normalization:** Uses {norm_layer_class_name} for normalization.
- **Activation:** Employs {hidden_act} activation function in MLP.
- **Positional Embeddings:** Uses {rotary_emb_class_name} positional embeddings.

This breakdown is based on the available configuration and might need adjustments for different model architectures.
"""
    return explanation

def display_menu():
    print("\nMenu:")
    print("1. Load Model")
    print("2. Display Details")
    print("3. Plot Visualizations")
    print("4. Run Inference")
    print("5. Quit")

def visualize_attention(attention_matrix, text, tokens, output_filename="attention_heatmap.png"):
    """
    attention_matrix: (seq_len, seq_len) after causal mask
    tokens: list of token strings to display on x/y axis
    """
    seq_len = attention_matrix.shape[0]
    
    # 1) Replace -inf with a large negative for numerical stability
    masked = attention_matrix.copy()
    masked[np.isinf(masked)] = -1e9
    
    # 2) Subtract row-wise max (avoid overflow)
    row_max = np.max(masked, axis=-1, keepdims=True)
    masked -= row_max
    
    exp_scores = np.exp(masked)
    row_sum = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_probs = exp_scores / row_sum  # shape: (seq_len, seq_len)

    plt.figure(figsize=(8, 6))
    plt.title(text + "\nCausal Attention Distribution")
    ax = sns.heatmap(attention_probs, cmap="coolwarm", center=0)

    # Show token labels on both axes
    ax.set_xticks(np.arange(seq_len) + 0.5)
    ax.set_yticks(np.arange(seq_len) + 0.5)
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens, rotation=0)

    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    ax.invert_yaxis()  # Flip Y-axis so 0 is at bottom
    plt.tight_layout()

    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved attention distribution as {output_filename}")

def reshape_for_heads(x, num_heads):
    """
    Reshape tensor x from shape (batch, seq_len, hidden_size)
    to (batch, num_heads, seq_len, head_dim).
    """
    batch, seq_len, hidden_size = x.shape if isinstance(x, Tensor) else x[0].shape
    head_dim = hidden_size // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3) if isinstance(x, Tensor) else x[0].view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

def save_all_attention_heatmaps(model, tokenizer, text, model_name, sample_num, output_base_dir="heat_map"):
    """
    Runs a single forward pass on the input text, and for every layer and every head,
    computes the dot product between the per-token query and key activations (i.e., QK^T),
    and saves the resulting attention matrix as both a PNG heatmap and a raw NumPy file.
    
    Note: This uses forward hooks to capture activations from qkv_proj 
          (since q_proj/k_proj are not separate in this model).
    """
    q_activations = {}
    k_activations = {}
    hooks = []

    # Identify the correct property to access the layers
    candidate_properties = []
    for name in dir(model):
        if not name.startswith("_"):  # Consider only public properties
            attr = getattr(model, name)
            if "Model" in attr.__class__.__name__:
                print(name + " : " + attr.__class__.__name__)
                candidate_properties.append(name)

    actual_model = None
    if hasattr(model, 'transformer'):
        actual_model = model.transformer
    elif hasattr(model, 'model'):
        actual_model = model.model
    elif hasattr(model, 'decoder'):
        actual_model = model.decoder
    elif candidate_properties:
        actual_model = getattr(model, candidate_properties[0])
    else:
        raise AttributeError("Could not find layer container in the model.")

    layer_container = None
    if hasattr(actual_model, 'layers'):
        layer_container = actual_model.layers
    elif hasattr(actual_model, 'h'):
        layer_container = actual_model.h
    else:
        raise AttributeError("Could not find layers in the actual model.")

    n_layers = len(layer_container)  # GPT-style model layers

    # Print sub-attributes for debugging
    for name in dir(layer_container[0].self_attn):
        if hasattr(layer_container[0], name):
            attr = getattr(layer_container[0], name)
            if hasattr(attr, '__class__'):
                print(name + " : " + attr.__class__.__name__)
            else:
                print(name + " : " + str(type(attr)))
        else:
            print(name + " : Attribute can't be accessed")

    # We hook qkv_proj, then split the result into Q and K
    def get_hook_qkv(layer_idx):
        def hook(module, input, output):
            # output has shape: (batch, seq_len, 3 * hidden_size)
            hidden_size = output.shape[-1] // 3
            # Split into Q, K, V (we only store Q and K)
            q_activations[layer_idx] = output[..., :hidden_size]
            k_activations[layer_idx] = output[..., hidden_size:2*hidden_size]
        return hook

    # Register one hook per layer on qkv_proj
    for layer_idx in range(n_layers):
        if not hasattr(layer_container[layer_idx].self_attn, 'qkv_proj'):
            raise AttributeError(
                f"Layer {layer_idx} has no 'qkv_proj' attribute. "
                "This model may not store Q, K, V in a single projection. "
                "Check the layer's architecture."
            )
        qkv_handle = layer_container[layer_idx].self_attn.qkv_proj.register_forward_hook(get_hook_qkv(layer_idx))
        hooks.append(qkv_handle)

    # Run forward pass once
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print("Starting inference")
    model(**inputs, use_cache=True)
    print("Inference done")

    # Remove hooks to avoid side effects later
    for h in hooks:
        h.remove()

    # Convert token IDs to text tokens for axis labeling
    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)  # list of strings (subword tokens)

    num_heads = model.config.num_attention_heads

    # Process and save attention matrices for each layer and each head
    for layer_idx in range(n_layers):
        # q_activations and k_activations have shape (batch, seq_len, hidden_size)
        q_act = q_activations[layer_idx]
        k_act = k_activations[layer_idx]

        q_act_heads = reshape_for_heads(q_act, num_heads)  # shape: (batch, num_heads, seq_len, head_dim)
        k_act_heads = reshape_for_heads(k_act, num_heads)

        for head_idx in range(num_heads):
            # Select the first batch element
            q_head = q_act_heads[0, head_idx].detach().cpu().numpy()  # shape: (seq_len, head_dim)
            k_head = k_act_heads[0, head_idx].detach().cpu().numpy()  # shape: (seq_len, head_dim)
            
            # Compute attention matrix: dot product between q and k transpose
            attention_matrix = np.dot(q_head, k_head.T)  # shape: (seq_len, seq_len)

            # Apply a causal mask so tokens cannot attend to future positions
            seq_len = attention_matrix.shape[0]
            for i in range(seq_len):
                for j in range(i+1, seq_len):
                    attention_matrix[i, j] = float('-inf')

            # Create output directory for this layer and head
            output_dir = os.path.join(output_base_dir, model_name, f"sample_{sample_num}", f"layer_{layer_idx}", f"head_{head_idx}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save heatmap image
            heatmap_filename = os.path.join(output_dir, "attention_heatmap.png")
            visualize_attention(attention_matrix, text, tokens, heatmap_filename)
            
            # Also save the raw attention matrix as a .npy file
            npy_filename = os.path.join(output_dir, "attention_matrix.npy")
            np.save(npy_filename, attention_matrix)
            print(f"Saved raw attention matrix as {npy_filename}")

def get_user_prompt():
    """Gets the prompt from the user."""
    prompt = input("Enter the prompt text: ")
    return prompt

def generate_stream(
    loaded_model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    keep_first: int,
    keep_last: int,
    onNextToken
):
    """
    Streams tokens one-by-one from a causal language model with dummy KV cache pruning.

    Parameters:
    -----------
    loaded_model : AutoModelForCausalLM
        The loaded model (e.g., GPT-2, LLaMA, etc.) with .to(device).
    tokenizer : PreTrainedTokenizer
        The corresponding tokenizer for 'loaded_model'.
    prompt : str
        Initial text prompt.
    max_tokens : int
        Maximum number of new tokens to generate.
    keep_first : int
        Number of tokens to keep at the beginning of the sequence.
    keep_last : int
        Number of tokens to keep at the end of the sequence.
    onNextToken : Callable[[str], None]
        Callback function that receives each decoded token as a string.
    """

    device = next(loaded_model.parameters()).device
    token_count = 0
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if device == torch.device("cuda") else 0

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # We'll accumulate the new tokens here for optional final return
    generated_tokens = []

    # Keep track of past_key_values across steps
    past_key_values = None

    # Generate token by token
    for _ in range(max_tokens):
        # Forward pass (autoregressive, so we pass the last token & the cache)
        outputs = loaded_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )

        # Extract logits and past_key_values
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # Greedy approach: pick the highest-prob token from the last timestep
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        # Store it in a list (useful if you want the final text at the end)
        generated_tokens.append(next_token.item())

        # Decode to string and call the callback
        token_str = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        onNextToken(token_str)

        # Prepare input_ids for the next step
        input_ids = next_token.unsqueeze(0)  # shape [1, 1]

        # Dummy KV pruning: keep first 100 + last 100 tokens
        past_key_values = prune_past_key_values(past_key_values, keep_first, keep_last)

        # Track token count
        token_count += 1

        # Calculate and print stats every 10 tokens
        if token_count % 10 == 0:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if device == torch.device("cuda") else 0
            memory_used = end_memory - start_memory
            time_taken = end_time - start_time
            tokens_per_second = token_count / time_taken
            clear_screen()
            move_cursor(1, 1)
            print(f"Tokens: {token_count}, GPU Memory Used: {memory_used / 1024**2:.2f} MB, Token Generation Speed: {tokens_per_second:.2f} tokens/second\n")
            print(token_str)

    # Optionally return the final generation as a single string
    return tokenizer.decode(generated_tokens, skip_special_tokens=False)

def run_inference(loaded_model, model_name):
    """Runs inference with the given model and prompt."""
    if not loaded_model:
        print("Please load a model first (Option 1).")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name,
        add_prefix_space=True,  # This can help preserve spaces for some models
        skip_special_tokens=False) 
    prompt = input("Enter the prompt text: ")
    try:
        kv_margin = int(input("Enter the KV Margin: "))
    except ValueError:
        print("Invalid input. Please enter an integer for KV Margin.")
        return

    try:
        max_tokens = int(input("Enter the maximum number of tokens: "))
    except ValueError:
        print("Invalid input. Please enter an integer for the maximum number of tokens.")
        return

    def onNextTokenHandler(token_str):
        return

    device = next(loaded_model.parameters()).device

    start_time = time.time()
    
    # GPU memory usage tracking
    if device == torch.device("cuda"):
        torch.cuda.reset_peak_memory_stats(device)  # Reset peak memory stats before inference
        start_memory = torch.cuda.memory_allocated(device)
    else:
        start_memory = 0

    final_output = generate_stream(
        loaded_model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        keep_first=kv_margin,
        keep_last=kv_margin,
        onNextToken=onNextTokenHandler
    )
    end_time = time.time()

    # GPU memory usage tracking
    if device == torch.device("cuda"):
        end_memory = torch.cuda.memory_allocated(device)
        memory_used = end_memory - start_memory
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"\nPeak GPU Memory Used: {peak_memory / 1024**2:.2f} MB")
    else:
        print("\nGPU Memory Used: 0 MB (CPU mode)")

    # Token generation speed tracking
    total_tokens = len(tokenizer(final_output)['input_ids'])
    time_taken = end_time - start_time
    tokens_per_second = total_tokens / time_taken
    print(f"Token Generation Speed: {tokens_per_second:.2f} tokens/second")

    print("\n\n--- Final Generation ---")
    print(final_output)

def plot_heat_map(loaded_model, model_name):
    """Plots the attention heatmaps for a given model and prompt."""
    if not loaded_model:
        print("Please load a model first (Option 1).")
        return

    try:
        tokenizer = loaded_model.tokenizer if hasattr(loaded_model, 'tokenizer') else AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get the prompt from the user
    text = get_user_prompt()
    
    # Ask user which visualization they want
    print("\nVisualization Options:")
    print("1. Attention Heat Maps")
    print("2. Token Confidence Bar Chart")
    viz_choice = input("Enter your choice (1 or 2): ")
    
    if viz_choice == '1':
        save_all_attention_heatmaps(loaded_model, tokenizer, text, model_name, 2)
    elif viz_choice == '2':
        plot_token_confidence(loaded_model, tokenizer, text, model_name)
    else:
        print("Invalid choice. Defaulting to Attention Heat Maps.")
        save_all_attention_heatmaps(loaded_model, tokenizer, text, model_name, 2)

def plot_token_confidence(model, tokenizer, text, model_name, output_filename=None, output_base_dir="token_confidence"):
    """
    Plots a bar chart showing token confidence levels from logits calculation.
    X-axis: tokens, Y-axis: confidence level of each token.
    """
    # Create output directory structure similar to save_all_attention_heatmaps
    sample_num = int(time.time()) % 1000  # Use timestamp modulo 1000 as sample number
    output_dir = os.path.join(output_base_dir, model_name, f"sample_{sample_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    if not output_filename:
        output_filename = os.path.join(output_dir, "token_confidence.png")
    
    device = next(model.parameters()).device
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # Run the model to get logits
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Get logits for the last token prediction (next token)
    logits = outputs.logits[0, -1, :]
    
    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the top N tokens with highest probabilities
    top_n = 20
    top_probs, top_indices = torch.topk(probabilities, top_n)
    
    # Convert to numpy for plotting
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    # Get token strings for the top indices
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
    
    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(top_tokens)), top_probs, color='skyblue')
    plt.xticks(range(len(top_tokens)), top_tokens, rotation=45, ha='right')
    plt.xlabel('Tokens')
    plt.ylabel('Confidence (Probability)')
    plt.title(f'Top {top_n} Token Confidence Levels for Next Token Prediction\nPrompt: "{text}"')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    # Also save the raw probabilities as a .npy file
    npy_filename = os.path.join(output_dir, "token_probabilities.npy")
    np.save(npy_filename, {"probabilities": top_probs, "indices": top_indices, "tokens": top_tokens})
    
    print(f"Saved token confidence bar chart as {output_filename}")
    print(f"Saved raw token probabilities as {npy_filename}")

while True:
    display_menu()
    choice = input("Enter your choice: ")

    if choice == '1':
        model_name = input("Enter model name to load (e.g., microsoft/Phi-3-mini-128k-instruct): ")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            print(f"Model '{model_name}' loaded successfully to {device}.")
            # Add tokenizer to the loaded model if it doesn't have one
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                loaded_model.tokenizer = tokenizer
            except:
                pass
        except Exception as e:
            print(f"Error loading model: {e}")
            loaded_model = None # Reset model on error
            model_name = None
    elif choice == '2':
        if loaded_model:
            try:
                explanation = analyze_model_architecture(loaded_model, model_name)
                print(explanation)
            except Exception as e:
                print(f"Error analyzing model architecture: {e}")
        else:
            print("Please load a model first (Option 1).")
    elif choice == '3':
        plot_heat_map(loaded_model, model_name)
    elif choice == '4':
        run_inference(loaded_model, model_name)
    elif choice == '5':
        print("Quitting.")
        break
    else:
        print("Invalid choice. Please enter 1, 2, 3, 4 or 5.")
