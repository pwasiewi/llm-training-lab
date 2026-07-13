class SentimentConfig:
    """
    Configuration for sentiment analysis transformer model.
    """

    def __init__(self):
        self.vocab_size = 50000
        self.hidden_size = 512
        self.num_heads = 4
        self.num_layers = 6
        self.max_seq_length = 512
        self.dropout = 0.1
        self.batch_size = 32
        self.learning_rate = 2e-4
        self.num_epochs = 10
        self.warmup_steps = 1000
        self.seed = 42  # Added seed for reproducibility
        self.train_size = 2000  # Size of training dataset
        self.test_size = 500  # Size of test dataset
        self.val_size = 200  # Size of validation dataset

def calculate_vram_usage(config):
    """
    Calculates approximate VRAM usage for the Sentiment Transformer model.
    
    Args:
        config: SentimentConfig object containing model parameters
        
    Returns:
        Dictionary with detailed memory usage in MB and total in GB
    """
    # Constants
    BYTES_PER_PARAM = 4  # Float32
    MB = 1024 * 1024    # Bytes in a MB
    
    def to_mb(bytes):
        return bytes / MB
    
    memory = {}
    
    # 1. Model Parameters
    # Embeddings
    token_embedding = config.vocab_size * config.hidden_size * BYTES_PER_PARAM
    position_embedding = config.max_seq_length * config.hidden_size * BYTES_PER_PARAM
    memory['embeddings'] = to_mb(token_embedding + position_embedding)
    
    # Transformer Blocks
    per_block = (
        # Multi-head attention
        (3 * config.hidden_size * config.hidden_size * BYTES_PER_PARAM) +  # QKV projection
        (config.hidden_size * config.hidden_size * BYTES_PER_PARAM) +      # Output projection
        # Feed-forward network
        (config.hidden_size * (4 * config.hidden_size) * BYTES_PER_PARAM) +  # First FFN
        ((4 * config.hidden_size) * config.hidden_size * BYTES_PER_PARAM) +  # Second FFN
        # Layer norms (parameters and running stats)
        (4 * config.hidden_size * BYTES_PER_PARAM)                          # 2 layer norms per block
    )
    memory['transformer_blocks'] = to_mb(per_block * config.num_layers)
    
    # Classifier layers
    classifier_params = (
        (config.hidden_size * (config.hidden_size // 2) * BYTES_PER_PARAM) +  # First layer
        ((config.hidden_size // 2) * BYTES_PER_PARAM) +                       # First layer bias
        ((config.hidden_size // 2) * 1 * BYTES_PER_PARAM) +                   # Output layer
        BYTES_PER_PARAM                                                       # Output layer bias
    )
    memory['classifier'] = to_mb(classifier_params)
    
    # 2. Batch Memory
    # Forward activations
    batch_memory = (
        config.batch_size * config.max_seq_length * config.hidden_size * BYTES_PER_PARAM
    )
    # Attention memory (scales quadratically with sequence length)
    attention_memory = (
        config.batch_size * config.num_heads * 
        config.max_seq_length * config.max_seq_length * BYTES_PER_PARAM
    )
    memory['batch_activations'] = to_mb(batch_memory)
    memory['attention_maps'] = to_mb(attention_memory)
    
    # 3. Gradient Memory (roughly same as forward pass)
    memory['gradients'] = memory['batch_activations'] + memory['attention_maps']
    
    # 4. Optimizer States (Adam uses 2 states per parameter)
    total_params = sum([
        config.vocab_size * config.hidden_size,                    # Token embedding
        config.max_seq_length * config.hidden_size,               # Position embedding
        per_block * config.num_layers,                            # Transformer blocks
        classifier_params                                          # Classifier
    ])
    memory['optimizer'] = to_mb(total_params * 2 * BYTES_PER_PARAM)  # 2 states for Adam
    
    # 5. Additional Buffer Memory (approximately 10% of total)
    subtotal = sum(memory.values())
    memory['buffers'] = subtotal * 0.1
    
    # Calculate totals
    memory['total_mb'] = sum(memory.values())
    memory['total_gb'] = memory['total_mb'] / 1024
    memory['recommended_gb'] = memory['total_gb'] * 1.5  # 50% safety margin
    
    return memory

def print_vram_requirements(config):
    """
    Prints formatted VRAM requirements for the model.
    
    Args:
        config: SentimentConfig object
    """
    memory = calculate_vram_usage(config)
    
    print("Estimated VRAM Usage Breakdown:")
    print("-" * 50)
    print(f"Model Embeddings:       {memory['embeddings']:.2f} MB")
    print(f"Transformer Blocks:     {memory['transformer_blocks']:.2f} MB")
    print(f"Classifier:            {memory['classifier']:.2f} MB")
    print(f"Batch Activations:     {memory['batch_activations']:.2f} MB")
    print(f"Attention Maps:        {memory['attention_maps']:.2f} MB")
    print(f"Gradients:            {memory['gradients']:.2f} MB")
    print(f"Optimizer States:      {memory['optimizer']:.2f} MB")
    print(f"Additional Buffers:    {memory['buffers']:.2f} MB")
    print("-" * 50)
    print(f"Total VRAM Required:   {memory['total_gb']:.2f} GB")
    print(f"Recommended VRAM:      {memory['recommended_gb']:.2f} GB")
    print("\nNote: Actual usage might be higher due to PyTorch's memory management")
    print("and other runtime operations.")

# Example usage
if __name__ == "__main__":
    # Default configuration
    config = SentimentConfig()
    print("Default Configuration VRAM Requirements:")
    print_vram_requirements(config)
    
    print("\n" + "="*60 + "\n")
    
    # Larger configuration example
    large_config = SentimentConfig()
    large_config.hidden_size = 512
    large_config.num_heads = 4
    large_config.num_layers = 6
    large_config.vocab_size = 50000
    large_config.batch_size = 32
    print("Large Configuration VRAM Requirements:")
    print_vram_requirements(large_config)
