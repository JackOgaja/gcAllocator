import torch
import torch.nn as nn
import gcallocator as gca
import gc

def test_oom_handling():
    """Test OOM handling with appropriately sized tensors"""
    
    # Advanced configuration with custom retry strategy
    retry_config = {
        'max_retries': 5,
        'initial_delay_ms': 50,
        'backoff_multiplier': 1.5,
        'max_delay_ms': 2000,
        'enable_cache_flush': True,
        'enable_gradient_checkpointing': True
    }

    # Install with configuration
    manager = gca.install_allocator(enable_logging=True, **retry_config)

    try:
        # Check available CUDA memory
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            print(f"Total CUDA memory: {total_memory / (1024**3):.2f} GB")
            
            # Use a more reasonable size that will actually use CUDA memory
            # and potentially trigger OOM for testing
            feature_size = min(50000, int((total_memory * 0.8) ** 0.5 / 4))  # Conservative estimate
            print(f"Using feature size: {feature_size}")
            
            # Create model directly on CUDA to bypass CPU allocation
            print("Creating model directly on CUDA...")
            model = nn.Linear(feature_size, feature_size, device='cuda', dtype=torch.float32)
            
            # Register checkpoint callback
            checkpoint_callback = gca.create_checkpoint_callback(model)
            manager.register_checkpoint_callback(checkpoint_callback)
            
            print("Starting training loop...")
            
            # Training loop that might trigger OOM
            for epoch in range(5):
                try:
                    print(f"Epoch {epoch + 1}/5")
                    
                    # Gradually increase batch size to trigger OOM
                    batch_size = 1000 + epoch * 500
                    print(f"  Batch size: {batch_size}")
                    
                    # Create input tensor directly on CUDA
                    x = torch.randn(batch_size, feature_size, device='cuda', dtype=torch.float32)
                    
                    # Forward pass
                    output = model(x)
                    loss = output.mean()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Simulate gradient update
                    with torch.no_grad():
                        for param in model.parameters():
                            param -= 0.01 * param.grad
                    
                    model.zero_grad()
                    
                    # Clean up
                    del x, output, loss
                    
                    if epoch % 2 == 0:
                        manager.print_stats()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  OOM caught at epoch {epoch + 1}: {e}")
                        # Force cleanup
                        gc.collect()
                        torch.cuda.empty_cache()
                        manager.print_stats()
                    else:
                        raise
                        
        else:
            print("CUDA not available - cannot test CUDA allocator")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        
    finally:
        print("Cleaning up...")
        gca.uninstall_allocator()

def test_extreme_oom():
    """Test extreme OOM conditions"""
    
    manager = gca.install_allocator(enable_logging=True, max_retries=10)
    
    try:
        if torch.cuda.is_available():
            print("Testing extreme OOM conditions...")
            
            # Get available memory
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Try to allocate more memory than available
            try:
                # This should trigger OOM and retry logic
                oversized_tensor = torch.zeros(
                    int(total_memory // 4) + 1000000,  # More than available
                    device='cuda',
                    dtype=torch.float32
                )
                print("Unexpected: Large allocation succeeded")
                del oversized_tensor
                
            except RuntimeError as e:
                print(f"Expected OOM caught: {e}")
                manager.print_stats()
                
    finally:
        gca.uninstall_allocator()

if __name__ == "__main__":
    print("=== Testing GCAllocator OOM Handling ===")
    test_oom_handling()
    
    print("\n=== Testing Extreme OOM Conditions ===")
    test_extreme_oom()
    
    print("\n=== Tests Complete ===")
