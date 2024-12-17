# Model_Manage

This project provides a **dynamic, flexible, and efficient framework** for managing and executing inference tasks for multiple machine learning models. It leverages Python's **registration mechanism** to encapsulate and register inference functions, making it easy to extend, maintain, and scale across diverse model architectures.

---

## Features

1. **Dynamic Inference Function Registration**  
   - Models are integrated using instance-level and class-level registration mechanisms.  
   - The `@decorator` approach allows new inference methods to be registered dynamically without modifying the core framework.

2. **Batch Inference Support**  
   - The system is optimized for batch processing, allowing multiple inputs to be processed efficiently in a single call.  
   - Tokenization, padding, and attention mask generation are automatically handled for batched inputs.

3. **Flexible Model Management**  
   - Models can be **easily loaded, switched, and unloaded** on demand to optimize resource utilization.  
   - Supports Hugging Face models (`transformers` library), `vLLM` for high-performance inference, and LoRA adapters for efficient fine-tuning.

4. **Seamless Integration of Different Models**  
   - Supports models such as:
     - Hugging Face Transformers (e.g., LLaMA, ChatGLM)
     - vLLM framework with LoRA (Low-Rank Adaptation)
     - Custom models via easy-to-extend interfaces.

5. **Efficient Resource Management**  
   - GPU memory usage is optimized with proper device handling and resource release (`torch.cuda.empty_cache()`).

6. **Modular and Extensible Design**  
   - New models and inference methods can be added by implementing and registering their respective inference logic.  
   - Clear separation of concerns: model loading, registration, and inference logic are decoupled.

---

## Directory Structure

```
model_manage/
│
├── .env                        # Environment file for managing API keys or configurations
├── batch_infer_register_cl.py  # Class-level batch inference registration implementation
├── batch_infer_register.py     # Instance-level batch inference registration implementation
├── bert_batch_infer.py         # Specific implementation for BERT batch inference
├── infer_register_cl.py        # Core class-level inference registration logic
├── infer_register.py           # Core instance-level inference registration logic
├── model_inform.py             # Model configuration and path management
└── readme.md                   # Project documentation
```

---

## Core Components

### 1. **Inference Registration Mechanism**  
- Both class-level (`infer_register` in `ModelInfer_Cl`) and instance-level registration are implemented.  
- Using the **decorator approach** (e.g., `@ModelInfer_Cl.infer_register`), developers can easily register new inference methods for specific model types.

**Example:**
```python
@ModelInfer_Cl.infer_register("hf_llama2")
def hf_llama2_infer(model, tokenizer, input_data, **kwargs):
    # Custom inference logic for Hugging Face LLaMA 2
    ...
    return responses
```

---

### 2. **Batch Inference Logic**  
- Input data is processed in **batches** to improve efficiency.  
- Automatic tokenization, padding, and attention mask generation ensure the framework handles variable input lengths seamlessly.

**Advantages of Batch Inference:**
- **Efficiency**: Reduces inference time by processing multiple inputs in parallel.  
- **Scalability**: Easily scales to large datasets without code changes.  
- **Flexibility**: Supports different frameworks (Hugging Face, vLLM) and custom configurations.

---

### 3. **Flexible Model Switching**  
- Models can be dynamically loaded and switched at runtime using `switch_model`.  
- Resources for previously loaded models are released to **optimize GPU memory**.

**Example:**
```python
batch_manager.switch_model(infer_type="hf", model_name="hf_llama2")
```

---

## Key Advantages

1. **Extensible Framework**:  
   - New models and inference methods can be added without modifying the core framework.

2. **Dynamic Registration**:  
   - Flexible registration ensures that inference methods are cleanly organized and reusable.

3. **Batch Processing Efficiency**:  
   - Optimized for batch inference to handle large datasets with minimal overhead.

4. **Multi-Framework Support**:  
   - Combines Hugging Face Transformers, vLLM, and LoRA for maximum flexibility.

5. **Resource Optimization**:  
   - Proper GPU memory management and resource release ensure efficient hardware usage.

---

## Usage Instructions

1. **Install Required Dependencies**  
   - Ensure `torch`, `transformers`, `vllm`, `peft`, and other required libraries are installed.

2. **Set Environment Variables**  
   Configure API keys and paths in the `.env` file.

   Example `.env`:
   ```env
   OPENAI_API_KEY=your_openai_key
   BASE_URL=your_base_url
   ```

3. **Run Batch Inference**  
   Load a model and perform batch inference on input data.

   **Example Script:**
   ```python
   from batch_infer_register_cl import ModelInfer_Cl

   batch_manager = ModelInfer_Cl()
   batch_manager.switch_model(infer_type="hf", model_name="hf_llama2")

   input_data = ["Input text 1", "Input text 2"]
   results = batch_manager.model_infer(input_data)
   print(results)
   ```


---
## Extending the Framework

To add a new inference method:
1. Define the inference function.
2. Register it using `@ModelInfer_Cl.infer_register("model_name")`.

**Example:**
```python
@ModelInfer_Cl.infer_register("new_model")
def new_model_infer(model, tokenizer, input_data, **kwargs):
    # Custom inference logic for the new model
    ...
    return responses
```

