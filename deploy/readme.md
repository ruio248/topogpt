### **Introduction**

This `deploy` directory contains all necessary components for deploying a large language model (LLM) efficiently using **vLLM**. Compared to traditional deployment frameworks like FastAPI and Flask, **vLLM** provides several key advantages:
1. **High Throughput**: Leveraging **Ray** as the backend, vLLM achieves superior throughput for serving large-scale language models.
2. **OpenAI-Compatible API**: The deployment mimics OpenAI's API format, making integration with OpenAI clients seamless.
3. **Scalability and Efficiency**: vLLM supports optimized batching and efficient GPU memory utilization, which are critical for serving large models in real-time scenarios.

This directory also includes tools to test and interact with the deployed LLM through a **web demo** and **command-line client**.

---

### **Directory Structure**

```plaintext
deploy/
│
├── .env                    # Environment file containing API keys and server configurations
├── openai_chat.py     # Command-line client for testing the OpenAI-compatible API
├── readme.md               # Documentation for deployment setup and usage
├── vllm_deploy.slurm       # SLURM script for deploying the LLM with vLLM
└── web_demo.py             # Web-based demo interface using Gradio to interact with the model
```

---

### **Component Descriptions**

1. **`.env`**  
   - Stores environment variables, such as the `OPENAI_API_KEY` and `OPENAI_API_BASE` URL for the API server.
   - Ensures sensitive configuration details are not hardcoded.

2. **`openai_chat_test.py`**  
   - A **command-line client** that mimics the OpenAI API client behavior.
   - Allows users to send test queries and interact with the deployed model via the vLLM API server.
   - Key Features:
     - Loads API configuration from the `.env` file.
     - Supports conversation history for chat-based tasks.

3. **`vllm_deploy.slurm`**  
   - A SLURM script for submitting the LLM deployment job on a high-performance computing (HPC) cluster.
   - Key Features:
     - Specifies job parameters such as GPU allocation, job name, and output logs.
     - Launches the vLLM server for hosting the model.

4. **`web_demo.py`**  
   - A **Gradio-based web interface** for interacting with the model in real-time.
   - Key Features:
     - Provides a user-friendly web UI for entering queries and receiving responses.
     - Designed for demonstration and debugging purposes.

---

### **Workflow**

1. **Deploy the Model**:
   - Submit the `vllm_deploy.slurm` script to the SLURM scheduler:
     ```bash
     sbatch vllm_deploy.slurm
     ```
   - This starts the vLLM API server on the specified node and GPU.

2. **Test the Deployment**:
   - Ensure the `.env` file is properly configured with:
     ```plaintext
     OPENAI_API_KEY=your_api_key_here
     OPENAI_API_BASE=http://<server_ip>:<port>/v1
     ```
   - Use the **command-line client** (`openai_chat.py`) to interact with the API server:
     ```bash
     python openai_chat.py
     ```

3. **Run the Web Demo**:
   - Launch the Gradio-based web demo to interact with the model through a browser:
     ```bash
     python web_demo.py
     ```
   - Gradio will provide a local and shareable URL for accessing the interface.

4. **Monitor and Debug**:
   - Monitor the output logs specified in the `vllm_deploy.slurm` script.
   - Test the API endpoints to ensure responses are accurate and timely.

