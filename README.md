# LLMops Model Versioning
![Static Badge](https://img.shields.io/badge/Mlflow-%23ffffff?style=for-the-badge&logo=Mlflow&logoColor=black&labelColor=%230194E2&color=white)
![Static Badge](https://img.shields.io/badge/PyTorch-%23ffffff?style=for-the-badge&logo=PyTorch&logoColor=black&labelColor=%23EE4C2C&color=white)
![Static Badge](https://img.shields.io/badge/HuggingFace-%23ffffff?style=for-the-badge&logo=HuggingFace&logoColor=black&labelColor=%23FFD21E&color=white)
![Static Badge](https://img.shields.io/badge/Transformers-%23ffffff?style=for-the-badge&logo=HuggingFace&logoColor=black&labelColor=%23FFD21E&color=white)
![Static Badge](https://img.shields.io/badge/Typing-%23ffffff?style=for-the-badge&logo=Python&logoColor=black&labelColor=%233776AB&color=white)


This project offers a path to version models, data, and runs using MLflow. This approach enables data professionals to compare model performances after changes to model hyperparameters, application architecture, data ingestion, and/or metrics used.

Comparing the performance of a data solution allows the data professional to select a model that balances latency, throughput, reliability, accuracy metrics, and other performance indicators. It is important not only to have the model with the best accuracy but also one that can be more easily implemented in the overall data solution.

However, as always, the first step in deciding whether to use an LLM solution is to analyze if it is truly necessary and viable for your current goal and overall context. If it is, then the following repository may help you start the journey of LLMops Model Versioning with MLflow.

## Setting Up the MLflow Server

1. **Create and Activate Environment**  
   Create and activate a Conda or Virtualenv environment.

2. **Install Dependencies**

   - First, visit [PyTorch's Get Started page](https://pytorch.org/get-started/locally/) to select and install the appropriate CUDA version for your system if you require GPU support.

   - Next, install the necessary dependencies and sub-dependencies listed in the `requirements.txt` file by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Start the MLflow Server**  
   To set up the MLflow server, you need an artifact store where MLflow saves model artifacts like models and plots. You can use S3, Azure Blob Storage, Google Cloud Storage, or even a shared filesystem. Additionally, a tracking server is needed to log experiment data. By default, it logs to the local filesystem, but for more robust use, you may want to set up a database like MySQL or SQLite. This project will configure an authentication mechanism to prevent unauthorized users from accessing the logged experiment data and artifacts. The MLflow server will use port 5000:
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --app-name basic-auth --port 5000
   ```

4. **Configure Authentication**  
   To use the `auth.py` file and change the login credentials, create a `.env` file with the following format:
   ```env
   MLFLOW_TRACKING_URI=http://localhost:5000
   MLFLOW_TRACKING_USERNAME=admin
   MLFLOW_TRACKING_PASSWORD=password
   MLFLOW_TRACKING_NEW_PASSWORD=new_password
   ```
   The default login credentials are `admin` and `password`.

## MLOps Best Practices

The versioning code is divided into files by each overall goal: authentication setup, training logging, and inferencing. Defined and used functions have docstrings to simplify debugging and code refactoring. Type hints are used for the same goal. The latter is especially useful because some objects from the `torch` and `transformers` libraries are similar, with the same methods but different architectures. For example, `FastTokenizer` is used for smaller input data, while `LazyTokenizer` is used for larger data volumes. Lazy models delay the loading of the model's full state until it is actually needed, drastically reducing the initial memory footprint.

## Fine-tuning

1. **Tokenization and Padding**  
   Using the same base model for tokenization and fine-tuning is important because different LLM architectures may use different tokenization strategies and padding mechanisms. For instance, models like GPT-2 (which is autoregressive) do not use eos tokens, which can affect how padding and tokenization work.

2. **Dataset Preparation**  
   The dataset used for fine-tuning is shuffled, tokenized, and saved to a parquet file format. Parquet datasets are used because they reduce storage space and speed up data loading and processing due to their optimized compression and efficient columnar access. After converting the data to parquet format, dataloaders are used to handle the data in batches for the fine-tuned model. Training and testing datasets are logged to the run before fine-tuning.

3. **GPU Check**  
   Since both the data and model will be moved to GPU (if available), the script checks if CUDA is available on the system.

4. **Optimizer**  
   For training, the AdamW optimizer is used for optimizing gradient descent calculation. AdamW is preferred over Adam because it applies weight decay (L2 regularization) directly to the parameters rather than through the gradients, resulting in more stable training.

5. **Performance Logging**  
   During training, the current model accuracy, precision, recall, and F1 performance are logged. While evaluating, model gradient calculation and dropout layers are disabled. During this stage, batch normalization is adjusted to use the mean and variance accumulated during training. This allows the data professional to analyze the overall model’s change in these performance criteria during each epoch.

   After training, both the base torch and fine-tuned models are logged and registered for later use.

## Inferencing with the Registered Models

Each of the models registered during the latter stage is instantiated from the MLflow artifact storage. Two simple demonstration examples are defined to showcase the difference in performance results of the fine-tuning process executed. As the demonstration examples are small, the fast version of the distillbert base uncased tokenizer is used to tokenize the inputs for the registered base torch model. Both models are used for inferencing separately, and their results are compared.

After inferencing, a new run is executed to log and register a new version of the base model. In MLflow, it is possible to keep multiple versions of a model, allowing the data professional to switch to the preferred one based on context and goals. After logging the new version of the base torch model, the previous version is chosen to be deleted. To demonstrate, the registered model itself is then deleted.
