### **All of the notebooks are executed in kaggle**  

### **Summary of AutoModel.ipynb**  
This notebook focuses on building a machine learning model using **TensorFlow** and **transformers** for text classification. It starts by installing dependencies such as `transformers`, `tensorflow`, and `nltk`. It imports various NLP-related libraries and datasets, including the **Supreme Court dataset** from `textacy`. The preprocessing pipeline includes **stopword removal, tokenization, and text normalization**. The model architecture utilizes **BERT embeddings** with dense layers for classification. The notebook includes performance evaluation using **confusion matrices, F1 scores, and classification reports**. The workflow follows a structured **train-test split**, and results are visualized using `matplotlib`.

This notebook utilizes **BERT (Bidirectional Encoder Representations from Transformers)** for **text classification**. The model is implemented using **TensorFlow and Keras**. 

1. **BERT Tokenization & Embedding**  
   - The notebook imports `BertTokenizer` and `TFBertModel` from `transformers`.  
   - Text is tokenized using **BERT Tokenizer**, converting text into numerical format for model processing.  
   - `TFBertModel` is used to extract contextual word embeddings.

2. **Neural Network Architecture**  
   - **Input Layer**: Takes tokenized text input.  
   - **BERT Layer**: A frozen or fine-tuned BERT model for feature extraction.  
   - **Fully Connected Layers**:  
     - **Dense layers with Batch Normalization & Dropout** are used for classification.  
     - **Activation Function**: `ReLU` is applied in hidden layers.  
   - **Output Layer**: Uses `Softmax` activation for classification.

3. **Evaluation Metrics**  
   - The model is trained with `Categorical Crossentropy` loss and optimized using `Adam`.  
   - Performance is evaluated using **F1-score, Confusion Matrix, and Classification Report**.


---

### **Summary of classify-llama-prompt.ipynb**  
This notebook integrates **Ollama** with `llama3` for **text classification and chat-based AI processing**. It starts by installing `ollama` and running the LLaMA3 model. The workflow includes querying LLaMA using the **Ollama API** for text-based interactions. Additionally, the **Supreme Court dataset** from `textacy` is loaded and processed into a `DataFrame`. The script leverages **asynchronous processing** using `AsyncClient` to handle multiple tasks efficiently. The notebook demonstrates querying LLaMA3 for text classification, making use of structured **prompt engineering** techniques. The dataset records are converted into a structured format, making it suitable for text-based ML applications.

This notebook integrates **LLaMA 3 (Large Language Model Meta AI)** for **text classification and NLP-based inference** via **Ollama**.

1. **LLaMA 3 via Ollama API**  
   - The model is downloaded and run using `ollama pull llama3`.  
   - The **Ollama API** is used to send and receive responses.  
   - It interacts with text data using structured **prompts**.

2. **Classification Model via Prompting**  
   - The function `classify_case_local()` sends a text query to the LLaMA model and retrieves a classification response.  
   - The model processes Supreme Court case text and provides legal text classification based on contextual understanding.  
   - The approach relies on **zero-shot or few-shot learning**, where the model classifies based on the provided prompt without explicit retraining.

3. **Asynchronous Processing**  
   - The `AsyncClient` is used to handle multiple requests efficiently for text classification.


---

### **Summary of RAG approach (still in planning)**  

I though that we could create a RAG with various embedding models and experiment around with metadata/custom weights to get good retrivals to pass to the LLM classification. We can approach this from two methods:

1. **One shot the RAG model**
   - We can classify single cases based on the retrieved docs from the RAG by applying a statisical model on it.
   - This would be a direct extension of the approach in classify-llama-prompt.ipynb.
   - It is expected to pass better context to the LLM.


2. **Identify phrases/words which are related to the case classification:**
   - We can ask the model's reasoning ability to highlight which phrases/words cause the classification using RAG.
   - Take those phrases/words and apply techniques to identify the relation of the phrases/words and create a model on it.
   - Then use this model to observe its accuracy. Sort of like a train-test scenario.


