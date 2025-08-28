Sequence-to-Sequence German to English Translation with PyTorch Lightning
=========================================================================

This project implements a Sequence-to-Sequence (Seq2Seq) neural machine translation model using LSTMs. The model is built with PyTorch and trained using the PyTorch Lightning framework to translate sentences from German to English.

Project Overview
----------------

The goal of this project is to build and train an Encoder-Decoder network capable of performing neural machine translation (NMT). The model learns to map a sequence of words in a source language (German) to a corresponding sequence in a target language (English).

This implementation showcases a comprehensive understanding of:

-   **Seq2Seq Architecture:** The classic Encoder-Decoder model using LSTM recurrent neural networks.

-   **NLP Data Processing:** A full pipeline including tokenization with `spacy`, building custom vocabularies, and numericalizing text data.

-   **Modern Training Frameworks:** Leveraging PyTorch Lightning to create clean, organized, and reproducible training and validation loops.

-   **Advanced Training Techniques:** Implementation of "teacher forcing" to aid in training convergence.

-   **Efficient Data Handling:** Using a custom `collate_fn` with `DataLoader` to handle variable-length sequences through padding.

Methodology
-----------

### 1\. Data Preparation & Preprocessing

-   **Dataset:** The model is trained on the **Multi30k** dataset, which contains 30,000 sentences, each with a German version and a corresponding English translation. The dataset is loaded using the Hugging Face `datasets` library.

-   **Tokenization:** The `spaCy` library is used to tokenize the German and English sentences. Special tokens, `<sos>` (start of sentence) and `<eos>` (end of sentence), are added to each sequence.

-   **Vocabulary Building:** Custom vocabularies are created for both the source (German) and target (English) languages. Words with a frequency below a minimum threshold are mapped to an `<unk>` (unknown) token. A `<pad>` token is also included for batching.

-   **Data Loading:** The tokenized text is converted into numerical indices. A custom `collate_fn` is used with the PyTorch `DataLoader` to pad all sequences in a batch to the length of the longest sequence, enabling efficient batch processing.

### 2\. Model Architecture: Encoder-Decoder with LSTMs

The model follows the standard Encoder-Decoder architecture:

-   **Encoder:** A multi-layer LSTM network that processes the input German sentence one token at a time. Its purpose is to encode the entire sentence into a fixed-size context vector, which is represented by the final hidden and cell states of the LSTM. This context vector aims to capture the semantic meaning of the source sentence.

-   **Decoder:** Another multi-layer LSTM network that takes the context vector from the Encoder as its initial hidden and cell states. It generates the English translation word by word. At each step, it takes the previously generated word as input and predicts the next word in the sequence.


### 3\. Training with PyTorch Lightning

The entire pipeline is managed within a `Seq2seq` `LightningModule`, which cleanly separates the model's logic from the training code.

-   **Training Loop:** The `training_step` defines how the model processes a batch of data. It uses **teacher forcing**, a technique where the decoder is fed the ground-truth target word from the previous time step instead of its own prediction. This is done with a certain probability (e.g., 50%) to stabilize training.

-   **Validation Loop:** The `validation_step` evaluates the model's performance on a validation set. During this phase, teacher forcing is turned off (`teacher_force_ratio=0`) to assess how the model performs on its own.

-   **Loss Function:** The model is trained using `CrossEntropyLoss`. The loss calculation is configured to ignore the index of the `<pad>` token, so the model is not penalized for predicting padding.

-   **Optimizer:** The `Adam` optimizer is used to update the model's weights.

Tensor boards 
---------------


<img width="707" height="506" alt="Screenshot 2025-08-28 at 01 55 46" src="https://github.com/user-attachments/assets/6bbc1636-e541-40fd-91bb-d53471ed0847" />
<img width="786" height="501" alt="Screenshot 2025-08-28 at 01 55 53" src="https://github.com/user-attachments/assets/0f242a40-fcfe-4f16-966e-2e93755b8f02" />
<img width="785" height="492" alt="Screenshot 2025-08-28 at 01 56 02" src="https://github.com/user-attachments/assets/5bb24027-ee1c-42ff-abbc-6cf62f203f11" />
<img width="785" height="490" alt="Screenshot 2025-08-28 at 01 56 11" src="https://github.com/user-attachments/assets/f4fb64c0-5c95-49c7-9659-164cae134bbd" />
<img width="784" height="492" alt="Screenshot 2025-08-28 at 01 56 21" src="https://github.com/user-attachments/assets/045b4a80-1254-44d9-be6c-85ecacbf41b1" />




