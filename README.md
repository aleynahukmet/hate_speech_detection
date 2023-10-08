# hate_speech_detection
Comparison of BERT-Base, BERTweet, RoBERTa, DistilBERT for hate speech detection 

Before we fine-tune pretrained BERT models, we download  🤗 Hugging Face datasets and prepare them for training.

## DATASETS 

1. [hatexplain](https://huggingface.co/datasets/hatexplain)
   
   First of all, I labeled the texts, taking into account the majority vote of annotators. Later on, I reduced the number of labels from three (hate, offensive, normal) to two, merging 'hate' and 'offensive' into one common category of 'hate'.
3. [hate_speech18](https://huggingface.co/datasets/hate_speech18)

   This dataset contained four labels,  hate, noHate, relation (sentence in the post doesn't contain hate speech on their own, but combination of serveral sentences does) or idk/skip (sentences that are not written in English or that don't contain information as to be classified into hate or noHate). I dropped the texts with 'relation' and 'idk/skip label' labels.
5. [goodwin278/labelled_hatespeech](https://huggingface.co/datasets/goodwin278/labelled_hatespeech)

   This dataset includes texts from 'Reddit', 'Twitter', '4Chan'. I included this dataset as is because it's already organized to two categories 'hate' and 'n-hate'.

7. [christinacdl/binary_hate_speech](https://huggingface.co/datasets/christinacdl/binary_hate_speech)

   This dataset is also a binary classed dataset, and it contains 'NOT_OFF_HATEFUL_TOXIC' and 'OFF_HATEFUL_TOXIC' labels. I mapped these labels to 0 and 1.
   
9. [tweets_hate_speech_detection](https://huggingface.co/datasets/tweets_hate_speech_detection)

   This one is a dataset of labeled tweets, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist.

After bringing all the datasets to the same format, we combined these datasets. And then, I created a new dataset with train_test_split:

```
DatasetDict({
    train: Dataset({
        features: ['text', 'labels'],
        num_rows: 82976
    })
    test: Dataset({
        features: ['text', 'labels'],
        num_rows: 20744
    })
}
```

## FINE-TUNING

As models like BERT do not expect text as direct input but rather input_ids, etc., we tokenize the text using the tokenizer. I used the AutoTokenizer API, which will automatically load the appropriate tokenizer based on the checkpoint on the hub. Therefore, for every model I trained, I defined the checkpoint of that model and let the AutoTokenizer API take care of the rest. I also included text padding and a truncation strategy to handle any variable sequence lengths. I used the Datasets map method to apply a preprocessing function over the entire dataset. 
```
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
def tokenize_data(example):
  return tokenizer(example['text'],padding='max_length',truncation=True)
tokenized_dataset = new_df.map(tokenize_data,batched=True)

```

Later on, for each model, I defined a model with a randomly initialized classification head (linear layer) on top. I fine-tuned this head. Since the dataset we want to fine-tune the models on has two labels, we initialized the model with the number of two labels.

```
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```

After I trained the models for the first time, I saw all of the models were overfitting on the training dataset. Unless we don't have massive datasets, transformer models can really lean towards overfitting since they have millions of parameters. By freezing some parameters, we can mitigate the risk of overfitting. Here, I made trainable the last 20 weights, which encorporate the last layer of the encoder stack, and froze the earlier layers.

Next, I created a TrainingArguments class, which contains all the hyperparameters:

```
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir='./results/bert-base',                          
    run_name = 'bert-base',
    evaluation_strategy="steps",      
    eval_steps=300,
    save_strategy="steps",           
    save_total_limit=5,
    save_steps=300,                                  
    load_best_model_at_end=True,                     
    num_train_epochs=4,                              
    per_device_train_batch_size=4,                   
    per_device_eval_batch_size=4,                    
    learning_rate=1e-4,                              
    weight_decay=0.01,                               
    warmup_steps=500,                                
    logging_steps=100,                               
    gradient_accumulation_steps=16,                  
)
```

## MODEL RESULTS

### bert-base

**eval/loss**|**train/train\_loss**|**eval/accuracy**
:-----:|:-----:|:-----:
0.316609|0.341321|0.856826

### bertweet-base

**eval/loss**|**train/train\_loss**|**eval/accuracy**
:-----:|:-----:|:-----:
0.315942|0.334273|0.861502

### roberta-base

**eval/loss**|**train/train\_loss**|**eval/accuracy**
:-----:|:-----:|:-----:
0.291966|0.3294	|0.86854

### distilbert

**eval/loss**|**train/train\_loss**|**eval/accuracy**
:-----:|:-----:|:-----:
0.307688|0.32511	|0.865214


