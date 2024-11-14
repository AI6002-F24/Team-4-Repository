These 5 notebooks are used to split the dataset into 5 groups and fine-tune the gpt model to improve it's adaptability and performance to the scenario.

1. GPT_1 : Trained on the most common ingredients
2. GPT_2 : 0th Row - 57909th Row
2. GPT_3 : 57909th Row - 115818th Row
3. GPT_4 : 115818th Row - 173727th Row
4. GPT_5 : 173727th Row - 231637 Row

The model parameters for each of these notebooks is given in: 
'''
training_args = TrainingArguments(
    output_dir='./results',
    warmup_steps=1000,
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.001,
    per_device_train_batch_size=4,  # Adjust based on your GPU's capacity
    save_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True
)
'''
