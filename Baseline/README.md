A simple baseline for the DialogSum Challenge using BART-Large model. It might help you quickly get familar with the task.

## Usage
````sh
A simple baseline for the DialogSum Challenge using BART-Large model. It might help you quickly get familar with the task.

## Usage
```sh
conda create -n dialogsumm python=3.7
conda activate dialogsumm
pip install -r requirements.txt
# mind the cuda version suitable to your GPU

# you may need to manualy download the files to run nltk tokenizer, this is very easy following https://www.nltk.org/data.html

python bart_train.py
# the code will output result on test set with regard to summary1, you can change it in the code.
# the parameters are not carefully tuned
# you may need to extra deal with the "#" symbol around the person name.

```
````

I got following result the on test set when comparing to the summary1

'eval_rouge1': 45.9458, 'eval_rouge2': 21.3612, 'eval_rougeL': 38.7189,