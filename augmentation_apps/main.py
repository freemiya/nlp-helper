# %%
import torch
import string
import copy
import random

from transformers import BertTokenizer, BertForMaskedLM
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

from transformers import BartTokenizer, BartForConditionalGeneration
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

from transformers import ElectraTokenizer, ElectraForMaskedLM
electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()

top_k = 10

model_dict = {'bert':(bert_tokenizer, bert_model),'xlmroberta':(xlmroberta_tokenizer, xlmroberta_model),
              'bart':(bart_tokenizer, bart_model),'electra': (electra_tokenizer, electra_model),
              'roberta': (roberta_tokenizer, roberta_model)}

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return tokens[:top_clean]

def decode_multiple_masks(tokenizer, predict, mask_positions, fillinblank_sentences, top_clean):
    # Place the predictions in the sentence
    for mask_idx in mask_positions:
        predicted_words = decode(tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

        for idx in range((len(predicted_words))):
            fillinblank_sentences[idx][mask_idx] = predicted_words[idx]
    return fillinblank_sentences


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_positions = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()

    return input_ids, mask_positions

def get_predictions(modelname, text_sentence, top_clean=5):
    """
    Psuedocode:
        Get the masked sentence.
        Encode it & pass it through model
        Now, decode at each position.
    """
    results = dict()
    tokenizer, model = model_dict[modelname]
    input_ids, mask_positions = encode(tokenizer, text_sentence)
    fillinblank_sentences = []

    """
    Create fill in the blanks to fill up 
    with predictions in the later stage
    """
    for _ in range(top_clean):
        # 'Ġ' is added to words by tokenizers in the case of RoBERTa, XLMRoBERTa, Bart.
        # So, removing it.
        predicted_sentence = [tokenn.replace('Ġ','') for tokenn in tokenizer.convert_ids_to_tokens(input_ids[0])]
        fillinblank_sentences.append(predicted_sentence)

    with torch.no_grad():
        predict = model(input_ids)[0]

    # Place the predictions in the sentence
    results = decode_multiple_masks(tokenizer, predict, mask_positions, fillinblank_sentences, top_clean)
    return results, mask_positions

def mask_sentence(tokens, tokenizer, style='bert'):
    """
    Reference: https://github.com/huggingface/transformers/blob/f9cde97b313c3218e1b29ea73a42414dfefadb40/examples/lm_finetuning/simple_lm_finetuning.py#L267

    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction

    Replace some with <mask>, some with random words.
    """
    output_label = []
    # mask_positions = [] # For storing the position where words are changed

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "<mask>"

            # 10% randomly change token to random token
            # elif prob < 0.9:
            #     tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            
            
            try:
                output_label.append(tokenizer.convert_tokens_to_ids(token))
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.convert_tokens_to_ids("[UNK]"))
                print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label 



def prepare_input( tokenizer: object, style:str, text: str):
    """
    Psuedocode:
    * Tokenize the sentence.
    * Send it to mask_sentence() -> Get the masked sentence and a
      list. This list will have Indices numbers for positions
      where masking is done.
    * Convert sentences to ids.

    Input : 
        :param -> tokenizer (transformer's object)
        :style -> Masking style
        :text  -> Sentence
    Return: 
        :param -> Masked sentence, 
        :param -> Mask labels
    """
    # Tokenize input
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text_ = copy.copy(tokenized_text)
    masktokenized_text, mask_labels = mask_sentence(tokenized_text_, tokenizer, style=style)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(masktokenized_text)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    segments_ids = [0]* len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    input_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return masktokenized_text, mask_labels

def get_mask_predictions(sentence: str, modelname: str, num_sents= 5):
    """
    Psuedocode:
    For each model,
        * Mask the sentence
        * Stitch it to a normal sentence (undo BPE). TODO
        * Send it through the model
    """
    masktokenized_text = ''
    results = dict()

    while '<mask>' not in masktokenized_text:
        masktokenized_text, _ = prepare_input(tokenizer=model_dict[modelname][0], \
                                                        style=f'{modelname}', text=sentence)
        masktokenized_text = [tokenn.replace('Ġ','') for tokenn in masktokenized_text]
        # 'Ġ' is added to words by tokenizers in the case of RoBERTa, XLMRoBERTa, Bart.
        # So, removing it.

    masked_sentence = ' '.join(masktokenized_text)

    # Adding 1 for every position since <CLS> & <SEP> are added at encode stage.
    pred_sents, mask_positions = get_predictions(modelname, masked_sentence, top_clean=num_sents)

    #Add html tags to it
    for idx, sent in enumerate(pred_sents):
        for pos in mask_positions:
            pred_sents[idx][pos] =  "<p style='color:blue; display:inline'><b>" + pred_sents[idx][pos] + "</b></p>"
    
    pred_sentences = [' '.join(sent[1:-1]) for sent in pred_sents]  #remove the cls and sep tag
    results[modelname] = "<br>".join(pred_sentences)
    return results