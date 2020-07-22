# %%
import torch
import string
import copy
import random

from transformers import BertTokenizer, BertForMaskedLM
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

# from transformers import XLNetTokenizer, XLNetLMHeadModel
# xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# xlnet_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased').eval()

# from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
# xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
# xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

# from transformers import BartTokenizer, BartForConditionalGeneration
# bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

# from transformers import ElectraTokenizer, ElectraForMaskedLM
# electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
# electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

# from transformers import RobertaTokenizer, RobertaForMaskedLM
# roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()

top_k = 10

model_dict = {'bert':(bert_tokenizer, bert_model)}

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return tokens[:top_clean]

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    return input_ids

def get_predictions(style, text_sentence, mask_positions, top_clean=5):
    """
    Psuedocode:
        Get the masked sentence.
        Encode it & pass it through model
        Now, decode at each position.
    """
    results = dict()
    # ========================= BERT =================================
    bert_tokenizer, bert_model = model_dict[style]
    input_ids = encode(bert_tokenizer, text_sentence)
    new_sentences = []
    for i in range(top_clean): 
        predicted_sentence = ('[CLS] '+text_sentence+' [SEP]').strip().split()
        new_sentences.append(predicted_sentence)
    

    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    
    # Place the predictions in the sentence
    for mask_idx in mask_positions:
        predicted_words = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
        for idx in range((len(predicted_words))):
            new_sentences[idx][mask_idx] = predicted_words[idx]

    results[style] = new_sentences    
    # # ========================= XLNET LARGE =================================
    # input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
    # perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    # perm_mask[:, :, mask_idx] = 1.0  # Previous tokens don't see last token
    # target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    # target_mapping[0, 0, mask_idx] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

    # with torch.no_grad():
    #     predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
    # xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= XLM ROBERTA BASE =================================
    # input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = xlmroberta_model(input_ids)[0]
    # xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= BART =================================
    # input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = bart_model(input_ids)[0]
    # bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= ELECTRA =================================
    # input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = electra_model(input_ids)[0]
    # electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    # # ========================= ROBERTA =================================
    # input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = roberta_model(input_ids)[0]
    # roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return results
    # return {'bert': bert,
    #         'xlnet': xlnet,
    #         'xlm': xlm,
    #         'bart': bart,
    #         'electra': electra,
    #         'roberta': roberta}

def mask_sentence(tokens, tokenizer, style='bert'):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction

    Replace some with <mask>, some with random words.
    """
    output_label = []
    mask_positions = [] # For storing the position where words are changed

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "<mask>"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # Let's store the position where words are changed
            mask_positions.append(i)
            # append current token to output (we will predict these later)
            
            
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, mask_positions, output_label 



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
    masktokenized_text, mask_positions, mask_labels = mask_sentence(tokenized_text_, tokenizer, style=style)

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(masktokenized_text)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    segments_ids = [0]* len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    input_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
                
    # return tokenized_text, masktokenized_text, input_tensor,\
    #         mask_labels, segments_tensors
    return masktokenized_text, mask_positions, mask_labels

def get_mask_predictions(sentence: str, num_sents= 5):
    """
    Psuedocode:
    For each model,
        * Mask the sentence
        * Stitch it to a normal sentence (undo BPE). TODO
        * Send it through the model
    """
    masktokenized_text = ''
    results = dict()
    for style in model_dict.keys():
        while '<mask>' not in masktokenized_text:
            masktokenized_text, mask_positions, _ = prepare_input(tokenizer=model_dict[style][0], \
                                                            style=f'{style}', text=sentence)
            
        masked_sentence = ' '.join(masktokenized_text)

        # Adding 1 for every position since <CLS> & <SEP> are added at encode stage.
        mask_positions =  [(pos+1) for pos in mask_positions]        
        results = get_predictions(style, masked_sentence, mask_positions, top_clean=num_sents)
        pred_sents = results[style]

        #Add html tags to it
        for idx, sent in enumerate(pred_sents):
            for pos in mask_positions:
                pred_sents[idx][pos] =  "<p style='color:blue; display:inline'><b>" + pred_sents[idx][pos] + "</b></p>"

        print(pred_sents)        
        pred_sentences = [' '.join(sent[1:-1]) for sent in pred_sents]  #remove the cls and sep tag
        results[style] = "<br>".join(pred_sentences)

    return results