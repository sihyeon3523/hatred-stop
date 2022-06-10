import re
import numpy as np


def predict_sentiment(sentence, tokenizer, model):

    SEQ_LEN = 300 # 최대 token 개수 이상의 값으로 임의로 설정

    # Tokenizing / Tokens to sequence numbers / Padding
    encoded_dict = tokenizer.encode_plus(text=re.sub("[^\s0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]", "", sentence),
                                         padding='max_length',
                                         truncation = True,
                                         max_length=SEQ_LEN) # SEQ_LEN == 300

    token_ids = np.array(encoded_dict['input_ids']).reshape(1, -1) # shape == (1, 128) : like appending to a list
    token_masks = np.array(encoded_dict['attention_mask']).reshape(1, -1)
    token_segments = np.array(encoded_dict['token_type_ids']).reshape(1, -1)

    new_inputs = (token_ids, token_masks, token_segments)

    # Prediction
    prediction = model.predict(new_inputs)
    predicted_probability = np.round(np.max(prediction) * 100, 2) # ex) [[0.0125517 0.9874483]] -> round(0.9874483 * 100, 2) -> round(98.74483, 2) -> 98.74
    predicted_class = ['혐오','비혐오'][np.argmax(prediction, axis=1)[0]]# ex) ['부정', '긍정'][[1][0]] -> ['부정', '긍정'][1] -> '긍정'
    #result = "{}% 확률로 {} 텍스트입니다.".format(predicted_probability, predicted_class)

    return predicted_probability,predicted_class
