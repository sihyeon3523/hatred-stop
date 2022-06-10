from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.conf import settings
from . import inference_bert


# Create your views here.

def index(request):
    return render(request, 'main_app/index.html', {})


def aboutus(request):
    return render(request, 'main_app/aboutus.html', {})


def chathome(request):
    return render(request, 'main_app/chathome.html',{})


def dl_emotion(request):

    target_sentence = request.POST['target_sentence']
    tokenizer = settings.TOKENIZER_KOBERT
    model = settings.MODEL_KOBERT

    result_prob, result_bert = inference_bert.predict_sentiment(target_sentence, tokenizer, model)

    if (result_bert=='혐오'):
        result = 0  # 혐오표현
    else:
        result = 1  # 아님

    context = {'target_sentence':target_sentence, 'result':result, 'result_prob':result_prob}

    return render(request, "main_app/chatresult.html", context)
