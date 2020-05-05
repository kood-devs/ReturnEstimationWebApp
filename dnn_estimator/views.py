import os

from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views.generic import TemplateView, ListView, DetailView, CreateView, DeleteView

from .models import LearningModel
from .dnn_estimator import *


class TopForm(TemplateView):
    template_name = 'top.html'
    model = LearningModel


class MainForm(ListView):
    template_name = 'main.html'
    model = LearningModel


class DetailForm(DetailView):
    template_name = 'detail.html'
    model = LearningModel


class SetParam(CreateView):
    template_name = 'set_param.html'
    model = LearningModel
    fields = ('title', 'train_start', 'train_end',
              'test_start', 'test_end', 'epoch', 'batch_size')
    success_url = reverse_lazy('main')


class DeleteParam(DeleteView):
    template_name = 'delete_param.html'
    model = LearningModel
    success_url = reverse_lazy('main')


def learn_dnn_model(request, pk):
    # learn individual model
    params = LearningModel.objects.get(pk=pk)
    result = learn_dnn(
        params.train_start, params.train_end, params.test_start, params.test_end, params.epoch, params.batch_size, params.title)
    params.train_acc, params.test_acc = result
    params.images = '{}.jpg'.format(params.title)
    # os.remove('media/{}.jpg'.format(params.title))
    params.save()

    return redirect('main')
    # return render(request, 'detail.html')  # 本当は学習完了後に結果が入力されたdetail画面に移動したい... -> 何故か結果が空に
