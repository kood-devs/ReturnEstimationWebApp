from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views.generic import ListView, DetailView, CreateView, DeleteView
from .models import LearningModel
from .dnn_estimator import *


class MainForm(ListView):
    template_name = 'main.html'
    model = LearningModel


class DetailForm(DetailView):
    template_name = 'detail.html'
    model = LearningModel


class SetParams(CreateView):
    template_name = 'set_params.html'
    model = LearningModel
    fields = ('title', 'train_start', 'train_end',
              'test_start', 'test_end', 'epoch', 'batch_size')
    success_url = reverse_lazy('main')


class DeleteParams(DeleteView):
    template_name = 'delete_params.html'
    model = LearningModel
    success_url = reverse_lazy('main')


def learn_dnn_model(request, pk):
    # learn individual model
    params = LearningModel.objects.get(pk=pk)
    print(params)
    result = learn_dnn(
        params.train_start, params.train_end, params.test_start, params.test_end, params.epoch, params.batch_size)
    params.train_acc, params.test_acc = result
    params.save()
    return redirect('main')
    # return render(request, 'main.html')
