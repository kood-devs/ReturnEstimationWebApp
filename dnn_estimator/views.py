from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import ListView, CreateView, DeleteView
from .models import LearningModel


class MainForm(ListView):
    template_name = 'main.html'
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
