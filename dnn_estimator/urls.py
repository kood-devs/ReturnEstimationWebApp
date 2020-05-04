from django.urls import path
from .views import TopForm, MainForm, DetailForm, SetParam, DeleteParam, learn_dnn_model

urlpatterns = [
    path('', TopForm.as_view(), name='top'),
    path('main/', MainForm.as_view(), name='main'),
    path('detail/<int:pk>', DetailForm.as_view(), name='detail'),
    path('set_param/', SetParam.as_view(), name='set_param'),
    path('delete_param/<int:pk>', DeleteParam.as_view(), name='delete_param'), 
    path('learn_dnn_model/<int:pk>', learn_dnn_model, name='learn_dnn_model'),    
]
