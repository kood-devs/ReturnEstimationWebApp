from django.urls import path
from .views import MainForm, SetParams, DeleteParams, learn_dnn_model

urlpatterns = [
    path('main/', MainForm.as_view(), name='main'),
    path('set_params/', SetParams.as_view(), name='set_params'),
    path('delete_params/<int:pk>', DeleteParams.as_view(), name='delete_params'), 
    path('learn_dnn_model/<int:pk>', learn_dnn_model, name='learn_dnn_model'),    
]
