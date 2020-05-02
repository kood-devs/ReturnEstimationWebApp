from django.urls import path
from .views import MainForm, SetParams, DeleteParams

urlpatterns = [
    path('main/', MainForm.as_view(), name='main'),
    path('set_params/', SetParams.as_view(), name='set_params'),
    path('delete_params/<int:pk>', DeleteParams.as_view(), name='delete_params'), 
]
