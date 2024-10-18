from django.urls import path
from . import views

app_name = 'apis'

urlpatterns = [
    path('recognize-dog-breed', views.recognize_dog_breed, name='recognize_dog_breed'),

]