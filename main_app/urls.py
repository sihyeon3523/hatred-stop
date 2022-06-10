from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static


app_name = 'main_app'

urlpatterns = [
    path('', views.index, name='index'),

    path('aboutus/', views.aboutus, name='aboutus'),
    path('chathome/', views.chathome, name='chathome'),
    path('chathome/dl_emotion/', views.dl_emotion, name='dl_emotion'),
]
