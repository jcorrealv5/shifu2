from django.urls import path
from . import views
urlpatterns = [
    path('DeteccionRostros', views.DeteccionRostros, name='DeteccionRostros'),
    path('DetectarRostros', views.DetectarRostros, name='DetectarRostros')
]