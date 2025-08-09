from django.urls import path
from . import views
urlpatterns = [
    path('ClasObjetos', views.ClasObjetos, name='ClasObjetos'),
    path('ClasificarObjeto', views.ClasificarObjeto, name='ClasificarObjeto')
]