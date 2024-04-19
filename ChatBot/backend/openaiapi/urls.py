from .views import MessageCreate
from django.urls import path


urlpatterns = [
    path('messages/',MessageCreate.as_view(),name='messages'),
]