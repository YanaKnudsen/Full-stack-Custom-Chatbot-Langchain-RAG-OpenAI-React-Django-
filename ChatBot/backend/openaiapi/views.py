from django.shortcuts import render
from rest_framework import views,status,generics,permissions,viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import MessagesSerializer
from .models import Messages



class MessageCreate(APIView):
  queryset=Messages.objects.all()
  serializer_class=MessagesSerializer

  def post(self,request,format=None):
         serializer=self.serializer_class(data=request.data)
         if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,status=status.HTTP_201_CREATED)
         else:
            return Response(serializer.errors,status=status.HTTP_400_BAD_REQUEST)

  def get(self,request,format=None):
           queryset=Messages.objects.all()
           serializer=MessagesSerializer(queryset,many=True) #what does many true means
           return Response(serializer.data,status=status.HTTP_200_OK)

