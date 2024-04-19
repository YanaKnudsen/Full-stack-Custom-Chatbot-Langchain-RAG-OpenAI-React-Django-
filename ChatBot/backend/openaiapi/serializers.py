from rest_framework import serializers
from .models import Messages
from .utils import chat_gpt


class MessagesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Messages
        fields = ('id', 'message','answer')
       # fields = '__all__' try it: serializers all the fields
        extra_kwargs={
              "answer":{"read_only":True}
              }

    def create(self,validated_data):
           gptans=Messages(**validated_data)
           answer=chat_gpt(validated_data["message"])
           gptans.answer=answer
           gptans.save()
           return gptans
