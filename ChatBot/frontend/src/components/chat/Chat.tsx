import {useState} from 'react'
import "./chat.scss"
import axios from "axios"
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css'
import {MainContainer,ChatContainer,MessageList,Message,MessageInput,TypingIndicator} from '@chatscope/chat-ui-kit-react'
import AxiosInstance from "./Axios.tsx";




function Chat() {
    const [typing,setTyping]=useState(false);
    const [messages,setMessages]=useState([
    {
        message:"Hello, I am chat gpt",
        sender:"ChatGPT",
        direction:"incoming"
    }
])




function WriteMessage(message) {
        const newMessage={
            message:message,
            sender:"user",
            direction:"outgoing"
        }
        const newMessages=[...messages,newMessage]
        setMessages(newMessages);
        setTyping(true);
        console.log(message);
    AxiosInstance.post(`openaiapi/messages/`,  {
            message:message,
        }
    ).then(res=>{
        console.log(res.data.answer);
        const newApiMessage={
                message:res.data.answer,
                sender:"ChatGPT",
                direction:"incoming"
        }
        const newApiMessages=[...messages,newMessage,newApiMessage]
        setTyping(false);
        setMessages(newApiMessages);
        console.log(messages);

    })

}



    return (

        <div className="chatCont">
            <MainContainer>
               <ChatContainer>

                   <MessageList
                       scrollBehavior="smooth" typingIndicator={typing? <TypingIndicator content="chat gpt is typing"/>:null}>
                       {messages.map((message,index)=>{
                           return <Message key={index} model={message}/>
                       })}
                   </MessageList>
                   <MessageInput placeholder='Type messsage here' onSend={WriteMessage}/>

               </ChatContainer>
            </MainContainer>

        </div>
    )
}

export default Chat