import styled from "styled-components"
import addIcon from "../assets/add_icon.png"
import sendMessageIcon from "../assets/send_message_icon.png"
import { useState } from "react"
import axios from "axios"

const Container = styled.div``
const InputField = styled.textarea`
  border: none;
  outline: none;

  &:focus {
    outline: none;
    box-shadow: none;
  }
`
const Buttons = styled.div``
const AddIcon = styled.img``
const SendMessageIcon = styled.img``


const ChatInput = ({setNewMessage}:any) => {
  const [userText, setUserText] = useState("")

  const handleUserText = (e: any) => {
    e.preventDefault()
    setUserText(e.target.value)
  }

  const sendMessage = async () => {
    if (!userText.trim()) return;
    setNewMessage("Human:"+userText)


    try{
      const res = await axios.post(`http://46.62.212.74:8080/api/infer`, {
        message: userText
      })
      console.log("res data", res.data.message)
      setNewMessage(res.data.message)
      setUserText("")
    }catch(error){
      console.error('Error sending message:', error);
    }finally{
      setUserText("")
    }
  }

  const handleKeyDown = (e: any) => {
    if (e.key == "Enter"){
      e.preventDefault()
      sendMessage()
      setUserText("")
    }

  }

  return (
    <Container
      className="w-[55%] h-[120px]  mt-auto
      bg-white rounded-[10px] border-[1px] border-[#b3b3b3]
      p-2 flex flex-col 
      "
      style={{
        boxShadow: '3px 8px 7px rgba(0, 0, 0, 0.1)'
      }}
  >
    <InputField
        className="w-full h-[80%]  resize-none
        font-semibold text-[#333333] bg-white
        p-2
        "
        value={userText}
        onChange={handleUserText}
        onKeyDown={handleKeyDown}
    />
    
    <Buttons
    className="w-full h-[40px] 
    flex flex-row items-center justify-between
    "
    >
        <AddIcon
            src={addIcon}
            className="h-[30px] w-[30px] hover:scale-150 transition-transform duration-300 cursor-pointer"
        />
        <SendMessageIcon
            src={sendMessageIcon}
            onClick={sendMessage}
            className="h-[19px] w-[19px] mr-2  hover:scale-150 transition-transform duration-300 cursor-pointer"
        />
    </Buttons>
  </Container>
  )
}

export default ChatInput
