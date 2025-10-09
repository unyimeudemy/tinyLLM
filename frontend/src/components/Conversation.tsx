import styled from "styled-components"
import Messages from "./Messages"
import ChatInput from "./ChatInput"
import { useState } from "react"


const Container = styled.div``

const Conversation = () => {
  const [newMessage, setNewMessage] = useState("")

  return (
    <Container
    className="h-full w-full p-2 flex flex-col items-center "
  >
    <Messages
      newMessage={newMessage}
    />
    <ChatInput
      setNewMessage={setNewMessage}
    />
  </Container>
  )
}

export default Conversation
