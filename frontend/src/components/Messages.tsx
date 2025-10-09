
import styled from "styled-components"
import ReceivedMessage from "./ReceivedMessage"
import SentMessage from "./SentMessage"
import { useEffect, useRef, useState } from "react"

const Container = styled.div.attrs({
  className:
    "w-[55%] h-[570px] flex flex-col gap-2 overflow-y-auto scrollbar-hide"
})``;

const Messages = ({ newMessage }: any) => {
  const [messagesList, setMessagesList] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (newMessage) {
      setMessagesList(prev => [...prev, newMessage]);
    }
  }, [newMessage]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messagesList]);

  return (
    <Container>
      {messagesList.map((message: string, index: number) =>
        message.startsWith("Human:") ? (
          <SentMessage key={index} message={message} />
        ) : (
          <ReceivedMessage key={index} message={message} />
        )
      )}
      <div ref={messagesEndRef} /> 
    </Container>
  );
};

export default Messages;

