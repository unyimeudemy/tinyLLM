import styled from "styled-components"


const Container = styled.div``
const Body = styled.div``


const SentMessage = ({message}:any) => {


  return (
    <Container className="w-full h-auto flex justify-end 
    ">
        <Body
            className="w-auto max-w-[70%] p-3 px-5 bg-white
                    text-stone-700  font-medium 
                    rounded-2xl text-left"
            >
            {message.replace("Human:", "")}
        </Body>
    </Container>
  )
}

export default SentMessage
