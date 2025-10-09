import styled from "styled-components"

const Container = styled.div``


const ReceivedMessage = ({message}:any) => {


    
  return (
    <Container className="
    w-full h-auto flex  text-stone-700  font-medium
    mt-7 mb-[65px] text-left 
    ">
    {message}
    </Container>
  )
}

export default ReceivedMessage


