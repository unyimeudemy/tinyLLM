import styled from "styled-components"
import Conversation from "./Conversation"

const Wrapper = styled.div``




const DashBoard = () => {


  return (
    <div className="h-full w-full  rounded-[10px] bg-[#f2f2f2]"
    >
        <Wrapper
          className="w-full h-full p-5  rounded-[10px]
          flex flex-row  
          "
        >
          <Conversation/>
        </Wrapper>
    </div>
  )
}

export default DashBoard
