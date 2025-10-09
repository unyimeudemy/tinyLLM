import styled from "styled-components"
import { useNavigate } from "react-router-dom"

const Container = styled.div`
`



const Setting = () => {
    const navigate = useNavigate()


  return (
    <Container>
        <div className=" text-yellow font-bold text-[60px]">
            Setting
        </div>
        <button 
            className=" h-[50px] w-[200px] bg-red text-black border border-blue-200"
            onClick={() => navigate("/")}
        >Back</button>
    </Container>

  )
}

export default Setting
