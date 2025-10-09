// import React from 'react'
import styled from "styled-components"
import NavBar from "../components/NavBar"

import DashBoard from "../components/DashBoard"

const Container = styled.div``
const Wrapper = styled.div``
const Body = styled.div``



const Home = () => {


  return (
    <Container className="h-full flex flex-col w-screen " >
      <NavBar/>
      <Wrapper className="w-full h-full flex flex-row ">
        <Body className="w-full h-full  bg-white ">
            <DashBoard/>
        </Body>
      </Wrapper>
    </Container>
  )
}

export default Home
