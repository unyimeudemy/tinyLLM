import styled from "styled-components"
import subsribeButton from "../assets/subscribe-button.webp"


const Container = styled.div``
const SubscribeButton = styled.img``


const NavBar = () => {
  return (
    <Container
      className="w-full h-[80px] flex flex-row items-center border-b border-[#b3b3b3]"
    >
      {/* Centered text */}
      <div className="flex-1 flex justify-center mt-5">
        <div className="text-[#412B6B] text-xl font-extrabold ml-[100px]">
          TinyLLM series - v1
        </div>
      </div>

      {/* Right aligned button */}
      <a
        href="https://www.youtube.com/@unyime_udoh"
        target="_blank"
        rel="noopener noreferrer"
      >
        <SubscribeButton
          className="w-[130px] mr-5 hover:cursor-pointer mt-6"
          src={subsribeButton}
        />
      </a>
    </Container>
  )
}


export default NavBar
