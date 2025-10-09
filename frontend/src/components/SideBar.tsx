import styled from 'styled-components'
import homeicon from "../assets/homeIcon.png"
import chats from "../assets/chats.png"
import settings from "../assets/settings.png"
import profile from "../assets/profileIcon.webp"





const Container = styled.div``
const HomeIcon = styled.img``
const ChatsIcon = styled.img``
const SettingsIcon = styled.img``
const ProfileIcon = styled.img``


const SideBar = () => {
  return (
    <Container className='h-full w-[6%] flex flex-col items-center  border-r-2 border-[#b3b3b3]
      pt-[50px] gap-[40px]
    '>
      <HomeIcon
        src={homeicon}
        className='h-[40px] w-[40px] hover:cursor-pointer'
      />
      <ChatsIcon
        src={chats}
        className='h-[40px] w-[40px] hover:cursor-pointer'
      />

      <SettingsIcon
        src={settings}
        className='h-[40px] w-[40px] hover:cursor-pointer'
      />
      <ProfileIcon
        src={profile}
        className='h-[60px] w-[60px] hover:cursor-pointer'
      />
      
    </Container>
  )
}

export default SideBar
